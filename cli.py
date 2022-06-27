# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""
import json
import os
from typing import Tuple
import warnings

import torch
import wandb
from knockknock import slack_sender

import log
import pet
from pet.argparsing import parser
from pet.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from pet.utils import eq_div
from pet.wrapper import SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig

logger = log.get_logger("root")
#webhook_url = open("slack_webhook.txt").read()


def load_pet_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        wrapper_type=args.wrapper_type,
        task_name=args.task_name,
        label_list=args.label_list,
        max_seq_length=args.pet_max_seq_length,
        verbalizer_file=args.verbalizer_file,
        cache_dir=args.cache_dir,
    )

    train_cfg = pet.TrainConfig(
        device=args.device,
        per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
        per_gpu_unlabeled_batch_size=args.pet_per_gpu_unlabeled_batch_size,
        n_gpu=args.n_gpu,
        num_train_epochs=args.pet_num_train_epochs,
        max_steps=args.pet_max_steps,
        min_steps=args.pet_min_steps,
        gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lm_training=args.lm_training,
        logging_steps=args.logging_steps,
        logging_number=args.logging_number,
        alpha=args.alpha,
        local_rank=args.local_rank,
    )

    eval_cfg = pet.EvalConfig(
        device=args.device,
        n_gpu=args.n_gpu,
        metrics=args.metrics,
        per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
        decoding_strategy=args.decoding_strategy,
        priming=args.priming,
        local_rank=args.local_rank,
    )

    return model_cfg, train_cfg, eval_cfg


def load_sequence_classifier_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for a regular sequence classifier from the given command line
    arguments. This classifier can either be used as a standalone model or as the final classifier for PET/iPET.
    """
    model_cfg = WrapperConfig(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER,
        task_name=args.task_name,
        label_list=args.label_list,
        max_seq_length=args.sc_max_seq_length,
        verbalizer_file=args.verbalizer_file,
        cache_dir=args.cache_dir,
    )

    train_cfg = pet.TrainConfig(
        device=args.device,
        per_gpu_train_batch_size=args.sc_per_gpu_train_batch_size,
        per_gpu_unlabeled_batch_size=args.sc_per_gpu_unlabeled_batch_size,
        n_gpu=args.n_gpu,
        num_train_epochs=args.sc_num_train_epochs,
        max_steps=args.sc_max_steps,
        min_steps=args.sc_min_steps,
        temperature=args.temperature,
        gradient_accumulation_steps=args.sc_gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        logging_number=args.logging_number,
        max_grad_norm=args.max_grad_norm,
        use_logits=args.method != "sequence_classifier",
        local_rank=args.local_rank,
    )

    eval_cfg = pet.EvalConfig(
        device=args.device,
        n_gpu=args.n_gpu,
        metrics=args.metrics,
        per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size,
        local_rank=args.local_rank,
    )

    return model_cfg, train_cfg, eval_cfg


def load_ipet_config(args) -> pet.IPetConfig:
    """
    Load the iPET config from the given command line arguments.
    """
    ipet_cfg = pet.IPetConfig(
        generations=args.ipet_generations,
        logits_percentage=args.ipet_logits_percentage,
        scale_factor=args.ipet_scale_factor,
        n_most_likely=args.ipet_n_most_likely,
    )
    return ipet_cfg


#@slack_sender(webhook_url=webhook_url, channel="Teven")
def main():
    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))

    # Setup CUDA, GPU & distributed training
    if args.local_rank != -1:
        args.n_gpu = 1
        args.device = args.local_rank if torch.cuda.is_available() and not args.no_cuda else "cpu"
    else:
        args.n_gpu = torch.cuda.device_count()
        args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    if args.verbalizer_file is not None:
        args.verbalizer_file = args.verbalizer_file.replace("[TASK_NAME]", args.task_name)
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    wandb_initalized = False

    if args.local_rank != -1:
        torch.distributed.init_process_group("nccl", rank=args.local_rank)

    for n_train_examples in args.train_examples:
        train_ex_per_label, test_ex_per_label = None, None
        train_ex, test_ex = n_train_examples, args.test_examples
        if args.split_examples_evenly:
            train_ex_per_label = eq_div(n_train_examples, len(args.label_list)) if n_train_examples != -1 else -1
            test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
            train_ex, test_ex = None, None

        data_dir = os.path.join(args.data_dir, args.task_name)
        output_dir = args.output_dir.replace("[TASK_NAME]", args.task_name)

        train_data = load_examples(
            args.task_name, data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label
        )
        dev_data = load_examples(
            args.task_name, data_dir, DEV_SET, num_examples=test_ex, num_examples_per_label=test_ex_per_label
        )
        if args.do_test:
            try:
                test_data = load_examples(
                    args.task_name, data_dir, TEST_SET, num_examples=test_ex, num_examples_per_label=test_ex_per_label
                )
            except (FileNotFoundError, NotImplementedError):
                test_data = None
                warnings.warn("Test data not found.")
        else:
            test_data = None
        try:
            unlabeled_data = load_examples(
                args.task_name, data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples
            )
        except FileNotFoundError:
            warnings.warn("Unlabeled data not found.")
            unlabeled_data = None

        args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

        pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)
        sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)
        ipet_cfg = load_ipet_config(args)

        try:
            if args.method == "pet":
                final_results = pet.train_pet(
                    pet_model_cfg,
                    pet_train_cfg,
                    pet_eval_cfg,
                    sc_model_cfg,
                    sc_train_cfg,
                    sc_eval_cfg,
                    pattern_ids=args.pattern_ids,
                    output_dir=output_dir,
                    ensemble_repetitions=args.pet_repetitions,
                    final_repetitions=args.sc_repetitions,
                    reduction=args.reduction,
                    train_data=train_data,
                    unlabeled_data=unlabeled_data,
                    dev_data=dev_data,
                    test_data=test_data,
                    do_train=args.do_train,
                    do_eval=args.do_eval,
                    no_distillation=args.no_distillation,
                    seed=args.seed,
                    overwrite_dir=args.overwrite_output_dir,
                    save_model=args.save_model,
                    local_rank=args.local_rank,
                )

            elif args.method == "ipet":
                final_results = pet.train_ipet(
                    pet_model_cfg,
                    pet_train_cfg,
                    pet_eval_cfg,
                    ipet_cfg,
                    sc_model_cfg,
                    sc_train_cfg,
                    sc_eval_cfg,
                    pattern_ids=args.pattern_ids,
                    output_dir=output_dir,
                    ensemble_repetitions=args.pet_repetitions,
                    final_repetitions=args.sc_repetitions,
                    reduction=args.reduction,
                    train_data=train_data,
                    unlabeled_data=unlabeled_data,
                    dev_data=dev_data,
                    test_data=test_data,
                    do_train=args.do_train,
                    do_eval=args.do_eval,
                    seed=args.seed,
                    overwrite_dir=args.overwrite_output_dir,
                    save_model=args.save_model,
                    local_rank=args.local_rank,
                )

            elif args.method == "sequence_classifier":
                final_results = pet.train_classifier(
                    sc_model_cfg,
                    sc_train_cfg,
                    sc_eval_cfg,
                    output_dir=output_dir,
                    repetitions=args.sc_repetitions,
                    train_data=train_data,
                    unlabeled_data=unlabeled_data,
                    dev_data=dev_data,
                    test_data=test_data,
                    do_train=args.do_train,
                    do_eval=args.do_eval,
                    seed=args.seed,
                    overwrite_dir=args.overwrite_output_dir,
                    save_model=args.save_model,
                    local_rank=args.local_rank,
                )

            else:
                raise ValueError(f"Training method '{args.method}' not implemented")

        except json.decoder.JSONDecodeError:
            warnings.warn("JSONDecodeError in transformers")
            continue

        if final_results is not None and args.local_rank in [-1, 0]:
            if not wandb_initalized:
                wandb.init(project=f"pvp-vs-finetuning-{args.task_name}", name=naming_convention(args))
                wandb_initalized = True
            final_results["training_points"] = n_train_examples
            wandb.log(final_results)


def naming_convention(args):
    method = f"PVP {args.pattern_ids[0]}" if args.method == "pet" else "CLF"
    model = args.model_type
    if args.verbalizer_file is None or method == "CLF":
        verbalizer = None
    elif "neutral" in args.verbalizer_file:
        verbalizer = "neutral"
    elif "reverse" in args.verbalizer_file:
        verbalizer = "reverse"
    else:
        raise ValueError(f"unrecognized verbalizer file {args.verbalizer_file}")
    name = f"{method} {model}" + (f" {verbalizer} verbalizer" if verbalizer is not None else "") + f" seed {args.seed}"
    return name


if __name__ == "__main__":
    main()
