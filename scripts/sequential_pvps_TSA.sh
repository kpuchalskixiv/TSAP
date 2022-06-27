python3 cli.py \
  --method pet \
  --pattern_ids 0 \
  --data_dir split_data/ \
  --model_type gpt2 \
  --model_name_or_path gpt2 \
  --task_name tsa \
  --output_dir experiments/[TASK_NAME]/gpt2/supervised \
  --do_train \
  --do_test \
  --do_eval \
  --pet_per_gpu_eval_batch_size 8 \
  --pet_per_gpu_train_batch_size 8 \
  --pet_gradient_accumulation_steps 8 \
  --pet_num_train_epochs 10 \
  --pet_max_seq_length 256 \
  --pet_repetitions 1 \
  --train_examples 30000 80000 149000\
  --test_examples 1000 \
  --logging_steps 100 \
  --overwrite_output_dir \
  --no_distillation \
  --sc_per_gpu_train_batch_size 8 \
  --sc_per_gpu_unlabeled_batch_size 8 \
  --sc_gradient_accumulation_steps 8 \
  --sc_num_train_epochs 10 \
  --sc_max_seq_length 256 \
  --sc_repetitions 1 \
