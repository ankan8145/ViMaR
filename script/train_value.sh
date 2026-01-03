#!/usr/bin/env bash
# set -e
# expose only GPUs 1,2,3
export CUDA_VISIBLE_DEVICES=1,2,3
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ./script/deepspeed_zero3.yaml --num_processes=3 ./value_training.py --per_device_train_batch_size 16 --gradient_accumulation_steps 8 --output_dir ./vimar-ckpt --bf16 --save_steps 2 --torch_dtype bfloat16 --report_to wandb --log_level info --logging_steps 1 --logging_strategy steps --gradient_checkpointing
