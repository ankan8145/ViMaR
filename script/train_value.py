import subprocess
import os

# Set your dataset and splits here:
dataset_name = "LLAVA_Data"
dataset_train_split = "train"
dataset_test_split  = "test"

# Define the training command
cmd = [
    "accelerate", "launch",
    "--mixed_precision", "fp16",
    # "python",
    "value_training.py",
    # TRL required args:
    "--dataset_name", dataset_name,
    "--dataset_train_split", dataset_train_split,
    "--dataset_test_split", dataset_test_split,
    "--output_dir", "./vimar-ckpt-full-finetune",

    # your original args:
    "--per_device_train_batch_size", "8",
    # "--gradient_accumulation_steps", "4",
    # "--num_items_in_batch", "16",
    "--bf16",
    
    # "--save_strategy",      "epoch",
    #  "--evaluation_strategy","epoch",
     
    "--save_strategy",      "steps",
    "--save_steps", "500",
    # "--eval_strategy","epoch",
    # "--eval_steps",         "1",
    
    #  "--metric_for_best_model", "eval_loss",
    #  "--load_best_model_at_end",
     "--save_total_limit",   "2",

    "--torch_dtype", "bfloat16",
    "--report_to", "wandb",
    "--log_level", "info",
    "--logging_steps", "50",
    "--logging_strategy", "steps",
    "--gradient_checkpointing",
   
   "--use_peft", "False",
#    "--lora_target_modules",      "q_proj", "k_proj", "v_proj", "o_proj",
    # "--lora_task_type",           "SEQ_2_SEQ_LM",
]



# Print the full command for verification
print("Running single-GPU training command:")
print(" ".join(cmd))

try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training failed with exit code {e.returncode}")

