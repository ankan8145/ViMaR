#!/bin/bash
#SBATCH --job-name=finetune_llava
#SBATCH --output=logs/finetune_llava_%j.out
#SBATCH --error=logs/finetune_llava_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3                # Request 3 GPUs
#SBATCH --cpus-per-task=12          # Adjust based on num_workers
#SBATCH --mem=48G                   # Adjust based on data/model size
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive                 # Optional: reserve full node

# Activate conda environment
source ~/.bashrc
conda activate DeepSeek-VL2_fine_tune

# Make logs directory if it doesn't exist
mkdir -p logs

# Run your training Python script
python value_training.py
