#!/usr/bin/env python
import subprocess
import os
import time 

# Define the parameters
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
value_net_pth = "VIMAR/VIMAR-LLaVA-Next-Mistral-7B"

output_prefix = "./delete/value_net_decoding_results_finetune_mmhal_VIMAR1_"
final_file_prefix = "./value_net_decoding_results_finetune_mmhal_VIMAR1_"
# output_prefix = "./vimar/value_net_decoding_results_mmvet_" 
# final_file_prefix = "./value_net_decoding_results_mmvet_"

num_chunks = 1

data_pth = "VIMAR/MMHal_data/response_template_clean.json"
image_folder = "VIMAR/MMHal_data/resized_images"

# data_pth = "VIMAR/MMVet/test-00000-of-00001.parquet"
# image_folder = "VIMAR/MMVet/MMVet"


# data_pth = "/storagepool/ankan/VIMAR_mod/VIMAR/output_files_1/train_data_with_clip_score_train_1_sample100.jsonl"
# image_folder = "/storagepool/ankan/Data/coco2017/train2017"

# Start all subprocesses in parallel
processes = []
start_time = time.perf_counter()
for i in range(num_chunks):
    command = [
        "python", "control_decoding.py",
        # "python", "control_decoding.py",
        "--model_id", model_id,
        "--value_net_pth", value_net_pth,
        "--data_pth", data_pth,
        "--image_folder", image_folder,
        "--output_file", f"{output_prefix}{i+1}.jsonl",
        "--step_size", "5",
        "--num-chunks", str(num_chunks),
        "--chunk-idx", str(i),
        "--gpu-id", "0"
    ]
    
    processes.append(subprocess.Popen(command))

# Wait for all subprocesses to complete
for process in processes:
    process.wait()
    
# elapsed = time.perf_counter() - start_time
# print(f"\n[control_decoding_ad.py] elapsed time: {elapsed:.2f} s")

# Concatenate all output files into one final file
with open(f"{final_file_prefix}.jsonl", "wb") as final_file:
    for i in range(num_chunks):
        chunk_file = f"{output_prefix}{i+1}.jsonl"
        if os.path.exists(chunk_file):
            with open(chunk_file, "rb") as infile:
                final_file.write(infile.read())
