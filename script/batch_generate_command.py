import subprocess
import glob
import os

# Parameters (adjust as needed)
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
output_prefix = "./final_result/pretrain_value_batch_generate_llava1_6_mistral_7b_res_split3_"
final_file_prefix = "./pretrain_value_batch_generate_llava1_6_mistral_7b_res_split3"
batch_size = 8
num_chunks = 1

# Ensure the output directory exists
os.makedirs("./final_result", exist_ok=True)

llava_data_pth = "/output_files_1/train_data_with_clip_score_train_1_sample100.jsonl"
image_folder = "/storagepool/ankan/Data/coco2017/train2017"
# Run each chunk sequentially.
# Note: For a single GPU system, we pass '--gpu-id' as 0 for all runs.
for i in range(num_chunks): 
    output_file = f"{output_prefix}{i+1}.jsonl"
    cmd = [
        "python", "batch_generate.py",
        "--model_id", model_id,
        "--llava_data_pth", llava_data_pth,
        "--image_folder", image_folder,
        "--output_file", output_file,
        "--per_gpu_batch_size", str(batch_size),
        "--num-chunks", str(num_chunks),
        "--chunk-idx", str(i),
        "--gpu-id", "2"
    ]
    print("Running command:", " ".join(cmd))
    subprocess.check_call(cmd)

# Concatenate all JSONL files into a single final file.
final_output_file = f"{final_file_prefix}.jsonl"
with open(final_output_file, "wb") as outfile:
    # Use sorted() to ensure files are concatenated in order.
    for filename in sorted(glob.glob(f"{output_prefix}*.jsonl")):
        with open(filename, "rb") as readfile:
            outfile.write(readfile.read())

print("All tasks completed and output files concatenated into:", final_output_file)
