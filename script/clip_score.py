import subprocess
import glob
import os

# Parameters (adjust as needed)
clip_id = "openai/clip-vit-large-patch14-336"
input_prefix = "./output_files_1/train_data_with_score_3_"
output_prefix = "./output_files_1/train_data_with_clip_score_3_"
final_file_prefix = "./train_data_with_clip_score_3_"
num_chunks = 1

# Ensure the output directory exists
os.makedirs("./output_files_1", exist_ok=True)

# Run each chunk sequentially on a single GPU
for i in range(num_chunks):
    input_file = f"{input_prefix}{i+1}.jsonl"
    output_file = f"{output_prefix}{i+1}.jsonl"
    cmd = [
        "python", "generate_clip_score.py",
        "--clip_id", clip_id,
        "--data_pth", input_file,
        "--output_file", output_file,
        "--gpu-id", "1"
    ]
    print("Running command:", " ".join(cmd))
    # Set CUDA_VISIBLE_DEVICES to the GPU you want to use (0 for single-GPU)
    # env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.check_call(cmd)

# Concatenate all JSONL files into a single final file.
final_output_file = f"{final_file_prefix}.jsonl"
with open(final_output_file, "wb") as outfile:
    for filename in sorted(glob.glob(f"{output_prefix}*.jsonl")):
        with open(filename, "rb") as readfile:
            outfile.write(readfile.read())

print("All tasks completed and output files concatenated into:", final_output_file)
