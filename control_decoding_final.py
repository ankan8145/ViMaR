from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, GenerationConfig
# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
# from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from PIL import Image
import requests
import json
from vlm_value_models import ValueModel
import math
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import os
from typing import List, Dict
import gc

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_data(data_path):
    datas = []
    with open(data_path, 'r') as files:
        for line in files:
            datas.append(json.loads(line))
    return datas

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])


# def get_dataset(data_pth, llava_image_pth):
#     with open(data_pth, "r", encoding="utf-8") as file:
#         llava_datas = json.load(file)
#     datas = []
#     for data in llava_datas:
#         prompt = data['prompt']
#         if prompt.startswith("<image>\n"):
#             prompt = prompt[len("<image>\n"):]
#         if prompt.endswith("\n<image>"):
#             prompt = prompt[:-len("\n<image>")]
#         datas.append({
#             'text': prompt,
#             'image': f'{llava_image_pth}//{data["image"]}',
#             'image_path': f'{llava_image_pth}//{data["image"]}'
#         })
#     return datas


def get_dataset(data_pth: str, llava_image_pth: str) -> List[Dict[str, str]]:
    dataset = []
    _, ext = os.path.splitext(data_pth)
    ext = ext.lower()

    if ext in {".parquet", ".pq"}:
        df = pd.read_parquet(data_pth)
        for idx, row in df.iterrows():
            question = row.get('question') or row.get('text') or ""
            image_field = row.get('image_path') or row.get('image') or f"{idx}.jpg"
            image_full_path = os.path.join(llava_image_pth, os.path.basename(image_field))
            dataset.append({
                'text': question,
                'image': image_full_path,
                'image_path': image_full_path
            })
    else:
        with open(data_pth, "r", encoding="utf-8") as file:
            records = json.load(file)
        for rec in records:
            prompt = rec.get('prompt') or rec.get('question') or ""
            if prompt.startswith("<image>\n"):
                prompt = prompt[len("<image>\n"):]
            if prompt.endswith("\n<image>"):
                prompt = prompt[:-len("\n<image>")]
            image_name = rec.get('image') or rec.get('image_path') or ""
            image_full_path = os.path.join(llava_image_pth, os.path.basename(image_name))
            dataset.append({
                'text': prompt,
                'image': image_full_path,
                'image_path': image_full_path
            })

    return dataset



def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ############# Load data ##############
    # datas = load_data(args.data_pth)
    # data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)

    dataset = get_dataset(data_pth=args.data_pth, llava_image_pth=args.image_folder)
    data_chunk = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    ############# Load VLM ##############
    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map=device if device == "cuda" else None)
    model = model.bfloat16()
    model.to(device)

    print("Model Loading....")
    ############# Load Value net ##############
    value_net = ValueModel(args.model_id)
    value_net.from_pretrained(args.value_net_pth)
    torch.cuda.empty_cache()
    value_net_dtype = torch.float16 if device == "cuda" else torch.float32
    value_net.to(device)
    value_net = value_net.bfloat16()
    print("value_net Model Loaded....")
    tokenizer_max_len = getattr(value_net.processor.tokenizer, "model_max_length", args.value_net_max_length)
    value_net_max_length = min(args.value_net_max_length, tokenizer_max_len)
    if value_net_max_length <= 0:
        raise ValueError("value_net_max_length must be positive.")
     
    import time
    decoding_results = []
    for data in tqdm(data_chunk, desc="Decoding Progress"):
        start_time = time.perf_counter()
        try:
            images = [Image.open(data['image_path'])]
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": data['text']},
                    {"type": "image"},
                ],
            }]

            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=images, text=[prompt], return_tensors="pt").to(device)
            question_input_length = inputs['input_ids'].shape[1]
            temp_generation_config_list = [
                GenerationConfig(
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.5,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.9,
                ),
                GenerationConfig(
                    do_sample=False,
                )
            ]

            candidate_replies = []
            candidate_states = []

            for temp_generation_config in temp_generation_config_list:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=temp_generation_config,
                        max_length=4096,
                        tokenizer=processor.tokenizer,
                        # stop_strings=['.'],
                        num_beams=args.num_beams,
                        num_return_sequences=args.num_return_sequences
                    )

                for output in outputs:
                    new_generated_reply = processor.decode(output[question_input_length:], skip_special_tokens=True)
                    candidate_replies.append(new_generated_reply)
                    
                    state = value_net.base_model.processor.apply_chat_template([
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": new_generated_reply},
                                {"type": "image"}
                            ]
                        }
                    ], tokenize=False)
                    candidate_states.append(state)

            batch = value_net.base_model.processor(
                text=candidate_states,
                images=len(candidate_states)*images,
                padding='max_length',
                max_length=value_net_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            current_inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'pixel_values': batch['pixel_values'],
                'image_sizes': batch['image_sizes'],
            }

            with torch.no_grad():
                candidate_values = value_net.base_model(current_inputs)

            max_index = torch.argmax(candidate_values).item()
            chosen_response = candidate_replies[max_index]

            ## Find the low-scoring segments
            sentences = [s.strip() for s in chosen_response.split('.') if s.strip()]
            if not sentences:                           # nothing to filter
                final_caption = chosen_response
            else:
                sentence_states = [
                    value_net.base_model.processor.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": s},
                                    {"type": "image"}          
                                ]
                            }
                        ],
                        tokenize=False
                    )
                    for s in sentences
                ]
                batch = value_net.base_model.processor(
                    text=sentence_states,
                    images=len(sentence_states) * images,   # replicate image per sentence
                    padding='max_length',
                    max_length=value_net_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                current_inputs = {
                    "input_ids":     batch["input_ids"],
                    "attention_mask":batch["attention_mask"],
                    "pixel_values":  batch["pixel_values"],
                    "image_sizes":   batch["image_sizes"],
                }

                with torch.no_grad():
                    values = value_net.base_model(current_inputs)     
                values = values.squeeze(-1) 
                # ── 5. filter sentences whose score ≥ 2.0 ──────────────────────────────────
                keep_mask         = values >= 2.14                     # Bool tensor
                filtered_sentences = [s for s, keep in zip(sentences, keep_mask) if keep]

                final_caption = '. '.join(filtered_sentences)
                if final_caption:                                      # add trailing period
                    final_caption += '.'

            new_generated_reply = final_caption


            del inputs
            assistant_reply = None
            # max_iterations = 5  # Prevent infinite loops  and iteration < max_iterations
            iteration = 0
            
            while assistant_reply != new_generated_reply:
                assistant_reply = new_generated_reply

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": data['text']},
                            {"type": "image"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": '{TEXT}'}, ],
                    }]
                conversation[-1]['content'][0]['text'] = chosen_response
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                prompt = prompt[:-5]  # remove trailing token (e.g., '</s>')
                inputs = processor(images=images, text=[prompt], return_tensors="pt").to(device)

                reply_input_length = inputs['input_ids'].shape[1]

                candidate_replies = []
                candidate_new_replies = []
                candidate_states = []

                for temp_generation_config in temp_generation_config_list:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            generation_config=temp_generation_config,
                            max_length=4096,
                            # max_length=512,
                            tokenizer=processor.tokenizer,
                            stop_strings=['.'],
                            num_beams=args.num_beams,
                            num_return_sequences=args.num_return_sequences
                        )
                        
                    
                    for output in outputs:
                        reply_candidate = processor.decode(output[question_input_length:], skip_special_tokens=True)
                        new_generated_candidate = processor.decode(output[reply_input_length:], skip_special_tokens=True)
                        candidate_replies.append(reply_candidate)
                        candidate_new_replies.append(new_generated_candidate)

                        state = value_net.base_model.processor.apply_chat_template([
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": new_generated_candidate},
                                    {"type": "image"}
                                ]
                            }
                        ], tokenize=False)
                        candidate_states.append(state)

                batch = value_net.base_model.processor(
                    text=candidate_states,
                    images=len(candidate_states)*images,
                    padding='max_length',
                    max_length=value_net_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)
                
                current_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'pixel_values': batch['pixel_values'],
                    'image_sizes': batch['image_sizes']
                }
                with torch.no_grad():
                    candidate_values = value_net.base_model(current_inputs)

                best_index = torch.argmax(candidate_values).item()
                chosen_response = candidate_replies[best_index]
                new_generated_reply = candidate_new_replies[best_index]

                del inputs, batch  # Free memory for next iteration
                iteration += 1 
                torch.cuda.empty_cache()
                gc.collect()
            print(chosen_response)
            decoding_results.append({
                'text': data['text'],
                'image': data['image'],
                'image_path': data['image_path'],
                'decoding_result': chosen_response,
            })

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        elapsed = time.perf_counter() - start_time
        print(f"\n[control_decoding_final.py] elapsed time: {elapsed:.2f} s")
    dump_to_jsonl(decoding_results, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--data_pth", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--value_net_pth", type=str, default=None)
    parser.add_argument("--value_net_max_length", type=int, default=2048)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--per_gpu_batch_size", type=int, default=2)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    main(args)
