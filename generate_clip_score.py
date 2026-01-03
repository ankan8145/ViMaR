import argparse
import json
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def main(args):
    # device = "cuda:{}".format(args.gpu_id)
    device = "cuda"
    model = CLIPModel.from_pretrained(args.clip_id, device_map=device)
    processor = CLIPProcessor.from_pretrained(args.clip_id)

    datas = []
    with open(args.data_pth, 'r') as files:
        for line in files:
            datas.append(json.loads(line))


    # def compute_similarity(inputs, model):
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #         image_embeds = outputs.image_embeds
    #         text_embeds = outputs.text_embeds
    #         image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    #         text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    #         similarity = torch.matmul(text_embeds, image_embeds.T)
    #         return similarity.item()


    def compute_similarity(inputs, model, tau=0.16, alpha=1.0):
        """
        If similarity ≥ tau, returns similarity as reward (no penalty).
        Otherwise applies margin-based penalty:
            pos = max(sim - tau, 0)
            neg = max(tau - sim, 0)
            reward = pos - alpha * neg
        """
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds  = outputs.text_embeds
    
            # normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)
    
            # cosine similarity for a single text/image pair
            sim = torch.matmul(text_embeds, image_embeds.T).squeeze()  # scalar tensor
    
            # conditional reward
            if sim.item() >= tau:
                reward = sim
            else:
                pos = F.relu(sim - tau)     # will be 0 since sim < tau
                neg = F.relu(tau - sim)     # = tau - sim
                reward = pos - alpha * neg  # negative penalty
    
            return reward.item()


    data_with_sim = []
    for i in tqdm(range(len(datas))):
    # for data in datas:
        tmp_dict = {
            'text': datas[i]['text'],
            'image': datas[i]['image'],
            'image_path': datas[i]['image_path'],
            'temp_02': datas[i]['temp_02'],
            'temp_04': datas[i]['temp_04'],
            'temp_06': datas[i]['temp_06'],
            'temp_08': datas[i]['temp_08'],
            'greedy': datas[i]['greedy'],
            'original_answer': datas[i]['original_answer'],
            'rewards': {
                'temp_02': {},
                'temp_04': {},
                'temp_06': {},
                'temp_08': {},
                'greedy': {},
                'original_answer': {},
            }
        }
        image = Image.open(datas[i]['image_path'])

        try:
            for decode_type in ['temp_02', 'temp_04', 'temp_06', 'temp_08', 'greedy', 'original_answer']:
                for sentence in datas[i][decode_type].split('[/INST] ')[-1].split('.'):
                    if sentence != ' ':
                        if decode_type == 'original_answer':
                            tmp_dict['rewards'][decode_type][sentence] = 0.27
                        else:
                            inputs = processor(text=[sentence], images=image, return_tensors="pt", padding=True).to(device)
                            tmp_dict['rewards'][decode_type][sentence] = compute_similarity(inputs, model)
            data_with_sim.append(tmp_dict)
        except:
            print(datas[i])

    dump_to_jsonl(data_with_sim, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_id", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--data_pth", type=str, default="./output_files/pretrain_value_batch_generate_llava1_6_mistral_7b_res_1.jsonl")
    parser.add_argument("--output_file", type=str, default="./output_files/pretrain_clip_score_1.jsonl")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    main(args)