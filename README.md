# ViMaR

Official codebase for the paper **“Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension.”**

This repository provides tools to generate responses, prepare training data, train a Value-guided Inference with Margin-based Reward
(ViMaR), and perform supervised fine-tuning (SFT) of a Vision-Language Model (VLM).

---

## 🖼️ Conference Presentation Poster

The following poster was presented at the conference and provides a concise overview of our motivation, methodology, and key results for **ViMaR**.

<p align="center">
  <img src="Poster.png" alt="ViMaR Conference Poster" width="900"/>
</p>

📄 **Paper:** [Dual-Stage Value-Guided Inference with Margin-Based Reward Adjustment for Fast and Faithful VLM Captioning](https://arxiv.org/pdf/2506.15649)

---

## 📦 Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate vimar
```

### Patch Required Libraries

After activating the environment, copy the modified utility files into the appropriate installed packages:

```bash
cp ./utils/modeling_llava_next.py \
~/.conda/envs/vimar/lib/python3.12/site-packages/transformers/models/llava_next/

cp ./utils/trainer/td_trainer.py \
~/.conda/envs/vimar/lib/python3.12/site-packages/trl/trainer/

cp ./utils/__init__.py \
~/.conda/envs/vimar/lib/python3.12/site-packages/trl/

cp ./utils/trainer/__init__.py \
~/.conda/envs/vimar/lib/python3.12/site-packages/trl/trainer/
```

⚠️ **Note:** These steps overwrite files in your local Python environment. It is strongly recommended to use a dedicated conda environment.

---

## 🧠 Generate Model Responses

Run batch inference to generate responses:

```bash
bash ./script/batch_generate.sh or python ./script/batch_generate_command.py
```

---

## 🧪 Prepare TD Training Data

Compute CLIP-based scores to prepare temporal-difference (TD) training data:

```bash
bash ./script/clip_score.sh or python ./script/clip_score.py
```

---

## 📈 Train Vision Value Model (ViMaR)

Train the value model using the prepared TD data:

```bash
bash ./script/train_value.sh or python ./script/train_value.py
```

---

## 🎯 Supervised Fine-Tuning (SFT) of VLM

Perform supervised fine-tuning of the vision-language model:

```bash
bash ./script/train_sft.sh
```

---

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{deria2025dual,
  title={Dual-Stage Value-Guided Inference with Margin-Based Reward Adjustment for Fast and Faithful VLM Captioning},
  author={Deria, Ankan and Dukre, Adinath Madhavrao and Tang, Feilong and Atito, Sara and Roy, Sudipta and Awais, Muhammad and Khan, Muhammad Haris and Razzak, Imran},
  journal={arXiv preprint arXiv:2506.15649},
  year={2025}
}
```


## 🙏 Acknowledgements

We thank the authors of the original VisVM repository for releasing their code:
https://github.com/si0wang/VisVM
