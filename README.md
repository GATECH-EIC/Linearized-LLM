## When Linear Attention Meets Autoregressive Decoding: Towards More Effective and Efficient Linearized Large Language Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

Haoran You, Yichao Fu, Zheng Wang, Amir Yazdanbakhsh, Yingyan (Celine) Lin

Accepted by [**ICML 2024**](https://icml.cc/Conferences/2024). More Info:
\[ [**Paper**](https://arxiv.org/abs/2406.07368) | [**Github**](https://github.com/GATECH-EIC/Linearized-LLM) \]
<!-- | **Slide** | [**Youtube**](TBD) -->

---

## News 🔥🔥 !
- [ ✅ New ] Jun. 11, 2024. 💥 Release our trained LLaMA-2-7B model checkpoints on [Huggingface](https://huggingface.co/LinearizedLLM)!
- [ ✅ New ] Jun. 11, 2024. 💥 [Linearized-LLM](https://github.com/GATECH-EIC/Linearized-LLM)'s PyTorch implementation codes are released!

## Table of Content

[Brief Introduction](#brief-introduction)

[Basic Usage](#basic-usage)
* [Set up Environment](#set-up-environment)
* [Download Trained Models ](#download-trained-models)
* [Reproduce Results](#reproduce-results)

[Train Your Own Linerized-LLM](#train-your-own-linerized-llm)
* [FLASH Training](#flash-training-from-scratch)
* [T5 Fine-tuning](#t5-fine-tuning)
* [GPT-2 Fine-tuning](#gpt-2-fine-tuning)
* [LLaMA-2 Fine-tuning](#llama-2-fine-tuning)

[Citation & Acknowledgement]()

## Basic Usage

The main implementation can be found in the `autoregressive_wrapper.py` and `flash_pytorch.py` files. The code is adapted from [FLASH](https://github.com/lucidrains/FLASH-pytorch).

### Set up Environment

Please set up the environment using the following commands and ensure that CUDA is included in your PATH:

```bash
export PATH=/PATH-TO-CUDA/:$PATH
conda create -n LinearLLM python==3.10
conda activate LinearLLM
pip install -r requirements.txt
pip install flash-attn
```

### Download Trained Models 

We provide our trained model checkpoints at this [HuggingFace repository](https://huggingface.co/LinearizedLLM). Follow the bash script below to download the model:

```bash
# Linearized LLaMA-2 weights
huggingface-cli download LinearizedLLM/llama-2-7b-aug-linear --local-dir llama-2-7b-aug-linear

# Medusa Head for Linearized LLaMA-2 weights
huggingface-cli download LinearizedLLM/llama-2-7b-medusa-head-aug-linear --local-dir llama-2-7b-medusa-head-aug-linear
```

### Reproduce Results

To reproduce Table 8 from the paper, which demonstrates the speedup of augmented linearized LLaMA-2 with speculative decoding, use the following bash script. The code is adapted from the [Medusa repository](https://github.com/FasterDecoding/Medusa).

```bash
cd experiments
bash run_medusa.sh
```

To reproduce Table 4, which shows latency and memory improvements with our augmented linear attention, use the following bash script. Note that we use `transformers==4.37.0`.

```bash
pip install transformers==4.37.0
cd experiments
bash run_benchmark.sh
```

## Train Your Own Linerized-LLM


### FLASH Training from Scratch

Use the bash script below to train a 24-layer FLASH Model from scratch:

```bash
bash runall-125k.sh
```

### T5 Fine-tuning

Use the bash script below to finetune T5 with augmented linear attention. The code is adapted from the [transformers repository](https://github.com/huggingface/transformers).

```bash
cd experiments
bash tasks_run-t5.sh
```

### GPT-2 Fine-tuning

Use the bash script below to finetune GPT-2 with augmented linear attention. The code is adapted from the [transformers repository](https://github.com/huggingface/transformers).

```bash
cd experiments
bash tasks_run-gpt2.sh
```

### LLaMA-2 Fine-tuning

Use the bash script below to finetune LLaMA-2 with augmented Linear Attention. The code is adapted from the [LongLoRA repository](https://github.com/dvlab-research/LongLoRA).

```bash
cd experiments
bash tasks_run-llama2.sh
```

## Citation & Acknowledgement

````bibtex
@inproceedings{you2024linear,
  title={When Linear Attention Meets Autoregressive Decoding: Towards More Effective and Efficient Linearized Large Language Models},
  author={You, Haoran and Fu, Yichao and Wang, Zheng and Yazdanbakhsh, Amir and Lin, Yingyan (Celine)},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML 2024)},
  year={2024},
}
````
Thanks to the developers of [FLASH](https://github.com/lucidrains/FLASH-pytorch), [transformers](https://github.com/huggingface/transformers), [LongLoRA](https://github.com/dvlab-research/LongLoRA), and [Medusa](https://github.com/FasterDecoding/Medusa) for providing their codebases!