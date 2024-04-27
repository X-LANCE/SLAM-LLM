#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export HF_ENDPOINT=https://hf-mirror.com

python /root/SLAM-LLM/src/llama_recipes/utils/fence.py