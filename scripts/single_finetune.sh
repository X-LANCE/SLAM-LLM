#!/bin/bash

#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=5

model_name=/data1/scratch/zhisheng/models/llama-2-hf
output_dir=/data1/scratch/zhisheng/models/llama-2-hf-finetune


python -m debugpy --listen 55555 --wait-for-client \
-m llama_recipes.finetuning  \
--use_peft --peft_method lora --quantization \
--dataset custom_dataset \
--batch_size_training 64 \
--model_name $model_name \
--output_dir $output_dir


# samsum_dataset, custom_dataset