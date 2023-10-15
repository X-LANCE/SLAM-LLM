#!/bin/bash

#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=5

#/path/to/audio encoder model, download from https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view?usp=share_link
audio_encoder_path=/home/zhisheng/models/AudioMAE/finetuned.pth

model_name=/home/zhisheng/models/llama-2-hf                    #/path/to/llama-2-hf
output_dir=/home/zhisheng/models/llama-2-hf-finetune           #/path/to/output_dir

# -m debugpy --listen 55555 --wait-for-client
python \
    -m llama_recipes.finetuning  \
    --use_peft --peft_method lora \
    --quantization \
    --dataset custom_dataset \
    --audio_encoder_path $audio_encoder_path \
    --batch_size_training 2 \
    --model_name $model_name \
    --output_dir $output_dir


# samsum_dataset, custom_dataset