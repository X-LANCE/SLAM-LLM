#!/bin/bash

#if running on multi-gpu machine
export CUDA_VISIBLE_DEVICES=2,3

cd /home/zhisheng/scratch/projects/llama-recipes

#/path/to/audio encoder model, download from https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view?usp=share_link
audio_encoder=/home/zhisheng/models/AudioMAE/finetuned.pth

model_name=/home/zhisheng/models/llama-2-hf                    #/path/to/llama-2-hf
output_dir=/home/zhisheng/models/llama-2-hf-finetune           #/path/to/output_dir

# -m debugpy --listen 55555 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python \
    -m llama_recipes.finetuning  \
    --use_peft --peft_method lora \
    --quantization \
    --audio_encoder $audio_encoder \
    --spatial_encoder $spatial_encoder \
    --dataset custom_dataset \
    --num_epochs 1 \
    --batch_size_training 2 \
    --model_name $model_name \
    --output_dir $output_dir
else
    torchrun \
        --nnodes 1 --nproc_per_node 2 \
        --master_port 31414 \
        -m llama_recipes.finetuning \
        --enable_fsdp --use_peft --peft_method lora \
        --audio_encoder $audio_encoder \
        --num_epochs 1 \
        --dataset custom_dataset \
        --batch_size_training 1 \
        --num_workers_dataloader 1 \
        --model_name $model_name \
        --output_dir $output_dir \
        --use_fast_kernels
fi