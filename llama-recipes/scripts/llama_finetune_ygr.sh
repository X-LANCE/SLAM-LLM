#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM/llama-recipes

#/path/to/audio encoder model, download from https://drive.google.com/file/d/18EsFOyZYvBYHkJ7_n7JFFWbj6crz01gq/view?usp=share_link
#audio_encoder=/home/oss/maziyang.mzy/models/AudioMAE/finetuned.pth
audio_encoder=/nfs/zhifu.gzf/init_model/whisper/large-v2.pt

model_name=/home/oss/zhifu.gzf/ckpt/Llama-2-7b-hf                    #/path/to/llama-2-hf
output_dir=/nfs/maziyang.mzy/models/llama-2-hf-finetune           #/path/to/output_dir

#     -m debugpy --listen 5678 --wait-for-client \
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python \
    -m llama_recipes.finetuning  \
    --use_peft --peft_method lora \
    --quantization \
    --audio_encoder $audio_encoder \
    --dataset custom_dataset \
    --num_epochs 1 \
    --batch_size_training 1 \
    --model_name $model_name \
    --output_dir $output_dir
else
    torchrun \
        --nnodes 1 --nproc_per_node 2 \
        --master_port 31414 \
        -m llama_recipes.finetuning \
        --enable_fsdp --use_peft --peft_method lora \
        --quantization \
        --audio_encoder $audio_encoder \
        --num_epochs 1 \
        --dataset custom_dataset \
        --batch_size_training 1 \
        --num_workers_dataloader 1 \
        --model_name $model_name \
        --output_dir $output_dir \
        --use_fast_kernels
fi