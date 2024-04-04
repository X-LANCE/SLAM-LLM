#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/slides-finetune-20230125-debug

# -m debugpy --listen 5678 --wait-for-client

python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--model_name asr \
--freeze_encoder \
--freeze_llm \
--llm_name vicuna-7b-v1.5 \
--llm_path $llm_path \
--llm_dim 4096 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset slides_dataset \
--batching_strategy custom \
--num_epochs 40 \
--batch_size_training 2 \
--val_batch_size 2 \
--num_workers_dataloader 0 \
--lr 1e-3 \
--output_dir $output_dir \
--metric acc \

