#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

audio_encoder_path=/home/oss/maziyang.mzy/models/AudioMAE/finetuned.pth
speech_encoder_path=/home/oss/maziyang.mzy/models/Whisper/base.pt
llm_path=/home/oss/zhifu.gzf/ckpt/Llama-2-7b-hf
output_dir=/nfs/maziyang.mzy/models/llama-2-hf-finetune

# -m debugpy --listen 5678 --wait-for-client
python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--model_name avsr \
--use_peft --peft_method lora \
--quantization \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 1 \
--batch_size_training 2 \
--output_dir $output_dir 