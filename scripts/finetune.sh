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
python src/llama_recipes/pipeline/finetune.py \
--model_name echat \
--use_peft --peft_method lora \
--quantization \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--dataset custom_dataset \
--custom_dataset.file src/llama_recipes/datasets/echat_dataset.py:get_audio_dataset \
--custom_dataset.data_path /nfs/zhifu.gzf/data/IEMOCAP_full_release/datalist.jsonl \
--batching_strategy custom \
--custom_dataset.max_words 1024 \
--num_epochs 1 \
--batch_size_training 2 \
--output_dir $output_dir 