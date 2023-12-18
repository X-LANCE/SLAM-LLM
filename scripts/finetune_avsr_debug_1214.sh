#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM


llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
output_dir=/nfs/yangguanrou.ygr/llama-2-hf-finetune

# -m debugpy --listen 5678 --wait-for-client
python -m debugpy --listen 5679 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--model_name avsr \
--freeze_llm \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 1 \
--batch_size_training 4 \
--lr 1e-5 \
--output_dir $output_dir \
--metric acc \
--log_file "/root/SLAM-LLM/log/test.log" \


# --avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \