#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

cd /root/SLAM-LLM

# speech_encoder_path= TODO!


llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-debug

# -m debugpy --listen 5678 --wait-for-client
# if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python -m debugpy --listen 5679 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--model_name avsr \
--freeze_encoder \
--freeze_llm \
--llm_name vicuna-7b-v1.5 \
--llm_path $llm_path \
--llm_dim 4096 \
--encoder_name sota_avsr \
--encoder_dim 768 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 1 \
--batch_size_training 2 \
--val_batch_size 2 \
--num_workers_dataloader 2 \
--lr 1e-4 \
--output_dir $output_dir \
--metric acc \
--log_file "/root/SLAM-LLM/log/debug.log" \
--modality "audio" \
--model_config.modal "AO" \
--dataset_config.modal "AV" \
--pretrained_model_path "/nfs/yangguanrou.ygr/auto_avsr/asr_trlrwlrs2lrs3vox2avsp_base.pth" \


# --avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \


# --encoder_path $speech_encoder_path \   #TODO!
# --encoder_dim 1280 \   #TODO!