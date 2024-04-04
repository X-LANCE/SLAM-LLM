#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM


speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5


output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-ao-whisper-1e-4-0123

ckpt_path=$output_dir/avsr/60
peft_ckpt=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-ao-whisper-1e-4-0123/avsr/60

decode_log=$ckpt_path/decode_LRS3_test_peft_beam4_repetition_penalty1

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
--model_name avsr \
--freeze_encoder \
--llm_name vicuna-7b-v1.5 \
--llm_path $llm_path \
--llm_dim 4096 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset_inference.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 1 \
--val_batch_size 1 \
--num_workers_dataloader 0 \
--output_dir $output_dir \
--ckpt_path $ckpt_path/model.pt \
--decode_log $decode_log \
--freeze_llm \
--test_split test \
--avsr_dataset.modal AO \
--model_config.modal AO \
--modality "audio" \
--peft_ckpt $peft_ckpt \
--use_peft --peft_method lora \
# --speech_dataset.val_data_path $val_data_path \