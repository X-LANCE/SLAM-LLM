#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1


cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/av_hubert/large_vox_433h.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-large_vox_433h-0131-prompt
ckpt_path=$output_dir/asr/448
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4

decode_log=$ckpt_path/decode_log_test_clean_beam4_repetition_penalty1

# -m debugpy --listen 5678 --wait-for-client -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
--config-path "/root/SLAM-LLM/scripts/conf_avsr" \
--config-name "avsr.yaml" \
hydra.run.dir=$output_dir \
model_config.llm_name="vicuna-7b-v1.5" \
model_config.llm_path=$llm_path \
model_config.llm_dim=4096 \
model_config.encoder_name=av_hubert \
model_config.encoder_path=$speech_encoder_path \
model_config.encoder_dim=1024 \
model_config.encoder_projector=linear \
model_config.encoder_projector_ds_rate=5 \
dataset_config.dataset=avsr_dataset \
dataset_config.modal=VO \
dataset_config.test_split=test \
model_config.modal=VO \
dataset_config.inference_mode=true \
train_config.model_name=asr \
train_config.batching_strategy=custom \
train_config.num_epochs=1 \
train_config.val_batch_size=1 \
train_config.num_workers_dataloader=0 \
train_config.output_dir=$output_dir \
+ckpt_path=$ckpt_path/model.pt \
+decode_log=$decode_log \
train_config.freeze_encoder=true \
train_config.freeze_llm=true \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \