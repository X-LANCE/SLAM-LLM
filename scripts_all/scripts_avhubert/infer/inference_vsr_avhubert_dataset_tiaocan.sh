#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1


cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/av_hubert/large_vox_433h.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-large_vox_433h-tri-dataset-tiaocan_again
ckpt_path=$output_dir/asr/850

decode_log=$ckpt_path/decode_log_test_clean_beam4_repetition_penalty1

# 
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
model_config.encoder_projector=cov1d-linear \
model_config.encoder_projector_ds_rate=5 \
dataset_config.dataset=avhubert_dataset \
dataset_config.file="src/llama_recipes/datasets/avhubert_dataset.py:get_audio_dataset" \
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