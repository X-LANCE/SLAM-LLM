#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1


cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/hubert_ckpt/hubert_large_ll60k_finetune_ls960.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_hubert/vicuna-7b-v1.5-hubert_large_ll60k_finetune_ls960-0129
ckpt_path=$output_dir/asr/8594
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4
val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_test_other_filtered.jsonl
decode_log=$ckpt_path/decode_log_test_other_beam4_repetition_penalty1_debug_bs1

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
--config-path "/root/SLAM-LLM/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$ckpt_path \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=hubert \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_type="finetune" \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=hubert_dataset \
++dataset_config.file="src/llama_recipes/datasets/hubert_dataset.py:get_speech_dataset" \
++dataset_config.prompt="Transcribe speech to text. " \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.inference_mode=true \
++dataset_config.normalize=true \
++train_config.model_name=asr \
++train_config.batching_strategy=custom \
++train_config.num_epochs=1 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=1 \
++train_config.output_dir=$output_dir \
++ckpt_path=$ckpt_path/model.pt \
++decode_log=$decode_log \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \