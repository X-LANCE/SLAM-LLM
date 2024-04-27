#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false

cd /root/SLAM-LLM


audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M.pt

llm_path=/root/models/vicuna-7b-v1.5

output_dir=/root/exps/clotho/wavcaps_btz16_pt4_clotho_ft_seed42_btz4_lr1e-5
ckpt_path=$output_dir/aac/1

# -m debugpy --listen 6666 --wait-for-client
python src/llama_recipes/pipeline/inference.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    model_config.llm_name="vicuna-7b-v1.5" \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name=eat \
    model_config.encoder_ds_rate=5 \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector=linear \
    model_config.encoder_projector_ds_rate=5 \
    dataset_config.dataset=audio_dataset \
    dataset_config.fbank_mean=-4.268 \
    dataset_config.fbank_std=4.569 \
    dataset_config.model_name=eat \
    dataset_config.fixed_length=true \
    dataset_config.target_length=1024 \
    +ckpt_path=$ckpt_path/model.pt \
    +wav_path="/root/data/Clotho/clotho_audio_files/evaluation/In the City.wav" \
    +prompt="Describe the audio you hear." \
    train_config.model_name=aac \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \
    +peft_ckpt=$ckpt_path \
    train_config.use_peft=true \
# --use_peft --peft_method lora \