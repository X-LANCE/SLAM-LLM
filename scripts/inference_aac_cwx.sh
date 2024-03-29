#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/tiny.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/base.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/small.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/medium.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2.pt
audio_encoder_path=/root/models/BEATs_iter3_plus_AS2M.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

# llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-intermediate-step-1431k-3T
# llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-Chat-v0.4
# llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
# llm_path=/nfs/maziyang.mzy/models/Llama-2-7b-chat-hf
# llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
llm_path=/root/models/vicuna-7b-v1.5

output_dir=/root/exps/test
ckpt_path=$output_dir/aac/1
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102-renew5/asr/1

# -m debugpy --listen 5678 --wait-for-client
python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/inference.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    model_config.llm_name="vicuna-7b-v1.5" \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name=beats \
    model_config.encoder_ds_rate=5 \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector=linear \
    dataset_config.fix_length_audio=64 \
    +ckpt_path=$ckpt_path/model.pt \
    +wav_path="/root/data/AudioCaps/waveforms/test/YnLZeG9LaLgw.wav" \
    +prompt="Describe the audio you hear. Output the audio caption directly without redundant content. Ensure that the output is not duplicated." \
    train_config.model_name=aac \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \
# ++model_config.encoder_projector=linear \
# ++model_config.encoder_projector_ds_rate=5 \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \