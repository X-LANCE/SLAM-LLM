#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/tiny.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/base.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/small.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/medium.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Base.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Large.pt
audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M.pt

# llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-intermediate-step-1431k-3T
# llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-Chat-v0.4
# llm_path=/nfs/maziyang.mzy/models/phi-2
# llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
# llm_path=/nfs/maziyang.mzy/models/Llama-2-7b-chat-hf
llm_path=/root/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/root/exps/eat_finetune_linear
ckpt_path=$output_dir/aac/4
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4
val_data_path=/root/data/AudioCaps/new_test.jsonl
decode_log=$ckpt_path/decode_log_test_clean_beam4_repetition_penalty1

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    hydra.run.dir=$ckpt_path \
    model_config.llm_name="vicuna-7b-v1.5" \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name=eat \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector=linear \
    model_config.encoder_projector_ds_rate=5 \
    +model_config.normalize=true \
    dataset_config.dataset=audio_dataset \
    +dataset_config.prompt="Describe the audio you hear. Output the audio caption directly without redundant content. Ensure that the output is not duplicated." \
    dataset_config.val_data_path=$val_data_path \
    dataset_config.fbank_mean=-4.268 \
    dataset_config.fbank_std=4.569 \
    dataset_config.model_name=eat \
    dataset_config.inference_mode=true \
    +dataset_config.normalize=true \
    +dataset_config.input_type=mel \
    train_config.model_name=aac \
    train_config.batching_strategy=custom \
    train_config.num_epochs=1 \
    train_config.val_batch_size=4 \
    train_config.num_workers_dataloader=4 \
    train_config.output_dir=$output_dir \
    +ckpt_path=$ckpt_path/model.pt \
    +decode_log=$decode_log \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \

# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \