#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

# llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/maziyang.mzy/exps/vicuna-7b-v1.5-finetune-asr-ds5-proj2048-lr1e-4-whisper-prompt-padding30-20240106
ckpt_path=/nfs/maziyang.mzy/exps/vicuna-7b-v1.5-finetune-asr-ds5-proj2048-lr1e-4-whisper-prompt-padding30-20240106/asr/2/model.pt
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102-renew5/asr/1

# -m debugpy --listen 5678 --wait-for-client
python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/inference.py \
--model_name asr \
--freeze_encoder \
--freeze_llm \
--llm_name vicuna-7b-v1.5 \
--llm_path $llm_path \
--llm_dim 4096 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--output_dir $output_dir \
--ckpt_path $ckpt_path \
--wav_path "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0032.wav" \
--prompt "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. " \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \