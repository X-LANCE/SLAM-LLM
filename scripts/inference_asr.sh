#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
output_dir=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference.py \
--model_name asr \
--freeze_llm \
--freeze_encoder \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--output_dir $output_dir \
--ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048/asr/13/model.pt" \
--wav_path "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0032.wav" \
--prompt "<|ASR|>" \
# --peft_ckpt "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/1" \
# --use_peft --peft_method lora \