#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/SLAM-LLM/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

cd /SLAM-LLM

speech_encoder_path=/cxgroup/model/whisper/large-v3.pt
llm_path=/cxgroup/model/vicuna-7b-v1.5

output_dir=/exps/vicuna-7b-v1.5-finetune-asr-linear-lora-16-steplrwarmupkeep1e-4-whisper-largev3-$(date +"%Y%m%d")-test
ckpt_path=$output_dir/asr/4
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102-renew5/asr/1

python src/llama_recipes/pipeline/inference.py \
--config-path "/SLAM-LLM/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.fix_length_audio=-1 \
++ckpt_path=$ckpt_path/model.pt \
++wav_path="/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0032.wav" \
++prompt="Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. " \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
# ++model_config.encoder_projector=linear \
# ++model_config.encoder_projector_ds_rate=5 \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \