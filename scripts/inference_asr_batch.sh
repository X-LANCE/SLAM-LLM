#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
output_dir=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt-renew5
ckpt_path=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt-renew5/asr/0/model.pt
peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt-renew5/asr/0
val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_test_clean_filtered.jsonl
decode_log=/root/decode_log_test_clean_bs8_beam4_repetition_penalty2

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
--model_name asr \
--freeze_encoder \
--use_peft --peft_method lora \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset custom_dataset \
--custom_dataset.file src/llama_recipes/datasets/speech_dataset_inference.py:get_audio_dataset \
--custom_dataset.val_data_path $val_data_path \
--batching_strategy custom \
--num_epochs 1 \
--val_batch_size 4 \
--num_workers_dataloader 4 \
--output_dir $output_dir \
--ckpt_path $ckpt_path \
--peft_ckpt $peft_ckpt \
--decode_log $decode_log \
# --freeze_llm \