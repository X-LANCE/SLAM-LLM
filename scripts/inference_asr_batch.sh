#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

# llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
# llm_path=/nfs/maziyang.mzy/models/Llama-2-7b-chat-hf
# llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-Chat-v0.4

output_dir=/nfs/maziyang.mzy/exps/TinyLlama-1.1B-Chat-v0.4-finetune-asr-ds5-proj2048-lr1e-4-freeze-whisper-large-v2-prompt-padding30-20240115
ckpt_path=$output_dir/asr/2
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4
val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_test_other_filtered.jsonl
decode_log=$ckpt_path/decode_log_test_other_beam4_repetition_penalty1

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
--model_name asr \
--freeze_encoder \
--llm_name tinyllama-1.1b-chat-v0.4 \
--llm_path $llm_path \
--llm_dim 2048 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset speech_dataset \
--speech_dataset.file src/llama_recipes/datasets/speech_dataset_inference.py:get_speech_dataset \
--speech_dataset.val_data_path $val_data_path \
--batching_strategy custom \
--num_epochs 1 \
--val_batch_size 4 \
--num_workers_dataloader 4 \
--output_dir $output_dir \
--ckpt_path $ckpt_path/model.pt \
--decode_log $decode_log \
--freeze_llm \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \