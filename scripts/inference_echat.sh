#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM

# speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/base.pt
speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
output_dir=/nfs/maziyang.mzy/models/llama-2-hf-finetune

# -m debugpy --listen 5678 --wait-for-client
#python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/finetune.py \
python  src/llama_recipes/pipeline/inference.py \
--model_name echat \
--freeze_llm \
--use_fp16 \
--quantization \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--dataset custom_dataset \
--custom_dataset.file src/llama_recipes/datasets/speech_text_dataset.py:get_audio_dataset \
--custom_dataset.data_path /nfs/zhifu.gzf/data/IEMOCAP_full_release/datalist.jsonl \
--batching_strategy custom \
--custom_dataset.max_words 1024 \
--num_epochs 1 \
--batch_size_training 2 \
--output_dir $output_dir \
--ckpt_path "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/1/model.pt" \
--wav_path "/nfs/zhifu.gzf/data/IEMOCAP_full_release/Session5/sentences/wav/Ses05M_impro04/Ses05M_impro04_F035.wav"
# --peft_ckpt "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/1"
# --use_peft --peft_method lora \