#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
# export CUDA_LAUNCH_BLOCKING=1

rm -r /root/SLAM-LLM
cp -r /nfs/chengxize.cxz/projects/SLAM-LLM/ /root/
cd /root/SLAM-LLM
pip install -e .

cd /nfs/yangguanrou.ygr/av_hubert/fairseq
pip install --editable ./
pip install python_speech_features
pip list
cd /root/SLAM-LLM

# speech_encoder_path= ???

# llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
# llm_path=/nfs/maziyang.mzy/models/Llama-2-7b-chat-hf

speech_encoder_path=/nfs/yangguanrou.ygr/av_hubert/self_large_vox_433h.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5


output_dir=/nfs/chengxize.cxz/exp/vicuna-7b-v1.5-large_vox_433h-VO
# ckpt_path=$output_dir/avsr/3
ckpt_path=$output_dir/asr/1
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4
# val_data_path= ??
decode_log=$ckpt_path/decode_log_test_other_beam4_repetition_penalty1

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
--config-path "/root/SLAM-LLM/scripts/conf" \
--config-name "avsr.yaml" \
model_config.llm_name="vicuna-7b-v1.5" \
model_config.llm_path=$llm_path \
model_config.llm_dim=4096 \
model_config.encoder_name=av_hubert \
model_config.encoder_path=$speech_encoder_path \
model_config.encoder_dim=1024 \
model_config.encoder_projector=cov1d-linear \
model_config.encoder_projector_ds_rate=5 \
train_config.model_name=asr \
train_config.freeze_encoder=true \
train_config.batching_strategy=custom \
train_config.num_epochs=1 \
train_config.val_batch_size=4 \
train_config.num_workers_dataloader=4 \
train_config.output_dir=$output_dir \
train_config.freeze_llm=true \
dataset_config.dataset=avhubert_dataset \
dataset_config.file="src/llama_recipes/datasets/avhubert_dataset.py:get_audio_dataset" \
dataset_config.inference_mode=true \
dataset_config.test_split='test' \
dataset_config.prompt='Transcribe the silent speech in this video to Spanish' \
+ckpt_path=$ckpt_path/model.pt \
+decode_log=$decode_log 


# --test_split test \

# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \
# --speech_dataset.val_data_path $val_data_path \