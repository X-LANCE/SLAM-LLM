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


output_dir=/nfs/chengxize.cxz/exp/vicuna-7b-v1.5-large_vox_433h-VO-convlinear
# ckpt_path=$output_dir/avsr/3
ckpt_path=$output_dir/vicuna7B-vsr/5
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4
# val_data_path= ??
decode_log=$ckpt_path/decode_log_test

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
--config-path "/root/SLAM-LLM/scripts_avhubert/conf/vsr" \
--config-name "vicuna7B-vsr.yaml" \
model_config.encoder_path=$speech_encoder_path \
model_config.encoder_projector=cov1d-linear \
train_config.enable_ddp=false \
train_config.num_epochs=1 \
train_config.val_batch_size=16 \
train_config.num_workers_dataloader=4 \
train_config.output_dir=$output_dir \
dataset_config.inference_mode=true \
dataset_config.test_split='test' \
dataset_config.prompt="Please repeat my words in English." \
+ckpt_path=$ckpt_path/model.pt \
+decode_log=$decode_log 


# --test_split test \

# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \
# --speech_dataset.val_data_path $val_data_path \