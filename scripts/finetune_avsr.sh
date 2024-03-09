#!/bin/bash
set -x
pip install wandb
export WANDB_API_KEY='c47ab15d9059a2894bdb7db1b190e71fd197c2b3'
# pip list
# export PYTHONPATH=/root/whisper:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_LAUNCH_BLOCKING=1
# export OMP_NUM_THREADS=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false
# debug setting for multiple gpus
export NCCL_DEBUG=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=f5
export NCCL_IB_GID_INDEX=3

rm -r /root/SLAM-LLM
cp -r /nfs/chengxize.cxz/projects/SLAM-LLM/ /root/
cd /root/SLAM-LLM
pip install -e .

cd /nfs/yangguanrou.ygr/av_hubert/fairseq
pip install --editable ./
pip install python_speech_features
pip list
cd /root/SLAM-LLM

# 这样是可以跑的！


speech_encoder_path=/nfs/yangguanrou.ygr/av_hubert/self_large_vox_433h.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/chengxize.cxz/exp/vicuna-7b-v1.5-large_vox_433h-VO
# ckpt_path=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-prompt-paddinglrfix8000-20240106/asr/2/model.pt

count=$1
gpu_num=$2

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26666}
"

echo $DISTRIBUTED_ARGS

# python \
torchrun $DISTRIBUTED_ARGS \
src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/conf" \
--config-name "avsr.yaml" \
hydra.run.dir=$output_dir \
model_config.llm_name="vicuna-7b-v1.5" \
model_config.llm_path=$llm_path \
model_config.llm_dim=4096 \
model_config.encoder_name=av_hubert \
model_config.encoder_path=$speech_encoder_path \
model_config.encoder_dim=1024 \
model_config.encoder_projector=cov1d-linear \
model_config.encoder_projector_ds_rate=5 \
dataset_config.dataset=avhubert_dataset \
dataset_config.file="src/llama_recipes/datasets/avhubert_dataset.py:get_audio_dataset" \
model_config.modal=VO \
train_config.model_name=asr \
train_config.freeze_encoder=true \
train_config.freeze_llm=true \
train_config.batching_strategy=custom \
train_config.warmup_steps=1000 \
train_config.total_steps=70000 \
train_config.lr=2e-4 \
train_config.scheduler=tri \
train_config.validation_interval=2000 \
train_config.batch_size_training=8 \
train_config.val_batch_size=8 \
train_config.num_workers_dataloader=0 \
train_config.output_dir=$output_dir \
train_config.enable_fsdp=false \
train_config.enable_ddp=true \
train_config.use_fp16=true \
+metric=acc \
log_config.log_file=/$output_dir/train.log \
log_config.use_wandb=true \
log_config.wandb_dir=$output_dir \
log_config.wandb_entity_name=exgc-cxz299 \
log_config.wandb_project_name=av-llm \
log_config.wandb_exp_name=vicuna-7b-v1.5-large_vox_433h-VO \
log_config.log_interval=10 \