#!/bin/bash
set -x
pip install wandb
export WANDB_API_KEY='c47ab15d9059a2894bdb7db1b190e71fd197c2b3'
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

exp_name=vicuna-7b-v1.5-large_vox_433h-VO-linear
output_dir=/nfs/chengxize.cxz/exp/$exp_name

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
--config-path "/root/SLAM-LLM/scripts_avhubert/conf/vsr" \
--config-name "vicuna7B-vsr.yaml" \
hydra.run.dir=$output_dir \
model_config.encoder_projector=linear \
train_config.output_dir=$output_dir \
+metric=acc \
log_config.log_file=/$output_dir/log \
log_config.wandb_dir=$output_dir \
log_config.wandb_exp_name=$exp_name