#!/bin/bash
set -x
pip install wandb
export WANDB_API_KEY='cc633438a8688eaf73ea5889022da2c394a92df2'
export HYDRA_FULL_ERROR=1
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
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

rm -r /root/SLAM-LLM
cp -r /nfs/yangguanrou.ygr/codes/SLAM-LLM/ /root/
cd /root/SLAM-LLM
pip install -e .
code_dir=examples/hotwords_librispeech


speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Large.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
train_data_path=/nfs/maziyang.mzy/data/gigaspeech/gigaspeech_train_1000h.jsonl
val_data_path=/nfs/maziyang.mzy/data/gigaspeech/gigaspeech_dev.jsonl

output_dir=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-lr5e-5-ws10000-mzy-$(date +"%Y%m%d")

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
 

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=cov1d-linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=2 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=10000 \
++train_config.total_steps=100000 \
++train_config.lr=5e-5 \
++train_config.validation_interval=6000 \
++train_config.val_batch_size=4 \
++train_config.batch_size_training=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
++log_config.log_file=/$output_dir/train.log \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir \
++log_config.wandb_entity_name=yanghaha \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=vicuna-7b-v1.5-WavLM-Large-gigaspeech-lr5e-5-ws10000-mzy \
++log_config.log_interval=5 \
"



torchrun $DISTRIBUTED_ARGS \
$code_dir/finetune_asr.py \
--config-path "conf" \
--config-name "prompt.yaml" \
++train_config.enable_fsdp=false \
++train_config.enable_ddp=true \
++train_config.use_fp16=true \
$hydra_args

