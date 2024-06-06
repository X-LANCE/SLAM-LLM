#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

SLAM_DIR=/mnt/lustre/hpc_stor03/sjtu_home/zhisheng.zheng/SLAM-LLM
cd $SLAM_DIR
code_dir=examples/seld_spatialsoundqa

audio_encoder_path=/mnt/lustre/hpc_stor03/sjtu_home/zhisheng.zheng/models/SpatialAST/SpatialAST.pth
llm_path=/mnt/lustre/hpc_stor03/sjtu_pub/cxgroup/model/Llama-2-7b-hf

stage=stage2-single
qa_data_root=/mnt/lustre/hpc_stor03/sjtu_home/zhisheng.zheng/data/SpatialAudio/closed-end
reverb_data_root=/mnt/lustre/hpc_stor03/sjtu_home/zhisheng.zheng/data/SpatialAudio/reverb/mp3d
anechoic_data_root=/mnt/lustre/hpc_stor03/sjtu_home/zhisheng.zheng/data/AudioSet

ckpt_path=/mnt/lustre/hpc_stor03/sjtu_home/zhisheng.zheng/SLAM-LLM/outputs/bat-llama-2-spatialAST-8qformer-steplrwarmupkeep1e-4-stage1-clsdoa-20240519/bat_epoch_3_step_3288
output_dir=${SLAM_DIR}/outputs/bat-llama-2-spatialAST-8qformer-steplrwarmupkeep1e-4-${stage}-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=llama-2-7b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=SpatialAST \
++model_config.encoder_projector=q-former \
++model_config.encoder_ckpt=$audio_encoder_path \
++dataset_config.stage=$stage \
++dataset_config.qa_data_root=$qa_data_root \
++dataset_config.anechoic_data_root=$anechoic_data_root \
++dataset_config.reverb_data_root=$reverb_data_root \
++dataset_config.fix_length_audio=64 \
++train_config.model_name=bat \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=10000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=16 \
++train_config.val_batch_size=16 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.use_peft=true \
++peft_config.peft_method=llama_adapter \
++metric=acc \
++log_config.log_file=$output_dir/log.txt \
++ckpt_path=$ckpt_path/model.pt \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -u -m debugpy --listen 55555 --wait-for-client $code_dir/finetune_seld.py \
        --config-path "conf" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 4 \
        --master_port=29503 \
        $code_dir/finetune_seld.py \
        --config-path "conf" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=false \
        $hydra_args
fi
