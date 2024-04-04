#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

export MASTER_ADDR=localhost # Or the actual IP if it's a remote cluster
export MASTER_PORT=12346 # A free port number
export WORLD_SIZE=1 # Assuming you have 4 GPUs
export RANK=0
export LOCAL_RANK=0

cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/av_hubert/large_noise_pt_noise_ft_433h_new.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-av


# torchrun \
# --nnodes 1 \
# --nproc_per_node 4 \
# src/llama_recipes/pipeline/finetune.py \
python -m debugpy --listen 5679 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/conf_avsr" \
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
model_config.modal=AV \
train_config.model_name=asr \
train_config.freeze_encoder=true \
train_config.freeze_llm=true \
train_config.batching_strategy=custom \
train_config.warmup_steps=1000 \
train_config.total_steps=70000 \
train_config.lr=1e-3 \
train_config.scheduler=tri \
train_config.validation_interval=2000 \
train_config.batch_size_training=12 \
train_config.val_batch_size=12 \
train_config.num_workers_dataloader=0 \
train_config.output_dir=$output_dir \
train_config.enable_fsdp=false \
train_config.enable_ddp=true \
train_config.use_fp16=true \
+metric=acc \
# log_config.log_file=/$output_dir/train.log \
# log_config.use_wandb=true \
# log_config.wandb_dir=$output_dir \
# log_config.wandb_entity_name=yanghaha \
# log_config.wandb_project_name=slam-llm-vox \
# log_config.wandb_exp_name=vicuna-7b-v1.5-large_vox_433h-tri-dataset-tiaocan \
# log_config.log_interval=10 \






# cd /root
# cp -r SLAM-LLM/ /nfs/yangguanrou.ygr/codes/