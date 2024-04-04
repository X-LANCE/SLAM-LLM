#!/bin/bash
# export PYTHONPATH=/opt/conda/envs/for_av/lib/python3.8/site-packages:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM

# speech_encoder_path=/nfs/yangguanrou.ygr/hubert_ckpt/hubert_finetune_best/new_checkpoint_best.pt
speech_encoder_path=/nfs/yangguanrou.ygr/av_hubert/large_vox_433h.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-large_vox_433h-0131-qformer

#-m debugpy --listen 5679 --wait-for-client
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/conf_avsr" \
--config-name "avsr.yaml" \
hydra.run.dir=$output_dir \
model_config.llm_name="vicuna-7b-v1.5" \
model_config.llm_path=$llm_path \
model_config.llm_dim=4096 \
model_config.encoder_name=av_hubert \
model_config.encoder_path=$speech_encoder_path \
model_config.encoder_dim=1024 \
model_config.encoder_projector=q-former \
dataset_config.fix_length_audio=88 \
dataset_config.dataset=avsr_dataset \
dataset_config.modal=VO \
model_config.modal=VO \
dataset_config.prompt="Transcribe video to text." \
train_config.model_name=asr \
train_config.freeze_encoder=true \
train_config.freeze_llm=true \
train_config.batching_strategy=custom \
train_config.warmup_steps=1000 \
train_config.total_steps=100000 \
train_config.lr=1e-4 \
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
log_config.wandb_entity_name=yanghaha \
log_config.wandb_project_name=slam-llm \
log_config.wandb_exp_name=vicuna-7b-v1.5-large_vox_433h-0131-qformer \
log_config.log_interval=10 \


#hydra.run.dir=/root/SLAM-LLM/src/llama_recipes/models/avhubert \

# q-former 怎么都跑不起来. cao 但很有可能是prompt的问题