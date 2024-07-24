#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/vsr_LRS3

speech_encoder_path=/nfs/yangguanrou.ygr/codes/av_hubert/large_vox_433h_new.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/root/tmp/vicuna-7b-v1.5-large_vox_433h_5e-5_4gpu-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
+model_config.llm_name=vicuna-7b-v1.5 \
+model_config.llm_path=$llm_path \
+model_config.llm_dim=4096 \
+model_config.encoder_name=av_hubert \
+model_config.encoder_path=$speech_encoder_path \
+model_config.encoder_dim=1024 \
+model_config.encoder_projector=cov1d-linear \
+model_config.encoder_projector_ds_rate=5 \
+dataset_config.dataset=avhubert_dataset \
+dataset_config.labels=[\"wrd\"] \
+train_config.model_name=vsr \
+train_config.num_epochs=10 \
+train_config.freeze_encoder=true \
+train_config.freeze_llm=false \
+train_config.use_peft=true \
+train_config.peft_config.r=32 \
+train_config.batching_strategy=custom \
+train_config.warmup_steps=1000 \
+train_config.total_steps=70000 \
+train_config.lr=5e-5 \
+train_config.validation_interval=2000 \
+train_config.batch_size_training=12 \
+train_config.val_batch_size=12 \
+train_config.num_workers_dataloader=0 \
+train_config.output_dir=$output_dir \
+metric=acc \
+log_config.log_file=/$output_dir/train.log \
+log_config.use_wandb=true \
+log_config.wandb_dir=$output_dir \
+log_config.wandb_entity_name=yanghaha \
+log_config.wandb_project_name=slam-llm \
+log_config.wandb_exp_name=large_vox_433h_lora_5e-5_4gpu \
+log_config.log_interval=5 \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_vsr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 4 \
        --master_port=29503 \
        $code_dir/finetune_vsr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        +train_config.enable_fsdp=false \
        +train_config.enable_ddp=true \
        +train_config.use_fp16=true \
        $hydra_args
fi



# 可以