#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/hubert_ckpt/hubert_xtralarge_ll60k_finetune_ls960.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/slides-finetune-20230202

torchrun \
--nnodes 1 \
--nproc_per_node 4 \
src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/slides_conf" \
--config-name "slides.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=hubert \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_type="finetune" \
++encoder_projector=linear \
++encoder_projector_ds_rate=5 \
++dataset_config.dataset=slides_dataset \
++dataset_config.use_ocr=true \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=0 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.output_dir=$output_dir \
++train_config.enable_fsdp=false \
++train_config.enable_ddp=true \
++train_config.use_fp16=true \
++\metric=acc \
++log_config.log_file=/$output_dir/train.log \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir \
++log_config.wandb_entity_name=yanghaha \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=slides-finetune-20230202 \
++log_config.log_interval=5 \

# ++dataset_config.train_scp_file_path=/nfs/yangguanrou.ygr/slidespeech/train_debug/ \
# ++dataset_config.dev_scp_file_path=/nfs/yangguanrou.ygr/slidespeech/dev_debug/ \

# 2024.1.25 8 太大了  load 数据不用一次性
# 4 也会OMM 有的OCR text 太长了 超过了vicuna 最长长度 4096
# validation的时候莫名其妙的会 dataloader's workers are out of shared memory

#++train_config.scheduler="tri" \