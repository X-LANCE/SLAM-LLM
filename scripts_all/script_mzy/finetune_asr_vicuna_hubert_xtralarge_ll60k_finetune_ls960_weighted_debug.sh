#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1


export MASTER_ADDR=localhost # Or the actual IP if it's a remote cluster
export MASTER_PORT=12346 # A free port number
export WORLD_SIZE=1 # Assuming you have 4 GPUs
export RANK=0
export LOCAL_RANK=0


cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/ckpts/hubert_ckpt/hubert_xtralarge_ll60k_finetune_ls960.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-hubert_xtralarge_ll60k_finetune_ls960_weighted

python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=hubert_weighted \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_type="finetune" \
++model_config.output_layer=48 \
++encoder_projector=linear \
++encoder_projector_ds_rate=5 \
++dataset_config.dataset=hubert_dataset \
++dataset_config.file="src/llama_recipes/datasets/hubert_dataset.py:get_speech_dataset" \
++dataset_config.train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
++dataset_config.val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other.jsonl \
++dataset_config.normalize=true \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++train_config.enable_fsdp=false \
++train_config.enable_ddp=true \
++train_config.use_fp16=true \
++metric=acc \
# ++log_config.log_file=/$output_dir/train.log \
# ++log_config.use_wandb=true \
# ++log_config.wandb_dir=$output_dir \
# ++log_config.wandb_entity_name=yanghaha \
# ++log_config.wandb_project_name=slam-llm \
# ++log_config.wandb_exp_name=vicuna-7b-v1.5-hubert_xtralarge_ll60k_finetune_ls960_weighted \
# ++log_config.log_interval=5 \





# cd /root
# cp -r SLAM-LLM/ /nfs/yangguanrou.ygr/codes/