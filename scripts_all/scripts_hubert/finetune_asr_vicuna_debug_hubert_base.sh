#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/hubert_ckpt/hubert_base_ls960.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/debug

# 
python -m debugpy --listen 5679 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=hubert \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=768 \
++model_config.encoder_type="pretrain" \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=hubert_dataset \
++dataset_config.file="src/llama_recipes/datasets/hubert_dataset.py:get_speech_dataset" \
++dataset_config.train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
++dataset_config.val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other.jsonl \
++dataset_config.normalize=false \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=20 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++metric=acc \
