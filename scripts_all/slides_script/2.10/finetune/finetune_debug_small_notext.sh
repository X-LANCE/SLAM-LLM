#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/yangguanrou.ygr/hubert_ckpt/hubert_xtralarge_ll60k_finetune_ls960.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/slides-finetune-20230125-debug

# -m debugpy --listen 5678 --wait-for-client
python -m debugpy --listen 5679 --wait-for-client src/llama_recipes/pipeline/finetune.py \
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
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.batch_size_training=2 \
++train_config.val_batch_size=2 \
++train_config.num_workers_dataloader=0 \
++train_config.lr=1e-4 \
++train_config.output_dir=$output_dir \
++metric=acc \
++dataset_config.train_scp_file_path=/nfs/yangguanrou.ygr/slidespeech/train_debug/ \
++dataset_config.dev_scp_file_path=/nfs/yangguanrou.ygr/slidespeech/dev_debug/ \
++dataset_config.use_ocr=false \