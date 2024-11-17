#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM
code_dir=examples/contextual_asr

speech_encoder_path=/nfs/yangguanrou.ygr/ckpts/wavlm_large_ft_libri960_char/wavlm_large_ft_libri960_char.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl
val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other.jsonl

output_dir=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-libri960-ft-char-hotwords-$(date +"%Y%m%d")-debug

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
++dataset_config.dataset=hotwords_dataset \
++dataset_config.file=src/slam_llm/datasets/hotwords_dataset.py:get_speech_dataset \
++train_config.model_name=asr \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=8000 \
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
++log_config.wandb_exp_name=vicuna-7b-v1.5-WavLM-Large-libri960-ft-char-hotwords \
++log_config.log_interval=5 \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_contextual_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29504 \
        $code_dir/finetune_contextual_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi
