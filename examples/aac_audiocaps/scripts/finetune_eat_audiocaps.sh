#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7


cd /root/SLAM-LLM

audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M.pt
llm_path=/root/models/vicuna-7b-v1.5

seed=42
btz=4
lr=1e-4
encoder_projector_ds_rate=5
long_prompt="Describe the audio you hear. Output the audio caption directly without redundant content. Ensure that the output is not duplicated."

exp_name=finetune_test

train_jsonl_path=/root/data/AudioCaps/train.jsonl
val_jsonl_path=/root/data/AudioCaps/val.jsonl

# category=audiocaps
category=test

output_dir=/root/exps/${category}/${exp_name}


# -m debugpy --listen 6666 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python /root/SLAM-LLM/src/slam_llm/pipeline/finetune.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    hydra.run.dir=$output_dir \
    model_config.llm_name='vicuna-7b-v1.5' \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name='eat' \
    model_config.encoder_ds_rate=2 \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector='linear' \
    model_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
    dataset_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
    +dataset_config.input_type=mel \
    dataset_config.dataset='audio_dataset' \
    dataset_config.train_data_path=${train_jsonl_path} \
    dataset_config.val_data_path=${val_jsonl_path} \
    dataset_config.prompt="${long_prompt}" \
    dataset_config.fbank_mean=-4.268 \
    dataset_config.fbank_std=4.569 \
    dataset_config.model_name=eat \
    train_config.model_name='aac' \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \
    train_config.batching_strategy='custom' \
    train_config.warmup_steps=1000 \
    train_config.total_steps=100000 \
    train_config.lr=$lr \
    train_config.validation_interval=1000 \
    train_config.batch_size_training=$btz \
    train_config.val_batch_size=$btz \
    train_config.num_workers_dataloader=4 \
    train_config.use_fp16=true \
    train_config.output_dir=$output_dir \
    log_config.log_file="${output_dir}/train.log" \
    train_config.seed=${seed} \
    log_config.wandb_dir=${output_dir} \
    log_config.wandb_entity_name=wxc12 \
    log_config.wandb_project_name=slam-llm \
    log_config.wandb_exp_name=$exp_name \
    dataset_config.fixed_length=true \
    dataset_config.target_length=1024 \
    log_config.use_wandb=true \

else
torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    src/llama_recipes/pipeline/finetune.py \
    --model_name aac \
    --freeze_encoder \
    --freeze_llm \
    --enable_fsdp \
    --llm_name llama-2-7b-hf \
    --llm_path $llm_path \
    --llm_dim 4096 \
    --encoder_name eat \
    --encoder_ds_rate 2 \
    --encoder_path $audio_encoder_path \
    --encoder_dim 768 \
    --encoder_projector linear \
    --encoder_projector_ds_rate 5 \
    --dataset audio_dataset \
    --audio_dataset.train_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
    --audio_dataset.val_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl \
    --batching_strategy custom \
    --num_epochs 100 \
    --batch_size_training 4 \
    --val_batch_size 4 \
    --num_workers_dataloader 4 \
    --lr 1e-4 \
    --output_dir $output_dir \
    --metric acc \
    --log_file /$output_dir/train.log \
    --use_wandb \
    --wandb_dir $output_dir \
    --wandb_entity_name wxc12 \
    --wandb_project_name slam-llm \
    --wandb_exp_name $exp_name \
    --log_interval 5 \

fi
