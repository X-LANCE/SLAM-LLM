#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7


run_dir=/data/xiquan.li/SLAM-LLM_new
cd $run_dir
code_dir=examples/aac_audiocaps

audio_encoder_path=/data/xiquan.li/models/EAT-base_epoch30_ft.pt
llm_path=/data/xiquan.li/models/vicuna-7b-v1.5

seed=42
btz=4
lr=1e-4
encoder_projector_ds_rate=5

train_jsonl_path=/root/data/AudioCaps/train.jsonl
val_jsonl_path=/root/data/AudioCaps/val.jsonl


exp_name=finetune_test_lora
output_dir=/data/xiquan.li/exps/test/${exp_name}


hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=eat \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++model_config.encoder_path=$audio_encoder_path \
++model_config.encoder_dim=768 \
++model_config.encoder_projector=linear \
++dataset_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
++dataset_config.dataset=audio_dataset \
++dataset_config.train_data_path=$train_jsonl_path \
++dataset_config.val_data_path=$val_jsonl_path \
++dataset_config.input_type=mel \
++dataset_config.fbank_mean=-4.268 \
++dataset_config.fbank_std=4.569 \
++dataset_config.model_name=eat \
++dataset_config.fixed_length=true \
++dataset_config.target_length=1024 \
++train_config.model_name=aac \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=$lr \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=$btz \
++train_config.val_batch_size=$btz \
++train_config.num_workers_dataloader=4 \
++train_config.use_fp16=true \
++train_config.output_dir=$output_dir \
++train_config.seed=${seed} \
++train_config.use_peft=true \
++log_config.log_file="${output_dir}/train.log" \
++log_config.wandb_dir=${output_dir} \
++log_config.wandb_entity_name=x-lance-lxq \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=$exp_name \
++log_config.use_wandb=false \
++metric=acc \
"

# note: to train the linear layer only, you could set '++train_config.use_peft=false' and 'train_config.freeze_llm=true'

# -m debugpy --listen 6666 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python $code_dir/finetune_aac.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=true \
        $hydra_args
fi
