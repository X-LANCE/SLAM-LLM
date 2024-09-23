#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7


run_dir=/data/xiquan.li/SLAM-LLM_new
cd $run_dir
code_dir=examples/drcap_zeroshot_aac

audio_encoder_path=/data/xiquan.li/models/clap/best_model.pt
llm_path=/data/xiquan.li/models/vicuna-7b-v1.5

seed=1234
btz=4
lr=1e-5
encoder_projector_ds_rate=1

train_jsonl_path=/data/xiquan.li/data/rz_cap/rag/audiocaps/train_3_sim_0.75-0.85.jsonl
val_jsonl_path=/data/xiquan.li/data/rz_cap/rag/audiocaps/val_single_3_sim_0.75-0.85.jsonl


exp_name=finetune_drcap_lora
output_dir=/data/xiquan.li/exps/test/${exp_name}


hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=clap \
++model_config.clap_config=$code_dir/conf/clap_config.yaml \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++model_config.encoder_path=$audio_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
++dataset_config.dataset=audio_dataset \
++dataset_config.train_data_path=$train_jsonl_path \
++dataset_config.val_data_path=$val_jsonl_path \
++dataset_config.input_type=text \
++dataset_config.model_name=clap \
++dataset_config.use_rag=true \
++dataset_config.rag_first=true \
++train_config.model_name=aac \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=40000 \
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
++log_config.use_wandb=true \
++metric=acc \
"

# note: to train the linear layer only, you could set '++train_config.use_peft=false' and 'train_config.freeze_llm=true'

# -m debugpy --listen 6666 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python $code_dir/finetune_drcap.py \
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