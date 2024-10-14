#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/v-wenxichen/anaconda3/envs/slam/lib:$LD_LIBRARY_PATH

run_dir=/home/v-wenxichen/SLAM-LLM
cd $run_dir
code_dir=examples/s2s

speech_encoder_path="small"   # whisper small
llm_path="Qwen/Qwen2-0.5B"


# train_data_path=/home/v-wenxichen/data/s2s/test/test_train.jsonl
# val_data_path=/home/v-wenxichen/data/s2s/test/test_val.jsonl
train_data_path="gpt-omni/VoiceAssistant-400K"
val_data_path="gpt-omni/VoiceAssistant-400K"
load_from_cache_file=true

batch_size_training=4
use_fp16=false
lr=1e-4
train_audio_embed_only=false
train_embed_only=false
tts_adapter=false
task_type=tts

# exp_name="s2s_train_v0"
# exp_name="debug"
exp_name="single_test_tts"

home_dir=/home/v-wenxichen/exp/s2s
# output_dir=$home_dir/$(TZ='Asia/Shanghai' date +"%Y_%m_%d")/$(TZ='Asia/Shanghai' date +"%H_%M_%S")
output_dir=$home_dir/$(TZ='Asia/Shanghai' date +"%Y_%m_%d")/$exp_name
ckpt_path=/home/v-wenxichen/exp/s2s/2024_09_23/s2s_train_test/s2s_epoch_1_step_15000

if [ "$exp_name" = "debug" ]; then
    use_wandb=false
else
    use_wandb=true
fi
wandb_exp_name=$exp_name


hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=qwen2-0.5b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=896 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=768 \
++model_config.encoder_projector=linear \
++model_config.tts_adapter=$tts_adapter \
++dataset_config.dataset=speech_dataset_s2s \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=80 \
++dataset_config.seed=42 \
++dataset_config.manifest_format=datasets \
++dataset_config.split_size=0.01 \
++dataset_config.load_from_cache_file=$load_from_cache_file \
++dataset_config.task_type=$task_type \
++train_config.model_name=s2s \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=120000 \
++train_config.lr=$lr \
++train_config.validation_interval=10000 \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++train_config.use_fp16=$use_fp16 \
++train_config.train_audio_embed_only=$train_audio_embed_only \
++train_config.train_embed_only=$train_embed_only \
++train_config.task_type=$task_type \
++metric=acc \
++log_config.use_wandb=$use_wandb \
++log_config.wandb_entity_name=wxc12 \
++log_config.wandb_project_name=SLAM-Omni \
++log_config.wandb_exp_name=$wandb_exp_name \
++log_config.log_file=$output_dir/exp.log \
++log_config.log_interval=10 \
"
# ++ckpt_path=$ckpt_path/model.pt \
# ↑ this line is for resuming training



if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    if [ "$exp_name" = "debug" ]; then
        python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt_tts.yaml" \
            $hydra_args
    else
        python $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt_tts.yaml" \
            $hydra_args
    fi
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_s2s.py \
        --config-path "conf" \
        --config-name "prompt_tts.yaml" \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false \7
        $hydra_args
fi

# ++train_config.use_fp16=true \
# bash /home/v-wenxichen/SLAM-LLM/examples/s2s/scripts/pretrain_tts_debug.sh