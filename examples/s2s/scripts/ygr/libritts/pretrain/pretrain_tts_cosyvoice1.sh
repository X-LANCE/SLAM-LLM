#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

run_dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM
cd $run_dir
code_dir=examples/s2s

speech_encoder_path="small"   # whisper small
llm_path="/nfs/yangguanrou.ygr/ckpts/Qwen/Qwen2-0.5B"


# vocabulary settings
code_layer=1            # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_vocabsize=156160  # 152000 + 4160 Sry: Here is not elegant to set the total_vocabsize manually, I may fix it later :)

# code settings
code_type=CosyVoice     # CosyVoice or SNAC
num_latency_tokens=1    # number of latency tokens (in front of the generated audio tokens)
do_layershift=false      # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# upsample settings
upsample_text_tokens=false
upsampling_factor=1
upsample_method=repeat  # repeat or blank

# train_data_path=/home/v-wenxichen/data/s2s/test/test_train.jsonl
# val_data_path=/home/v-wenxichen/data/s2s/test/test_val.jsonl
train_data_path=/nfs/yangguanrou.ygr/data/libritts/cosyvoice_semantic_token/train.jsonl
val_data_path=/nfs/yangguanrou.ygr/data/libritts/cosyvoice_semantic_token/dev-clean.jsonl
load_from_cache_file=true

batch_size_training=4
use_fp16=false
lr=5e-4
train_audio_embed_only=false
train_embed_only=false
tts_adapter=false
task_type=tts

exp_name="tts_train_v1"
# exp_name="debug"
# exp_name="single_test_tts"

home_dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts
# output_dir=$home_dir/$(TZ='Asia/Shanghai' date +"%Y_%m_%d")/$(TZ='Asia/Shanghai' date +"%H_%M_%S")
output_dir=$home_dir/$(TZ='Asia/Shanghai' date +"%Y_%m_%d_%H_%M_%S")/$exp_name
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
++model_config.vocab_config.code_layer=$code_layer \
++model_config.vocab_config.total_vocabsize=$total_vocabsize \
++model_config.code_type=$code_type \
++dataset_config.dataset=speech_dataset_s2s \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=80 \
++dataset_config.seed=42 \
++dataset_config.manifest_format=jsonl \
++dataset_config.split_size=0.01 \
++dataset_config.load_from_cache_file=$load_from_cache_file \
++dataset_config.task_type=$task_type \
++dataset_config.upsample_text_tokens=$upsample_text_tokens \
++dataset_config.upsampling_factor=$upsampling_factor \
++dataset_config.upsample_method=$upsample_method \
++dataset_config.vocab_config.code_layer=$code_layer \
++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.code_type=$code_type \
++dataset_config.num_latency_tokens=$num_latency_tokens \
++dataset_config.do_layershift=$do_layershift \
++train_config.model_name=s2s \
++train_config.num_epochs=10 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=3000 \
++train_config.total_steps=300000 \
++train_config.lr=$lr \
++train_config.validation_interval=3000 \
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
++log_config.wandb_entity_name=yanghaha \
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
        --nproc_per_node 4 \
        --master_port=29503 \
        $code_dir/finetune_s2s.py \
        --config-path "conf" \
        --config-name "prompt_tts.yaml" \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false \
        $hydra_args
fi

# ++train_config.use_fp16=true \
# bash /home/v-wenxichen/SLAM-LLM/examples/s2s/scripts/pretrain_tts_debug.sh