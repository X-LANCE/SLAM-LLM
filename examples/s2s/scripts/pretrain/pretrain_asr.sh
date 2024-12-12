#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/v-wenxichen/anaconda3/envs/slam/lib:$LD_LIBRARY_PATH
export WANDB_API_KEY=406faa59cf62a3646fa3479a7e133c4cf5a77100       # please replace with your own wandb key thxxxx, unless you want to share your experiment results with me :)

code_dir=examples/s2s
num_gpus_per_node=$(( $(echo ${CUDA_VISIBLE_DEVICES} | tr -cd ',' | wc -c) + 1 ))
num_nodes=1
num_gpus=$(( num_gpus_per_node * num_nodes ))

whisper_size=small                  # tiny base small medium large-v3
speech_encoder_path="/valleblob/v-wenxichen/models/whisper/${whisper_size}.pt"   # different whisper size
llm_path="/valleblob/v-wenxichen/models/models--Qwen--Qwen2-0.5B/snapshots/ff3a49fac17555b8dfc4db6709f480cc8f16a9fe"  # Qwen/Qwen2-0.5B, you can choose other Qwen models (Qwen2 or Qwen2.5)
llm_name=Qwen2-0.5b

encoder_dim=768                     # 384 512 768 1024 1280
mel_size=80                         # 80 128 ( only whisper-large-v3 supports 128 )
llm_dim=896                         # 896 1536 3584 8192  -> 0.5B 1.5B 3B 7B

# vocabulary settings
code_layer=3                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=4160          # the vocab size of the codec token
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen2 here)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# code settings
code_type=CosyVoice                 # CosyVoice or SNAC
num_latency_tokens=5                # number of latency tokens (in front of the generated audio tokens)
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# dataset settings
train_data_path="/valleblob/v-wenxichen/data/s2s/VoiceAssistant-400K-v2-arrow"
val_data_path="/valleblob/v-wenxichen/data/s2s/VoiceAssistant-400K-v2-arrow"
load_from_cache_file=false          # set to true if you have already generated the cache file, otherwise set to false

# training settings
batch_size_training=6
use_fp16=true
use_peft=false
num_epochs=10
lr=1e-4
task_type=asr
warmup_steps=1000
total_steps=100000

# validation settings
validation_interval=3000
split_size=0.01

# model settings
group_decode=false
group_decode_adapter_type=linear

# log settings
exp_name="asr-${llm_name}-gpu${num_gpus}-btz${batch_size_training}-lr${lr}-nofp16-epochs${num_epochs}-whisper_${whisper_size}-latency${num_latency_tokens}-group${code_layer}"
if [ "$use_fp16" = true ]; then
    exp_name="asr-${llm_name}-gpu${num_gpus}-btz${batch_size_training}-lr${lr}-fp16-epochs${num_epochs}-whisper_${whisper_size}-latency${num_latency_tokens}-group${code_layer}"
fi
# exp_name="debug"
wandb_entity_name=wxc12
wandb_project_name=SLAM-Omni

home_dir=/valleblob/v-wenxichen/exp/asr
# output_dir=$home_dir/$(TZ='Asia/Shanghai' date +"%Y_%m_%d")/$(TZ='Asia/Shanghai' date +"%H_%M_%S")
output_dir=$home_dir/$exp_name
# ckpt_path=/valleblob/v-wenxichen/exp/s2s/s2s_train_v3-gpu16-btz3-lr5e-4-fp16-epochs10-whisper_small-latency5-group3/gpu16-btz3-lr5e-4-fp16-epochs10-whisper_small-latency5-group3-s2s_epoch_4_step_1179  # this line is for resuming training

if [ "$exp_name" = "debug" ]; then
    use_wandb=false
else
    use_wandb=true
fi
wandb_exp_name=$exp_name

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=linear \
++model_config.vocab_config.code_layer=$code_layer \
++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++model_config.vocab_config.total_vocabsize=$total_vocabsize \
++model_config.code_type=$code_type \
++model_config.group_decode=$group_decode \
++model_config.group_decode_adapter_type=$group_decode_adapter_type \
++dataset_config.dataset=speech_dataset_s2s \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=$mel_size \
++dataset_config.seed=42 \
++dataset_config.manifest_format=datasets \
++dataset_config.split_size=$split_size \
++dataset_config.load_from_cache_file=$load_from_cache_file \
++dataset_config.task_type=$task_type \
++dataset_config.vocab_config.code_layer=$code_layer \
++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.code_type=$code_type \
++dataset_config.num_latency_tokens=$num_latency_tokens \
++dataset_config.do_layershift=$do_layershift \
++train_config.model_name=s2s \
++train_config.num_epochs=$num_epochs \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=$warmup_steps \
++train_config.total_steps=$total_steps \
++train_config.lr=$lr \
++train_config.validation_interval=$validation_interval \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=$batch_size_training \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++train_config.use_fp16=$use_fp16 \
++train_config.task_type=$task_type \
++train_config.use_peft=$use_peft \
++metric=acc \
++log_config.use_wandb=$use_wandb \
++log_config.wandb_entity_name=$wandb_entity_name \
++log_config.wandb_project_name=$wandb_project_name \
++log_config.wandb_exp_name=$wandb_exp_name \
++log_config.wandb_dir=$output_dir \
++log_config.log_file=$output_dir/exp.log \
++log_config.log_interval=100 \
"
# ++ckpt_path=$ckpt_path/model.pt \
# ↑ this line is for resuming training


if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    if [ "$exp_name" = "debug" ]; then
        python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt_asr.yaml" \
            $hydra_args
    else
        python $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt_asr.yaml" \
            $hydra_args
    fi
else
    torchrun \
        --nnodes $num_nodes \
        --nproc_per_node $num_gpus_per_node \
        --master_port=29503 \
        $code_dir/finetune_s2s.py \
        --config-path "conf" \
        --config-name "prompt_asr.yaml" \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false \
        $hydra_args
fi

# for multi-machine training, you should add the following line to the torchrun command
# --node_rank=$node_rank \
# --master_addr=$master_addr \

# bash ./examples/s2s/scripts/pretrain/pretrain_asr.sh

# 1GPU + 12w steps + btz4 = 1epoch
# 1GPU + 24w steps + btz2 = 1epoch 

# 40GB max batch size = 2
# 80GB max batch size = 4

# code_path -> cd /tmp/amlt-code-download