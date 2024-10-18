#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

SLAM_DIR=/path/to/SLAM-LLM
cd $SLAM_DIR
code_dir=examples/seld_spatialsoundqa

audio_encoder_path=/data1/scratch/zhisheng/models/SpatialAST/SpatialAST.pth # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialAST/finetuned.pth
llm_path=/home/zhisheng/models/llama-2-hf # https://huggingface.co/meta-llama/Llama-2-7b-hf

stage=stage3-mixup
qa_data_root=/data3/scratch/zhisheng/SpatialAudio/SpatialSoundQA/closed-end # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/closed-end
reverb_data_root=/data3/scratch/zhisheng/SpatialAudio/SpatialSoundQA/mp3d_reverb # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
anechoic_data_root=/data3/scratch/zhisheng/SpatialAudio/SpatialSoundQA/AudioSet # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/AudioSet

split=eval-stage3-distance-direction
output_dir=./outputs/bat-llama-2-spatialAST-8qformer-steplrwarmupkeep1e-4-${stage}

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=llama-2-7b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=SpatialAST \
++model_config.encoder_projector=q-former \
++model_config.qformer_layers=8 \
++model_config.encoder_ckpt=$audio_encoder_path \
++dataset_config.test_split=${split} \
++dataset_config.stage=$stage \
++dataset_config.qa_data_root=$qa_data_root \
++dataset_config.anechoic_data_root=$anechoic_data_root \
++dataset_config.reverb_data_root=$reverb_data_root \
++dataset_config.max_words=96 \
++dataset_config.fix_length_audio=64 \
++train_config.model_name=bat \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=20000 \
++train_config.total_steps=200000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=2000 \
++train_config.batch_size_training=8 \
++train_config.val_batch_size=8 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.use_peft=true \
++peft_config.peft_method=lora \
++metric=acc \
++log_config.log_file=$output_dir/log.txt \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -u -m debugpy --listen 55555 --wait-for-client $code_dir/finetune_seld.py \
        --config-path "conf" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 8 \
        --master_port=39503 \
        $code_dir/finetune_seld.py \
        --config-path "conf" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=false \
        $hydra_args
fi
