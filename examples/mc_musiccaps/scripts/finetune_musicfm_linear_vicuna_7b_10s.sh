#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=$PWD
cd $run_dir
code_dir=examples/music_caption

music_encoder_path=path/to/pretrained/musicfm/pretrained_msd.pt
music_encoder_stat_path=path/to/pretrained/musicfm/msd_stats.json
music_encoder_config_path=facebook/wav2vec2-conformer-rope-large-960h-ft

llm_path=lmsys/vicuna-7b-v1.5


train_data_path=/root/cq7_haina/data/LP-MusicCaps-MC/LP-MusicCaps-MC.train.exist.jsonl
val_data_path=/root/cq7_haina/data/LP-MusicCaps-MC/LP-MusicCaps-MC.valid.exist.jsonl

output_dir=/root/cq7_haina/save/music-caption/musicfm_vicuna7b_mc_10s_$(date +"%Y%m%d_%H:%M:%S")


hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_path=$llm_path \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_dim=4096 \
++model_config.encoder_name=musicfm \
++model_config.normalize=false \
++model_config.encoder_layer_idx=9 \
++dataset_config.normalize=false \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$music_encoder_path \
++model_config.encoder_stat_path=$music_encoder_stat_path \
++model_config.encoder_config_path=$music_encoder_config_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=mir_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++dataset_config.fixed_duration=10.0 \
++dataset_config.audio_label_freq=25 \
++train_config.model_name=mir \
++train_config.num_epochs=10000 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=3000 \
++train_config.batch_size_training=1 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_mir.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_mir.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=false \
        $hydra_args
fi
