#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

SLAM_DIR=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/SLAM-LLM
cd $SLAM_DIR
code_dir=examples/seld_spatialsoundqa

stage=classification
qa_data_root=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/data/SpatialAudio/closed-end
reverb_data_root=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/data/SpatialAudio/reverb/mp3d
anechoic_data_root=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/data/AudioSet

audio_encoder_path=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/models/SpatialAST/SpatialAST.pth
llm_path=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/models/llama-2-hf

split=eval
# output_dir=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/SLAM-LLM/outputs/bat-vicuna-7b-v1.5-spatialAST-qformer-steplrwarmupkeep1e-4-${stage}-$(date +"%Y%m%d")
output_dir=/mnt/cloudstorfs/sjtu_home/zhisheng.zheng/SLAM-LLM/outputs/bat-llama-2-spatialAST-qformer-steplrwarmupkeep1e-4-classification-20240507
ckpt_path=$output_dir/bat_epoch_2_step_2576
decode_log=$ckpt_path/decode_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
python -u $code_dir/inference_seld_batch.py \
        --config-path "conf" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=llama-2-7b \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=SpatialAST \
        ++model_config.encoder_projector=q-former \
        ++model_config.encoder_ckpt=$audio_encoder_path \
        ++dataset_config.stage=$stage \
        ++dataset_config.qa_data_root=$qa_data_root \
        ++dataset_config.anechoic_data_root=$anechoic_data_root \
        ++dataset_config.reverb_data_root=$reverb_data_root \
        ++dataset_config.fix_length_audio=64 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=bat \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=8 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++train_config.use_peft=true \
        ++peft_config.peft_method=llama_adapter \
        ++log_config.log_file=$output_dir/test.log \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
