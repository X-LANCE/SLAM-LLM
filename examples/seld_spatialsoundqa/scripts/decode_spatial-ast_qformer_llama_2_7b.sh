#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

SLAM_DIR=/path/to/SLAM-LLM
cd $SLAM_DIR
code_dir=examples/seld_spatialsoundqa

audio_encoder_path=/data1/scratch/zhisheng/models/SpatialAST/SpatialAST.pth # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialAST/finetuned.pth
llm_path=/home/zhisheng/models/llama-2-hf # https://huggingface.co/meta-llama/Llama-2-7b-hf

stage=stage2-single
qa_data_root=/data3/scratch/zhisheng/SpatialAudio/SpatialSoundQA/closed-end # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/closed-end
reverb_data_root=/data3/scratch/zhisheng/SpatialAudio/SpatialSoundQA/mp3d_reverb # https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/mp3d_reverb.zip
anechoic_data_root=/data3/scratch/zhisheng/SpatialAudio/SpatialSoundQA/AudioSet # https://huggingface.co/datasets/zhisheng01/SpatialAudio/tree/main/SpatialSoundQA/AudioSet

split=eval-stage2-classification
output_dir=?? # be same as in finetune script
ckpt_path=$output_dir/bat_epoch_4_step_18223
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
        ++model_config.qformer_layers=8 \
        ++model_config.encoder_ckpt=$audio_encoder_path \
        ++dataset_config.test_split=${split} \
        ++dataset_config.stage=$stage \
        ++dataset_config.qa_data_root=$qa_data_root \
        ++dataset_config.anechoic_data_root=$anechoic_data_root \
        ++dataset_config.reverb_data_root=$reverb_data_root \
        ++dataset_config.fix_length_audio=64 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=BAT \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++train_config.use_peft=true \
        ++peft_config.peft_method=lora \
        ++log_config.log_file=$output_dir/test.log \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt