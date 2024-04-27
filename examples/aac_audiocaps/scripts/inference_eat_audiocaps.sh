#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false

cd /root/SLAM-LLM

audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M.pt

llm_path=/root/models/vicuna-7b-v1.5

seed=42
num_beams=4
encoder_projector_ds_rate=5
prompt="Describe the audio you hear. Output the audio caption directly without redundant content. Ensure that the output is not duplicated."

output_dir=/root/exps/audiocaps/eat_finetune_linear_btz4
ckpt_path=$output_dir/aac/2

inference_data_path=/root/data/AudioCaps/test.jsonl

decode_log=$ckpt_path/decode_log_test_clean_beam${num_beams}_repetition_penalty1


# -m debugpy --listen 6666 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    hydra.run.dir=$ckpt_path \
    model_config.llm_name="vicuna-7b-v1.5" \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name=eat \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector=linear \
    model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
    +model_config.normalize=true \
    dataset_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
    dataset_config.dataset=audio_dataset \
    dataset_config.prompt="${prompt}" \
    dataset_config.val_data_path=$inference_data_path \
    dataset_config.fbank_mean=-4.268 \
    dataset_config.fbank_std=4.569 \
    dataset_config.model_name=eat \
    dataset_config.inference_mode=true \
    +dataset_config.normalize=true \
    +dataset_config.input_type=mel \
    train_config.model_name=aac \
    train_config.batching_strategy=custom \
    train_config.num_epochs=1 \
    train_config.val_batch_size=8 \
    train_config.num_workers_dataloader=8 \
    train_config.output_dir=$output_dir \
    +decode_log=$decode_log \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \
    +ckpt_path=$ckpt_path/model.pt \
    dataset_config.fixed_length=true \
    dataset_config.target_length=1024 \
    model_config.num_beams=$num_beams \