#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/aac_audiocaps

audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M.pt
llm_path=/root/models/vicuna-7b-v1.5

seed=42
encoder_projector_ds_rate=5
num_beams=4

inference_data_path=/root/data/AudioCaps/new_test.jsonl
output_dir=/root/exps/test/finetune_test_lora/aac_epoch_2_step_6682
decode_log=$output_dir/decode_log_test_clean_beam${num_beams}_repetition_penalty1


# -m debugpy --listen 6666 --wait-for-client
python $code_dir/inference_aac_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    hydra.run.dir=$output_dir \
    ++model_config.llm_name="vicuna-7b-v1.5" \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=4096 \
    ++model_config.encoder_name=eat \
    ++model_config.encoder_path=$audio_encoder_path \
    ++model_config.encoder_dim=768 \
    ++model_config.encoder_projector=linear \
    ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
    ++model_config.normalize=true \
    ++dataset_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
    ++dataset_config.dataset=audio_dataset \
    ++dataset_config.val_data_path=$inference_data_path \
    ++dataset_config.fbank_mean=-4.268 \
    ++dataset_config.fbank_std=4.569 \
    ++dataset_config.model_name=eat \
    ++dataset_config.inference_mode=true \
    ++dataset_config.normalize=true \
    ++dataset_config.input_type=mel \
    ++train_config.model_name=aac \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=8 \
    ++train_config.num_workers_dataloader=8 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=false \
    ++ckpt_path=$output_dir/model.pt \
    ++dataset_config.fixed_length=true \
    ++dataset_config.target_length=1024 \
    ++peft_ckpt=$output_dir \
    ++train_config.use_peft=true \

# note: to inference model trained the linear layer only, you could set '++train_config.use_peft=false' and 'train_config.freeze_llm=true'
