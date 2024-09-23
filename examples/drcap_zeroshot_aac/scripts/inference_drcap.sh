#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

run_dir=/data/xiquan.li/SLAM-LLM_new   #dir to the SLAM project
cd $run_dir
code_dir=examples/drcap_zeroshot_aac

audio_encoder_dir=/data/xiquan.li/models/clap    #dir to the clap encoder 
output_dir=/data/xiquan.li/models/drcap/audiocaps    #dir to the pretrained linear mapping network
llm_path=/data/xiquan.li/models/vicuna-7b-v1.5   #path to llm

audio_encoder_path=$audio_encoder_dir/best_model.pt
pd_text_support=$audio_encoder_dir/support_embeddings/audiocaps_text_support.pt


encoder_projector_ds_rate=1
num_beams=4

inference_data_path=examples/drcap_zeroshot_aac/data/audiocaps_test.jsonl
decode_log=$output_dir/decode_log_test_clean_beam${num_beams}_repetition_penalty1


# -m debugpy --listen 6666 --wait-for-client
python $code_dir/inference_drcap_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    hydra.run.dir=$output_dir \
    ++model_config.llm_name="vicuna-7b-v1.5" \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=4096 \
    ++model_config.encoder_name=clap \
    ++model_config.encoder_path=$audio_encoder_path \
    ++model_config.encoder_dim=1024 \
    ++model_config.encoder_projector=linear \
    ++model_config.normalize=true \
    ++model_config.pd_text_support=$pd_text_support \
    ++dataset_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
    ++dataset_config.dataset=audio_dataset \
    ++dataset_config.val_data_path=$inference_data_path \
    ++dataset_config.model_name=clap \
    ++dataset_config.inference_mode=true \
    ++dataset_config.normalize=true \
    ++dataset_config.input_type=raw \
    ++dataset_config.use_rag=true \
    ++dataset_config.rag_first=true \
    ++train_config.model_name=aac \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=4 \
    ++train_config.num_workers_dataloader=8 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=false \
    ++ckpt_path=$output_dir/model.pt \
    ++peft_ckpt=$output_dir \
    ++train_config.use_peft=true \

# note: to inference model trained the linear layer only, you could set '++train_config.use_peft=false' and 'train_config.freeze_llm=true'