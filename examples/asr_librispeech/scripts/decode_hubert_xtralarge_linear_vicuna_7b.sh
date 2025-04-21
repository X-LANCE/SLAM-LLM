#!/bin/bash
export PYTHONPATH=/home/v-yifyang/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

run_dir=/home/v-yifyang/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/home/v-yifyang/ckpt/hubert_xtralarge_ll60k_finetune_ls960.pt
llm_path=/home/v-yifyang/ckpt/vicuna-7b-v1.5

output_dir=/home/v-yifyang/vicuna-7b-v1.5-hubert_xtralarge_ll60k_finetune_ls960
ckpt_path=/home/v-yifyang/ckpt
split=librispeech_test-clean
val_data_path=/home/v-yifyang/${split}.jsonl
decode_log=${output_dir}/decode_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=hubert \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_type=finetune \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++dataset_config.prompt="Transcribe speech to text. " \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=0 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
