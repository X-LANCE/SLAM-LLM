#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/mala_asr_slidespeech

speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Large.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm
ckpt_path=$output_dir/asr/3840
split=test #dev
val_data_path=/nfs/yangguanrou.ygr/slidespeech/${split}_oracle_v1/
decode_log=$ckpt_path/decode_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_mala_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=wavlm \
        ++model_config.normalize=true \
        ++dataset_config.normalize=true \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_projector=cov1d-linear \
        ++dataset_config.dataset=slidespeech_dataset \
        ++dataset_config.use_ocr=true \
        ++dataset_config.dev_scp_file_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=mala_asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \