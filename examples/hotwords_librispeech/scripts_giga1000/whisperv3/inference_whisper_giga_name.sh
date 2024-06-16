#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1


run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/hotwords_librispeech

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v3.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm
ckpt_path=$output_dir/asr/3840

decode_log=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_name

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch_whisper.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_ds_rate=2 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=cov1d-linear \
        ++model_config.encoder_projector_ds_rate=5 \
        ++dataset_config.dataset=whispername_dataset \
        ++dataset_config.file=src/slam_llm/datasets/whisper_dataset_giganame.py:get_speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.inference_mode=true \
        ++dataset_config.input_type=mel \
        ++train_config.model_name=asr \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=1 \
        ++train_config.output_dir=$output_dir \
        ++ckpt_path=$ckpt_path/model.pt \
        ++decode_log=$decode_log \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \

python src/slam_llm/utils/whisper_tn.py ${decode_log}_gt ${decode_log}_gt.proc
python src/slam_llm/utils/whisper_tn.py ${decode_log}_pred ${decode_log}_pred.proc
python src/slam_llm/utils/compute_wer.py ${decode_log}_gt.proc ${decode_log}_pred.proc ${decode_log}.proc.wer
