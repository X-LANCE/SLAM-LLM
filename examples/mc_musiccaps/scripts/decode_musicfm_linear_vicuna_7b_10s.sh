#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=$PWD
cd $run_dir
code_dir=examples/mc_musiccaps


music_encoder_path=path/to/pretrained/musicfm/pretrained_msd.pt
music_encoder_stat_path=path/to/pretrained/musicfm/msd_stats.json
music_encoder_config_path=facebook/wav2vec2-conformer-rope-large-960h-ft

llm_path=lmsys/vicuna-7b-v1.5


output_dir=/root/cq7_haina/save/music-caption/musicfm_vicuna7b_mc_10s_20240513_15:07:28
ckpt_path=$output_dir/mir_epoch_3_step_900


split=LP-MusicCaps-MC.test.exist
val_data_path=/root/cq7_haina/data/LP-MusicCaps-MC/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_avg


python $code_dir/inference_mir_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=vicuna-7b-v1.5 \
        ++model_config.llm_dim=4096 \
        ++model_config.llm_path=$llm_path \
        ++model_config.encoder_name=musicfm \
        ++dataset_config.normalize=false \
        ++model_config.encoder_layer_idx=9 \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$music_encoder_path \
        ++model_config.encoder_stat_path=$music_encoder_stat_path \
        ++model_config.encoder_config_path=$music_encoder_config_path \
        ++model_config.encoder_dim=1024 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=mir_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.fixed_duration=10.0 \
        ++dataset_config.audio_label_freq=25 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=mir \
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

