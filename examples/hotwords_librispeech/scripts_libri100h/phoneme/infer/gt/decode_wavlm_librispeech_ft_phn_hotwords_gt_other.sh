#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/hotwords_librispeech

speech_encoder_path=/nfs/yangguanrou.ygr/ckpts/wavlm_large_finetune_librispeech_phoneme/wavlm_large_finetune_librispeech_phoneme.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-ft-phn-hotwords-again-20240513
ckpt_path=$output_dir/asr_epoch_3_step_19780
split=librispeech_test_other
val_data_path=/nfs/maziyang.mzy/data/librispeech/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam4_gt

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
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
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++dataset_config.infer_type=gt \
        ++dataset_config.dataset=hotwordsinfer_dataset \
        ++dataset_config.file=src/slam_llm/datasets/hotwordsinfer_dataset.py:get_speech_dataset \
        ++dataset_config.infer_file=/nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/my_ref/test-other.biasing_100.tsv \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \

python src/slam_llm/utils/whisper_tn.py ${decode_log}_gt ${decode_log}_gt.proc
python src/slam_llm/utils/whisper_tn.py ${decode_log}_pred ${decode_log}_pred.proc
python src/slam_llm/utils/compute_wer.py ${decode_log}_gt.proc ${decode_log}_pred.proc ${decode_log}.proc.wer
