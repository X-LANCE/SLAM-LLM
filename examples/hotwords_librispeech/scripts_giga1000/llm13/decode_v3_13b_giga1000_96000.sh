#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/hotwords_librispeech

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v3.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_gigaspeech/ft_v3_13b_giga1000-20240616/
ckpt_path=$output_dir/asr_epoch_1_step_96000

for ref_split in dev; do
        split=gigaspeech_${ref_split}
        val_data_path=/nfs/maziyang.mzy/data/gigaspeech/${split}.jsonl
        decode_log=$ckpt_path/decode_${split}_beam4
        python $code_dir/inference_asr_batch.py \
                --config-path "conf" \
                --config-name "prompt.yaml" \
                hydra.run.dir=$ckpt_path \
                ++model_config.llm_name="vicuna-13b-v1.5" \
                ++model_config.llm_path=$llm_path \
                ++model_config.llm_dim=5120 \
                ++model_config.encoder_name=whisper \
                ++model_config.encoder_ds_rate=2 \
                ++model_config.normalize=true \
                ++dataset_config.normalize=true \
                ++model_config.encoder_projector_ds_rate=5 \
                ++model_config.encoder_path=$speech_encoder_path \
                ++model_config.encoder_dim=1280 \
                ++model_config.encoder_projector=cov1d-linear \
                ++dataset_config.dataset=speech_fix_dataset \
                ++dataset_config.file=src/slam_llm/datasets/speech_fix_dataset.py:get_speech_dataset \
                ++dataset_config.val_data_path=$val_data_path \
                ++dataset_config.input_type=mel \
                ++dataset_config.mel_size=128 \
                ++dataset_config.inference_mode=true \
                ++train_config.model_name=asr \
                ++train_config.freeze_encoder=true \
                ++train_config.freeze_llm=true \
                ++train_config.batching_strategy=custom \
                ++train_config.num_epochs=1 \
                ++train_config.val_batch_size=1 \
                ++train_config.num_workers_dataloader=1 \
                ++train_config.output_dir=$output_dir \
                ++decode_log=$decode_log \
                ++ckpt_path=$ckpt_path/model.pt && \

        trans=${decode_log}_gt
        preds=${decode_log}_pred
        python src/slam_llm/utils/giga_tn.py ${trans} ${trans}.proc1
        python src/slam_llm/utils/giga_tn.py ${preds} ${preds}.proc1
        python src/llama_recipes/utils/whisper_tn.py ${trans}.proc1 ${trans}.proc2
        python src/llama_recipes/utils/llm_tn.py ${preds}.proc1 ${preds}.proc2
        python src/llama_recipes/utils/compute_wer.py ${trans}.proc2 ${preds}.proc2 ${preds}.gigaproc.wer

        tail -3 ${preds}.gigaproc.wer

done