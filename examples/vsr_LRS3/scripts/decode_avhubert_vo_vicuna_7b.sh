#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/SLAM-LLM
cd $run_dir
code_dir=examples/vsr_LRS3

speech_encoder_path=/nfs/yangguanrou.ygr/codes/av_hubert/self_large_vox_433h_new.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5

output_dir=/nfs/yangguanrou.ygr/experiments_avhubert/vicuna-7b-v1.5-large_vox_433h-tri-dataset-tiaocan_again
ckpt_path=$output_dir/asr/850

decode_log=$ckpt_path/decode_${split}_beam4

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_vsr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        +model_config.llm_name="vicuna-7b-v1.5" \
        +model_config.llm_path=$llm_path \
        +model_config.llm_dim=4096 \
        +model_config.encoder_name=av_hubert \
        +model_config.encoder_projector_ds_rate=5 \
        +model_config.encoder_path=$speech_encoder_path \
        +model_config.encoder_dim=1024 \
        +model_config.encoder_projector=cov1d-linear \
        +dataset_config.dataset=avhubert_dataset \
        +dataset_config.inference_mode=true \
        +dataset_config.test_split=test \
        +train_config.model_name=vsr \
        +train_config.freeze_encoder=true \
        +train_config.freeze_llm=true \
        +train_config.batching_strategy=custom \
        +train_config.num_epochs=1 \
        +train_config.val_batch_size=8 \
        +train_config.num_workers_dataloader=0 \
        +train_config.output_dir=$output_dir \
        +decode_log=$decode_log \
        +ckpt_path=$ckpt_path/model.pt \
        # +peft_ckpt=$ckpt_path \
        # +train_config.use_peft=true \
        # +train_config.peft_config.r=32 \
        # +dataset_config.normalize=true \
        # +model_config.encoder_projector=q-former \
        # +dataset_config.fix_length_audio=64 \

python src/slam_llm/utils/whisper_tn.py ${decode_log}_gt ${decode_log}_gt.proc
python src/slam_llm/utils/whisper_tn.py ${decode_log}_pred ${decode_log}_pred.proc
python src/slam_llm/utils/compute_wer.py ${decode_log}_gt.proc ${decode_log}_pred.proc ${decode_log}.proc.wer
