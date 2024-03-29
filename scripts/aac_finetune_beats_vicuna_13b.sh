#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=7

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

cd /root/SLAM-LLM

# speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
# audio_encoder_path=/root/models/BEATs_iter3_plus_AS2M.pt
audio_encoder_path=/root/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
# speech_encoder_path=/root/models/BEATs_iter3_plus_AS2M.pt

exp_name=vicuna_13b_beats_finetune_linear
llm_path=/root/models/vicuna-13b-v1.5/snapshots/model
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5/vicuna-13b-v1.5

output_dir=/root/exps/$exp_name

# -m debugpy --listen 6666 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python /root/SLAM-LLM/src/llama_recipes/pipeline/finetune.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    hydra.run.dir=$output_dir \
    model_config.llm_name='vicuna-13b-v1.5' \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=5120 \
    model_config.encoder_name='beats' \
    model_config.encoder_ds_rate=2 \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector='linear' \
    model_config.encoder_projector_ds_rate=5 \
    +dataset_config.input_type=mel \
    dataset_config.dataset='audio_dataset' \
    dataset_config.train_data_path='/root/data/AudioCaps/train.jsonl' \
    dataset_config.val_data_path='/root/data/AudioCaps/val.jsonl' \
    dataset_config.fbank_mean=15.41663 \
    dataset_config.fbank_std=6.55582 \
    dataset_config.model_name=beats \
    train_config.model_name='aac' \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \
    train_config.batching_strategy='custom' \
    train_config.warmup_steps=1000 \
    train_config.total_steps=100000 \
    train_config.lr=1e-4 \
    train_config.validation_interval=1000 \
    train_config.batch_size_training=2 \
    train_config.val_batch_size=4 \
    train_config.num_workers_dataloader=4 \
    train_config.output_dir=$output_dir \
    train_config.use_fp16=true \
    log_config.log_file="${output_dir}/train.log" \
    log_config.wandb_dir=${output_dir} \
    log_config.wandb_entity_name=wxc12 \
    log_config.wandb_project_name=slam-llm \
    log_config.wandb_exp_name=$exp_name \
    log_config.use_wandb=true \
    # train_config.use_peft=true \
    # train_config.peft_config.peft_method=lora \
    # ++metric=acc \
    # train_config.use_peft=true \
    # train_config.peft_config.peft_method=lora \
    # model_config.encoder_projector='q-former' \


    # --ckpt_path "/root/ckpt/peft/model.pt" \
    # --peft_ckpt "/root/ckpt/peft" \

# --log_interval 5 \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt/asr/5/model.pt" \
# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt/asr/5" \
# --use_peft --peft_method lora \

else
torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    /root/SLAM-LLM/src/llama_recipes/pipeline/finetune.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    hydra.run.dir=$output_dir \
    model_config.llm_name='vicuna-7b-v1.5' \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name='beats' \
    model_config.encoder_ds_rate=2 \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector='linear' \
    model_config.encoder_projector_ds_rate=5 \
    +dataset_config.input_type=mel \
    dataset_config.dataset='audio_dataset' \
    dataset_config.train_data_path='/root/data/AudioCaps/train.jsonl' \
    dataset_config.val_data_path='/root/data/AudioCaps/val.jsonl' \
    dataset_config.fbank_mean=15.41663 \
    dataset_config.fbank_std=6.55582 \
    dataset_config.model_name=beats \
    train_config.enable_ddp=true \
    train_config.model_name='aac' \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \
    train_config.batching_strategy='custom' \
    train_config.warmup_steps=1000 \
    train_config.total_steps=100000 \
    train_config.lr=1e-4 \
    train_config.validation_interval=1000 \
    train_config.batch_size_training=8 \
    train_config.val_batch_size=8 \
    train_config.num_workers_dataloader=4 \
    train_config.output_dir=$output_dir \
    train_config.use_fp16=true \
    log_config.log_file="${output_dir}/train.log" \

# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4/model.pt" \
# --use_peft --peft_method lora \
fi

# {"key": "1001-134707-0000_ASR", "prompt": "<ASR>", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only.", "target_len": 157, "source_len": 1581, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}
# {"key": "1688-142285-0005", "prompt": "<ASR>", "source": "/nfs/beinian.lzr/workspace/datasets/data/16k/opendata/librispeech/test_other/wav/1688-142285-0005.wav", "target": "YOU WHO WERE ALWAYS ACCUSING PEOPLE OF BEING SHOPPY AT HELSTONE", "target_len": 11, "source_len": 220, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}