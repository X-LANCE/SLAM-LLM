#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7


cd /root/SLAM-LLM

audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M.pt  # finetune 原本的定长版本 —— 需要修改 images.py
llm_path=/root/models/vicuna-7b-v1.5

seed=666
threshold=0.8
btz=16
lr=1e-4
pt_epoch=20
encoder_projector_ds_rate=5
short_prompt="Describe the audio you hear."
long_prompt="Describe the audio you hear. Output the audio caption directly without redundant content. Ensure that the output is not duplicated."
pre_tune_data_version=v4

pretrain_ckpt_path=/root/exps/wavcaps_pt-seed42_btz16_lr1e-4-random_crop/aac/${pt_epoch}

# exp_name=audiocaps_keyword_audio_prompt_seed${seed}_threshold${threshold}
# exp_name=wavcaps_btz16_epoch${pt_epoch}-audiocaps_ft-seed${seed}-btz${btz}-lr${lr}-short_prompt
exp_name=wavcap_pt_${pre_tune_data_version}-seed${seed}-btz${btz}-lr${lr}-short_prompt

# train_jsonl_path=/root/data/AudioCaps/train.jsonl
# train_jsonl_path=/root/data/AudioCaps/hot_word/updated_train.jsonl
# train_jsonl_path=/root/data/AudioCaps/hot_word/train/updated_train_${threshold}.jsonl
train_jsonl_path=/root/data/merged_data_${pre_tune_data_version}.jsonl

val_jsonl_path=/root/data/AudioCaps/val.jsonl
# val_jsonl_path=/root/data/AudioCaps/hot_word/updated_val.jsonl
# val_jsonl_path=/root/data/AudioCaps/hot_word/val/updated_val_${threshold}.jsonl

# category=audiocaps
# category=clotho
category=pre-tune
# category=test


output_dir=/root/exps/${category}/${exp_name}

# note: 下面用于 pretrain（关掉按照 eval 来存 ckpt）
# train_config.run_validation=false
# -m debugpy --listen 6666 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python /root/SLAM-LLM/src/llama_recipes/pipeline/finetune.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    hydra.run.dir=$output_dir \
    model_config.llm_name='vicuna-7b-v1.5' \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name='eat' \
    model_config.encoder_ds_rate=2 \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector='linear' \
    model_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
    dataset_config.encoder_projector_ds_rate=${encoder_projector_ds_rate} \
    +dataset_config.input_type=mel \
    dataset_config.dataset='audio_dataset' \
    dataset_config.train_data_path=${train_jsonl_path} \
    dataset_config.val_data_path=${val_jsonl_path} \
    dataset_config.prompt="${short_prompt}" \
    dataset_config.fbank_mean=-4.268 \
    dataset_config.fbank_std=4.569 \
    dataset_config.model_name=eat \
    train_config.model_name='aac' \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=false \
    train_config.batching_strategy='custom' \
    train_config.warmup_steps=1000 \
    train_config.total_steps=100000 \
    train_config.lr=$lr \
    train_config.validation_interval=500 \
    train_config.batch_size_training=$btz \
    train_config.val_batch_size=$btz \
    train_config.num_workers_dataloader=4 \
    train_config.use_fp16=true \
    train_config.output_dir=$output_dir \
    log_config.log_file="${output_dir}/train.log" \
    train_config.use_peft=true \
    train_config.peft_config.peft_method=lora \
    train_config.specaug=true \
    train_config.seed=${seed} \
    log_config.wandb_dir=${output_dir} \
    log_config.wandb_entity_name=wxc12 \
    log_config.wandb_project_name=slam-llm \
    log_config.wandb_exp_name=$exp_name \
    dataset_config.fixed_length=true \
    dataset_config.target_length=1024 \
    log_config.use_wandb=true \
    train_config.run_validation=false
    
    
    # dataset_config.use_keyword=true \
    # dataset_config.keyword_first=true \
    # train_config.keyword_first=true \

    # train_config.model_eval=true
    # train_config.run_validation=false

    # +ckpt_path="${pretrain_ckpt_path}/model.pt" \
    # +peft_ckpt="$pretrain_ckpt_path" \
    # train_config.use_neft=true \
    # ++metric=acc \
    # train_config.use_peft=true \
    # train_config.peft_config.peft_method=lora \


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
    src/llama_recipes/pipeline/finetune.py \
    --model_name aac \
    --freeze_encoder \
    --freeze_llm \
    --enable_fsdp \
    --llm_name llama-2-7b-hf \
    --llm_path $llm_path \
    --llm_dim 4096 \
    --encoder_name eat \
    --encoder_ds_rate 2 \
    --encoder_path $audio_encoder_path \
    --encoder_dim 768 \
    --encoder_projector linear \
    --encoder_projector_ds_rate 5 \
    --dataset audio_dataset \
    --audio_dataset.train_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
    --audio_dataset.val_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl \
    --batching_strategy custom \
    --num_epochs 100 \
    --batch_size_training 4 \
    --val_batch_size 4 \
    --num_workers_dataloader 4 \
    --lr 1e-4 \
    --output_dir $output_dir \
    --metric acc \
    --log_file /$output_dir/train.log \
    --use_wandb \
    --wandb_dir $output_dir \
    --wandb_entity_name wxc12 \
    --wandb_project_name slam-llm \
    --wandb_exp_name $exp_name \
    --log_interval 5 \
# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4/model.pt" \
# --use_peft --peft_method lora \
fi

# {"key": "1001-134707-0000_ASR", "prompt": "<ASR>", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only.", "target_len": 157, "source_len": 1581, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}
# {"key": "1688-142285-0005", "prompt": "<ASR>", "source": "/nfs/beinian.lzr/workspace/datasets/data/16k/opendata/librispeech/test_other/wav/1688-142285-0005.wav", "target": "YOU WHO WERE ALWAYS ACCUSING PEOPLE OF BEING SHOPPY AT HELSTONE", "target_len": 11, "source_len": 220, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}