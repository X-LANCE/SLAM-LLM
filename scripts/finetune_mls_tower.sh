#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/SLAM-LLM/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

code_dir=/SLAM-LLM
cd $code_dir

speech_encoder_path=/cxgroup/model/whisper/large-v3.pt
llm_path=/cxgroup/model/TowerInstruct-7B-v0.2
output_dir=/exps/tower-7b-finetune-asr-linear-lora-16-steplrwarmupkeep1e-4-whisper-largev3-pl-LID-longprompt-average-$(date +"%Y%m%d")-test

# {"key": "1001-134707-0000_ASR", "prompt": "<ASR>", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only.", "target_len": 157, "source_len": 1581, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}
# {"key": "1688-142285-0005", "prompt": "<ASR>", "source": "/nfs/beinian.lzr/workspace/datasets/data/16k/opendata/librispeech/test_other/wav/1688-142285-0005.wav", "target": "YOU WHO WERE ALWAYS ACCUSING PEOPLE OF BEING SHOPPY AT HELSTONE", "target_len": 11, "source_len": 220, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}

torchrun \
--nnodes 1 \
--nproc_per_node 2 \
--master_port=29450 \
src/slam_llm/pipeline/finetune.py \
--config-path "${code_dir}/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="tower-7b" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=/data/polish/train.jsonl \
++dataset_config.val_data_path=/data/polish/dev.jsonl \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128 \
++dataset_config.prompt="Transcribe speech to polish text. Output the transcription directly without redundant content. Ensure that the output is not duplicated." \
++train_config.use_peft=true \
++train_config.peft_config.r=16 \
++train_config.peft_config.lora_alpha=32 \
++train_config.model_name=asr \
++train_config.num_epochs=6 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
++train_config.enable_fsdp=false \
++train_config.enable_ddp=true \
++train_config.use_fp16=true \