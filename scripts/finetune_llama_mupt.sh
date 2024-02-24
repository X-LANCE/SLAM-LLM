#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

cd /root/SLAM-LLM

# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/tiny.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/base.pt
# speech_encoder_path=//nfs/maziyang.mzy/models/Whisper/small.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/medium.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Base.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Base+.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/wavlm/WavLM-Large.pt
text_encoder_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-Chat-v0.4

# llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-intermediate-step-1431k-3T
# llm_path=/nfs/maziyang.mzy/models/TinyLlama-1.1B-Chat-v0.4
# llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
# llm_path=/nfs/maziyang.mzy/models/Llama-2-7b-chat-hf
# llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5
llm_path=/nfs/maziyang.mzy/models/MuPT_v1_8192

output_dir=/nfs/maziyang.mzy/exps/Llama-2-7b-chat-hf-finetune-symbol-ds1-proj2048-steplrwarmup1e-4decay-lora-20240224-test
# ckpt_path=/nfs/maziyang.mzy/exps/vicuna-7b-v1.5-finetune-asr-ds5-proj2048-steplrwarmup1e-4keep-whisper-largev2-promptshort-lowergt-20240131/asr/4

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="MuPT_v1_8192" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=1536 \
++model_config.encoder_name="TinyLlama-1.1B-Chat-v0.4" \
++model_config.encoder_path=$text_encoder_path \
++model_config.encoder_dim=2048 \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=1 \
++dataset_config.dataset=text_dataset \
++dataset_config.file="src/llama_recipes/datasets/text_dataset.py:get_text_dataset" \
++dataset_config.tokenizer_path=$text_encoder_path \
++dataset_config.train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
++dataset_config.val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl \
++dataset_config.input_type=features \
++train_config.model_name=mupt \
++train_config.freeze_encoder=true \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=1 \
++train_config.val_batch_size=1 \
++train_config.num_workers_dataloader=1 \
++train_config.output_dir=$output_dir \
++metric=acc \
# ++train_config.freeze_llm=true \
# ++ckpt_path=$ckpt_path/model.pt \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
# ++peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-lora-prompt/asr/5" \


else
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
--master_port=29501 \
src/llama_recipes/pipeline/finetune.py \
--config-path "/root/SLAM-LLM/scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="MuPT_v1_8192" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=1536 \
++model_config.encoder_name="TinyLlama-1.1B-Chat-v0.4" \
++model_config.encoder_path=$text_encoder_path \
++model_config.encoder_dim=2048 \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=1 \
++dataset_config.dataset=text_dataset \
++dataset_config.file="src/llama_recipes/datasets/text_dataset.py:get_text_dataset" \
++dataset_config.tokenizer_path=$text_encoder_path \
++dataset_config.train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
++dataset_config.val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl \
++dataset_config.input_type=features \
++train_config.model_name=mupt \
++train_config.freeze_encoder=true \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=10000 \
++train_config.total_steps=1000000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=5000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++train_config.enable_fsdp=false \
++train_config.enable_ddp=true \
++train_config.use_fp16=true \
++metric=acc \
++log_config.log_file=/$output_dir/train.log \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir \
++log_config.wandb_entity_name=zym22 \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=${0##*/%.*} \
++log_config.log_interval=5 \
# ++train_config.freeze_llm=true \
# ++ckpt_path=$ckpt_path/model.pt \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
# ++peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
fi

# {"key": "1001-134707-0000_ASR", "prompt": "<ASR>", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only.", "target_len": 157, "source_len": 1581, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}
# {"key": "1688-142285-0005", "prompt": "<ASR>", "source": "/nfs/beinian.lzr/workspace/datasets/data/16k/opendata/librispeech/test_other/wav/1688-142285-0005.wav", "target": "YOU WHO WERE ALWAYS ACCUSING PEOPLE OF BEING SHOPPY AT HELSTONE", "target_len": 11, "source_len": 220, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}