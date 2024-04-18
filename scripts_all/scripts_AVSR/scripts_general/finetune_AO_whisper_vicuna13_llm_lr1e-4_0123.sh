#!/bin/bash
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt


llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
# llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-ao-whisper-1e-4-0123

torchrun \
--nnodes 1 \
--nproc_per_node 4 \
src/llama_recipes/pipeline/finetune.py \
--model_name avsr \
--freeze_encoder \
--use_peft --peft_method lora \
--use_fp16 \
--enable_fsdp \
--llm_name vicuna-7b-v1.5 \
--llm_path $llm_path \
--llm_dim 4096 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 60 \
--batch_size_training 8 \
--val_batch_size 2 \
--num_workers_dataloader 0 \
--lr 1e-4 \
--output_dir $output_dir \
--metric acc \
--log_file "/root/SLAM-LLM/log/vicuna-7b-v1.5-finetune-ao-whisper-1e-4-0123" \
--avsr_dataset.modal AO \
--model_config.modal AO \
--use_wandb \
--wandb_dir $output_dir \
--wandb_entity_name yanghaha \
--wandb_project_name slam-llm \
--wandb_exp_name vicuna-7b-v1.5-finetune-ao-whisper-1e-4-0123 \
--log_interval 5 \

# --peft_ckpt "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4" \
# --ckpt_path "/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-5-whisper-prompt-padding30-20231228/asr/4/model.pt" \
# --use_peft --peft_method lora \
# --master_port=29501 \

# --use_peft --peft_method lora \


# 没用 encoder_ds_rate

# 1.15

# 7b batch size 开到2 ok的

#  6 2 0 可以