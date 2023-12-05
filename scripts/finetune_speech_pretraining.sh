#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

cd /root/SLAM-LLM

#speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/base.pt
speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
output_dir=/nfs/zhifu.gzf/models/llama-2-hf-finetune/speech_pretraining_qwen-audio-exp1

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python  src/llama_recipes/pipeline/finetune.py \
--model_name echat \
--freeze_encoder \
--use_fp16 \
--use_peft --peft_method lora \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset custom_dataset \
--custom_dataset.file src/llama_recipes/datasets/speech_dataset.py:get_audio_dataset \
--custom_dataset.data_path /nfs/beinian.lzr/workspace/datasets/speech_llm/train_dataset/data_wav_json/asr/librispeech_train_960h_wav_speech_llm_train_data.json \
--batching_strategy custom \
--custom_dataset.max_words 1024 \
--num_epochs 100 \
--batch_size_training 2 \
--output_dir $output_dir \
--num_workers_dataloader 4 \
--run_test_during_validation true \
--run_test_during_validation_file /nfs/zhifu.gzf/data/IEMOCAP_full_release/Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_M013.wav \
--metric acc \
# --ckpt_path "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7/model.pt" \
# --peft_ckpt "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7" \

else
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
src/llama_recipes/pipeline/finetune.py \
--model_name echat \
--freeze_encoder \
--use_fp16 \
--enable_fsdp --low_cpu_fsdp \
--use_peft --peft_method lora \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset custom_dataset \
--custom_dataset.file src/llama_recipes/datasets/speech_dataset.py:get_audio_dataset \
--custom_dataset.data_path /nfs/beinian.lzr/workspace/datasets/speech_llm/train_dataset/data_wav_json/asr/librispeech_train_960h_wav_speech_llm_train_data.json \
--batching_strategy custom \
--custom_dataset.max_words 1024 \
--num_epochs 100 \
--batch_size_training 8 \
--num_workers_dataloader 4 \
--val_batch_size 8 \
--output_dir $output_dir \
--run_test_during_validation \
--run_test_during_validation_file /nfs/zhifu.gzf/data/IEMOCAP_full_release/Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_M013.wav \
--metric acc \
# --ckpt_path "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7/model.pt" \
# --peft_ckpt "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7" \
fi