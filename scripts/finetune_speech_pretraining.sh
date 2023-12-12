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

speech_encoder_path=/nfs/zhifu.gzf/ckpt/Whisper/large-v2.pt
# speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt
llm_path=/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf
output_dir=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
python -m debugpy --listen 5678 --wait-for-client src/llama_recipes/pipeline/finetune.py \
--model_name asr \
--freeze_encoder \
--freeze_llm \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset custom_dataset \
--custom_dataset.file src/llama_recipes/datasets/speech_dataset.py:get_audio_dataset \
--custom_dataset.train_data_path /nfs/beinian.lzr/workspace/datasets/speech_llm/train_dataset/data_wav_json/asr/librispeech_train_960h_wav_speech_llm_train_data.json \
--custom_dataset.val_data_path /nfs/beinian.lzr/workspace/datasets/data/16k/opendata/librispeech/dev_other/librispeech_dev_other.jsonl \
--batching_strategy custom \
--num_epochs 100 \
--batch_size_training 4 \
--val_batch_size 4 \
--lr 1e-5 \
--output_dir $output_dir \
--run_test_during_validation \
--run_test_during_validation_file "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0000.wav" \
--run_test_during_validation_prompt "<|ASR|>" \
--metric acc \
# --ckpt_path "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7/model.pt" \
# --peft_ckpt "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7" \
# --use_peft --peft_method lora \

else
torchrun \
--nnodes 1 \
--nproc_per_node 4 \
src/llama_recipes/pipeline/finetune.py \
--model_name asr \
--freeze_encoder \
--freeze_llm \
--use_fp16 \
--enable_fsdp \
--llm_name llama-2-7b-hf \
--llm_path $llm_path \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_projector linear \
--encoder_projector_ds_rate 5 \
--dataset custom_dataset \
--custom_dataset.file src/llama_recipes/datasets/speech_dataset.py:get_audio_dataset \
--custom_dataset.train_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_train_960h_wav_speech_llm_train_data.json \
--custom_dataset.val_data_path /nfs/maziyang.mzy/data/librispeech/librispeech_dev_other.jsonl \
--batching_strategy custom \
--num_epochs 100 \
--batch_size_training 8 \
--val_batch_size 8 \
--num_workers_dataloader 4 \
--lr 1e-5 \
--output_dir $output_dir \
--run_test_during_validation \
--run_test_during_validation_file "/nfs/beinian.lzr/workspace/datasets/data/16k/opendata/librispeech/test_other/wav/1688-142285-0000.wav" \
--run_test_during_validation_prompt "<|ASR|>" \
--metric acc \
# --ckpt_path "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7/model.pt" \
# --peft_ckpt "/nfs/maziyang.mzy/models/llama-2-hf-finetune/echat/7" \
# --use_peft --peft_method lora \
fi

# {"key": "1001-134707-0000_ASR", "prompt": "<ASR>", "source": "/cpfs01/shared/Group-speech/beinian.lzr/data/open_data/librispeech_audio/audio/se_librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only.", "target_len": 157, "source_len": 1581, "text-type": "Transcribe", "audio_language": "en", "text_language": "en", "task-type": "<ASR>"}