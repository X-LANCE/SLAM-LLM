export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

cd /root/SLAM-LLM


speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2-qwen.pt

# llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
llm_path=/nfs/maziyang.mzy/models/vicuna-13b-v1.5

output_dir=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-debug

# -m debugpy --listen 5678 --wait-for-client
python src/llama_recipes/pipeline/finetune.py \
--model_name avsr \
--freeze_encoder \
--use_peft --peft_method lora \
--llm_name vicuna-13b-v1.5 \
--llm_path $llm_path \
--llm_dim 5120 \
--encoder_name whisper \
--encoder_ds_rate 2 \
--encoder_path $speech_encoder_path \
--encoder_dim 1280 \
--encoder_projector q-former \
--dataset avsr_dataset \
--avsr_dataset.file src/llama_recipes/datasets/avsr_dataset.py:get_audio_dataset \
--batching_strategy custom \
--num_epochs 1 \
--batch_size_training 2 \
--val_batch_size 2 \
--num_workers_dataloader 2 \
--lr 1e-3 \
--output_dir $output_dir \
--metric acc \
--avsr_dataset.modal AO \
--model_config.modal AO \
--log_file "/root/SLAM-LLM/log/debug.log" \
