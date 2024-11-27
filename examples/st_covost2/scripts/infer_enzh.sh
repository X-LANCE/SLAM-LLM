export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=2,3
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

echo "GPU number: $gpu_count"

current_script=$(readlink -f "$0")
current_dir=$(dirname "$current_script")
code_dir=$(realpath "$current_dir/../../../../")
cd ${code_dir}/SLAM-LLM




#supported translation languages are Chinese (zh), German (de), and Japanese (ja).
# Check if command-line arguments are provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <source_language>"
  exit 1
fi
source=$1
echo "Source language is $source"

ckpt_path=${code_dir}/SLAM-LLM/cotst/model.pt
if [ ! -f "$ckpt_path" ]; then
    echo "Download ckpt..."
    git clone https://huggingface.co/yxdu/cotst
fi

echo $ckpt_path


decode_log=${code_dir}/SLAM-LLM/examples/st_covost2/covost2_${source}.jsonl

echo "Decode log saved to: ${decode_log}"




torchrun \
    --nnodes 1 \
    --nproc_per_node ${gpu_count} \
    --master_port=29503 \
    ${code_dir}/SLAM-LLM/examples/st_covost2/inference_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++fsdp_config.pure_bf16=true \
    ++model_config.llm_name="Qwen2-7B" \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=3584 \
    ++model_config.query_len=80 \
    ++model_config.encoder_name=whisper \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_path_hf=$encoder_path_hf \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_projector=q-former \
    ++dataset_config.dataset=hf_dataset \
    ++dataset_config.file=examples/st_covost2/dataset/hf_dataset.py:get_speech_dataset \
    ++dataset_config.val_data_path=$val_data_path \
    ++dataset_config.input_type=mel \
    ++dataset_config.fix_length_audio=80 \
    ++dataset_config.mel_size=128 \
    ++dataset_config.inference_mode=true \
    ++dataset_config.source=$source \
    ++train_config.model_name=asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=16 \
    ++train_config.num_workers_dataloader=8 \
    ++log_config.decode_log=$decode_log \
    ++ckpt_path=$ckpt_path \
    $hydra_args
fi

python ${code_dir}/SLAM-LLM/examples/st_covost2/test_werbleu.py --file $decode_log 