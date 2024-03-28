#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/work/SLAM-LLM
cd $run_dir
code_dir=examples/asr/

speech_encoder_path=/cxgroup/model/whisper/large-v3.pt

# llm_path=/cxgroup/model/Llama-2-7b-chat-hf
llm_path=/cxgroup/model/vicuna-7b-v1.5

output_dir=/work/exps/vicuna-7b-v1.5-mls-french-linear-lora-32-steplrwarmupkeep1e-4-whisper-largev3-20240313-test
ckpt_path=$output_dir/asr/1
val_data_path=data/mls/french_test.jsonl
decode_log=$ckpt_path/long_prompt_french_test_beam4

# -m debugpy --listen 5678 --wait-for-client
# python $code_dir/inference_asr_batch.py \

# prompt="Transcrire la parole en français. Sortez la transcription directement sans contenu redondant. Assurez-vous que la sortie n'est pas dupliquée. "
prompt="Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. Ensure that the spell and the grammar are correct."
# python -m debugpy --listen 5678 --wait-for-client $code_dir/inference_asr_batch.py \

mkdir -p ${code_dir}/override
FILE="decode_french.yaml"

cat > ${code_dir}/override/${FILE} <<- EOF
hydra:
    run:
        dir: $ckpt_path
model_config:
    llm_name: "vicuna-7b-v1.5"
    llm_path: $llm_path
    llm_dim: 4096
    encoder_name: whisper
    encoder_path: $speech_encoder_path
    encoder_dim: 1280
    encoder_projector: linear
dataset_config:
    prompt: $prompt
    dataset: speech_dataset
    val_data_path: $val_data_path
    input_type: mel
    mel_size: 128
    inference_mode: true
train_config:
    model_name: asr
    batching_strategy: custom
    num_epochs: 1
    val_batch_size: 4
    num_workers_dataloader: 4
    output_dir: $output_dir
    freeze_encoder: true
    freeze_llm: true
decode_log: $decode_log
ckpt_path: $ckpt_path/model.pt
EOF


python $code_dir/inference_asr_batch.py \
        --config-path "override" \
        --config-name $FILE \

python src/slam_llm/utils/compute_wer.py ${decode_log}_gt ${decode_log}_pred ${decode_log}_wer
