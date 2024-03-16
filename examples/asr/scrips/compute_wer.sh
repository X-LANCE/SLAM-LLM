#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

code_dir=/work/SLAM-LLM
cd $code_dir

speech_encoder_path=/host/model_ckpt/whisper/large-v3.pt

output_dir=/work/exps/vicuna-7b-v1.5-finetune-asr-linear-lora-32-steplrwarmupkeep1e-4-whisper-largev3-20240308-test
ckpt_path=$output_dir/asr/4
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4
val_data_path=data/mls/polish_tem.jsonl
decode_log=$ckpt_path/decode_log_polish_test_beam4_repetition_penalty1

# python src/llama_recipes/utils/compute_wer.py ${decode_log}_gt ${decode_log}_pred ${decode_log}_wer
python src/llama_recipes/utils/compute_wer.py ${decode_log}_gt /work/whisper/mls_polish_test /work/whisper/mls_polish_test_wer
