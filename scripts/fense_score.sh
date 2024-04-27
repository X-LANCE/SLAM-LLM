#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

origin_path=/root/exps/clotho/wavcaps_btz16_pt3_clotho_ft_seed42_btz4_lr1e-5/aac/1
origin_pred=${origin_path}/decode_log_test_clean_beam4_repetition_penalty1_pred
csv_pred=${origin_path}/pred.csv


awk -F'\t' 'BEGIN {print "file_name,caption"}; {sub(/\t/, ".wav\",\""); print "\"" $1  $2 "\""}' "$origin_pred" > "$csv_pred"

python /root/SLAM-LLM/src/llama_recipes/utils/fence.py \
    --device cuda \
    --dataset clotho \
    --ref_dir /root/fense/test_data/clotho_eval.csv\
    --cands_dir ${csv_pred} | tee "${origin_path}/fense.log"