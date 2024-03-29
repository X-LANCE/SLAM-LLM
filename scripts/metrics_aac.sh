#!/bin/bash

PYTHON_SCRIPT="/root/SLAM-LLM/src/llama_recipes/utils/eval_metrics.py"

PRED_FILE="/root/exps/beats_finetune_linear/aac/3/decode_log_test_clean_beam4_repetition_penalty1_pred"
GT_FILE="/root/data/AudioCaps/gt_fold/gt_all"

python $PYTHON_SCRIPT \
    --pred_file $PRED_FILE \
    --gt_file $GT_FILE
