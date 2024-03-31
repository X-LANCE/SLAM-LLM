#!/bin/bash

PYTHON_SCRIPT="/root/SLAM-LLM/src/llama_recipes/utils/eval_metrics.py"
EXP_FOLDER="/root/exps/eat_lora_specaug/aac/3"
PRED_FILE="${EXP_FOLDER}/decode_log_test_clean_beam4_repetition_penalty1_pred"
GT_FILE="/root/data/AudioCaps/gt_fold/gt_all"

python $PYTHON_SCRIPT \
    --pred_file $PRED_FILE \
    --gt_file $GT_FILE | tee "${EXP_FOLDER}/metrics.log"