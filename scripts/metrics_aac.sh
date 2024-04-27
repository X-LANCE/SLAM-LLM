#!/bin/bash

PYTHON_SCRIPT="/root/SLAM-LLM_old/src/llama_recipes/utils/eval_metrics.py"
EXP_FOLDER="/root/exps/test/finetune_test/aac_epoch_2_step_10682"

# PRED_FILE="${EXP_FOLDER}/decode_nucleus_pred"
PRED_FILE="${EXP_FOLDER}/decode_log_test_clean_beam_repetition_penalty1_pred"
# PRED_FILE="${EXP_FOLDER}/decode_keyword_test_pred"

GT_FILE=/root/data/AudioCaps/gt_fold/gt_all
# GT_FILE=/root/data/Clotho_v2/gt_all

python $PYTHON_SCRIPT \
    --pred_file $PRED_FILE \
    --gt_file $GT_FILE | tee "${EXP_FOLDER}/metrics.log"