#!/bin/bash

cd /SLAM-LLM

python src/slam_llm/model_checkpointing/average_checkpoint.py \
    --input "/exps/mixtral-7b-finetune-asr-cvmls-linear-lora-24-projector-2048-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240404-test/asr/3,\
/exps/mixtral-7b-finetune-asr-cvmls-linear-lora-24-projector-2048-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240404-test/asr/4,\
/exps/mixtral-7b-finetune-asr-cvmls-linear-lora-24-projector-2048-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240404-test/asr/5" \
    --output "/exps/mixtral-7b-finetune-asr-cvmls-linear-lora-24-projector-2048-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240404-test/asr/average" \
    --peft

cp /exps/mixtral-7b-finetune-asr-cvmls-linear-lora-24-projector-2048-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240404-test/asr/3/adapter_config.json /exps/mixtral-7b-finetune-asr-cvmls-linear-lora-24-projector-2048-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240404-test/asr/average