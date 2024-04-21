#cd /root/SLAM-LLM

trans="/exps/mixtral-7b-finetune-asr-mls-linear-lora-24-projector-2048-ctc-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240413-test/asr/average/decode_log_test_beam4_gt"
preds="/exps/mixtral-7b-finetune-asr-mls-linear-lora-24-projector-2048-ctc-steplrwarmupkeep1e-4-whisper-largev3-fr-LID-longprompt-average-20240413-test/asr/average/decode_log_test_beam4_pred"

cd /SLAM-LLM
#python src/slam_llm/utils/whisper_tn.py ${trans} ${trans}.proc.whisper
#python src/slam_llm/utils/llm_tn.py ${preds} ${preds}.proc.whisper
python src/slam_llm/utils/preprocess_text_mls.py ${preds} ${preds}.proc fr
python src/slam_llm/utils/preprocess_text_mls.py ${trans} ${trans}.proc fr

python /SLAM-LLM/src/slam_llm/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

tail -3 ${preds}.proc.wer

# echo "-------num2word------"
# python src/llama_recipes/utils/num2word.py ${preds}.proc ${preds}.proc.words
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc.words ${preds}.proc.wer.words

# tail -3 ${preds}.proc.wer.words