#cd /root/SLAM-LLM

trans="/work/exps/vicuna-7b-v1.5-mls-french-linear-lora-32-steplrwarmupkeep1e-4-whisper-largev3-20240313-test/asr/1/decode_french_prompt_french_test_beam4_gt"
# preds="/work/exps/vicuna-7b-v1.5-finetune-asr-linear-lora-32-steplrwarmupkeep1e-4-whisper-largev3-20240313-test/asr/4/decode_log_polish_test_beam4_repetition_penalty1_pred"
preds="/work/exps/vicuna-7b-v1.5-mls-french-linear-lora-32-steplrwarmupkeep1e-4-whisper-largev3-20240313-test/asr/1/decode_french_prompt_french_test_beam4_pred"

# python src/llama_recipes/utils/preprocess_text.py ${preds} ${preds}.proc
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc ${preds}.proc.wer

python src/slam_llm/utils/whisper_tn.py ${trans} ${trans}.proc
python src/slam_llm/utils/whisper_tn.py ${preds} ${preds}.proc
# python src/slam_llm/utils/llm_tn.py ${preds} ${preds}.proc
python src/slam_llm/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

tail -3 ${preds}.proc.wer

# echo "-------num2word------"
# python src/llama_recipes/utils/num2word.py ${preds}.proc ${preds}.proc.words
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc.words ${preds}.proc.wer.words

# tail -3 ${preds}.proc.wer.words
