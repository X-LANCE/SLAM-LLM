#cd /root/SLAM-LLM

trans="/nfs/maziyang.mzy/exps/vicuna-7b-v1.5-finetune-asr-qformer64-steplrwarmupkeep1e-4-whisper-largev2-promptshort-lowergt-padding30-20240126/asr/3/decode_log_test_clean_beam4_repetition_penalty1_gt"
preds="/nfs/maziyang.mzy/exps/vicuna-7b-v1.5-finetune-asr-qformer64-steplrwarmupkeep1e-4-whisper-largev2-promptshort-lowergt-padding30-20240126/asr/3/decode_log_test_clean_beam4_repetition_penalty1_pred"

# python src/llama_recipes/utils/preprocess_text.py ${preds} ${preds}.proc
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc ${preds}.proc.wer

python src/llama_recipes/utils/whisper_tn.py ${trans} ${trans}.proc
python src/llama_recipes/utils/llm_tn.py ${preds} ${preds}.proc
python src/llama_recipes/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

tail -3 ${preds}.proc.wer

# echo "-------num2word------"
# python src/llama_recipes/utils/num2word.py ${preds}.proc ${preds}.proc.words
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc.words ${preds}.proc.wer.words

# tail -3 ${preds}.proc.wer.words