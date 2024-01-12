#cd /root/SLAM-LLM

trans="/nfs/maziyang.mzy/exps/llama-2-chat-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-prompt-padding30-20240111/asr/3/decode_log_test_other_beam4_repetition_penalty1_gt"
preds="/nfs/maziyang.mzy/exps/llama-2-chat-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-prompt-padding30-20240111/asr/3/decode_log_test_other_beam4_repetition_penalty1_pred"

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