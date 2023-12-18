#cd /root/SLAM-LLM

trans="/Users/zhifu/Downloads/decode_log_test_clean_gt.txt"
preds="/Users/zhifu/Downloads/decode_log_test_clean_pred.txt"

python src/llama_recipes/utils/preprocess_text.py ${preds} ${preds}.proc
python src/llama_recipes/utils/num2word.py ${preds}.proc ${preds}.proc.words

python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc ${preds}.proc.wer

tail -3 ${preds}.proc.wer

echo "-------num2word------"

python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc.words ${preds}.proc.wer.words

tail -3 ${preds}.proc.wer.words