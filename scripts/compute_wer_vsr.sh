#cd /root/SLAM-LLM


trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-vsr-20230116/avsr/20/decode_LRS3_test_beam4_repetition_penalty1_gt"
preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-vsr-20230116/avsr/20/decode_LRS3_test_beam4_repetition_penalty1_pred"


# python src/llama_recipes/utils/preprocess_text.py ${preds} ${preds}.proc
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc ${preds}.proc.wer
#-m debugpy --listen 5678 --wait-for-client 

# 我觉得还是不要前两个了 没什么用
# python src/llama_recipes/utils/whisper_tn.py ${trans} ${trans}.proc
# python src/llama_recipes/utils/llm_tn.py ${preds} ${preds}.proc
# python src/llama_recipes/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer
python src/llama_recipes/utils/compute_wer.py ${trans} ${preds} ${preds}.proc.wer

tail -3 ${preds}.proc.wer

# echo "-------num2word------"
# python src/llama_recipes/utils/num2word.py ${preds}.proc ${preds}.proc.words
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc.words ${preds}.proc.wer.words

# tail -3 ${preds}.proc.wer.words