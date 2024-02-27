trans=/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_gt_tn
preds=/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_pred_tn


trans=/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_gt_tn
preds=/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_pred_tn

python src/llama_recipes/utils/compute_wer.py ${trans} ${preds} ${preds}.proc.wer

tail -3 ${preds}.proc.wer