trans=/nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm/asr/3840/decode_test_beam4_shishilv_gt
preds=/nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm/asr/3840/decode_test_beam4_shishilv_pred

python src/slam_llm/utils/whisper_tn.py ${trans} ${trans}.proc
python src/slam_llm/utils/llm_tn.py ${preds} ${preds}.proc
python src/slam_llm/utils/compute_wer_old.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

# python src/slam_llm/utils/whisper_tn.py ${trans} ${trans}.proc
# python src/slam_llm/utils/whisper_tn.py ${preds} ${preds}.proc
# python src/slam_llm/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer


tail -3 ${preds}.proc.wer