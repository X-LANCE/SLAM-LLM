export CUDA_VISIBLE_DEVICES=0

mkdir -p /nfs/yangguanrou.ygr/slidespeech/for_pr/slides-finetune-wavlm/asr/3840/test/
python /nfs/yangguanrou.ygr/slidespeech/compute_wer_details/compute_wer_details.py --v 1 \
--ref /nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm/asr/3840/paper/decode_log_test_clean_beam4_repetition_penalty1_gt.proc \
--ref_ocr /nfs/yangguanrou.ygr/slidespeech/test_oracle_v1/hot_related/ocr_1gram_top50_mmr070_hotwords_list \
--ref2session /nfs/yangguanrou.ygr/slidespeech/test_oracle_v1/utt2spk \
--rec_name base \
--rec_name hot \
--rec_file /nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm_notext/asr/1760/decode_log_test_clean_beam4_repetition_penalty1_pred.proc \
--rec_file /nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm/asr/3840/paper/decode_log_test_clean_beam4_repetition_penalty1_pred.proc \
> /nfs/yangguanrou.ygr/slidespeech/for_pr/slides-finetune-wavlm/asr/3840/test/metric.log

#-m debugpy --listen 5678 --wait-for-client
