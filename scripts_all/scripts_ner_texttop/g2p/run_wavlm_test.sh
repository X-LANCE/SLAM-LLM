export CUDA_VISIBLE_DEVICES=0

mkdir -p /nfs/yangguanrou.ygr/slides-finetune-wavlm/asr/3840/test/
python /nfs/yangguanrou.ygr/slidespeech/compute_wer_details.py --v 1 \
--ref /nfs/yangguanrou.ygr/slides-finetune-wavlm/asr/3840/decode_log_test_clean_beam4_repetition_penalty1_gt.proc \
--ref_ocr /nfs/yangguanrou.ygr/data/ner/giga_name_test/2/giga_ner_hotwords_list.txt \
--ref2session /nfs/yangguanrou.ygr/slidespeech/test_oracle_v1/utt2spk \
--rec_name base \
--rec_name hot \
--rec_file /nfs/yangguanrou.ygr/slides-finetune-wavlm_notext/asr/1760/decode_log_test_clean_beam4_repetition_penalty1_pred.proc \
--rec_file /nfs/yangguanrou.ygr/slides-finetune-wavlm/asr/3840/decode_log_test_clean_beam4_repetition_penalty1_pred.proc \
> /nfs/yangguanrou.ygr/slides-finetune-wavlm/asr/3840/test/metric.log

#-m debugpy --listen 5678 --wait-for-client
