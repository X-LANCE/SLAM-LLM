export CUDA_VISIBLE_DEVICES=0

mkdir -p /root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/giga_speechtop_g2p/
python /nfs/yangguanrou.ygr/data/slidespeech/compute_wer_details/compute_wer_details.py --v 1 \
--ref /root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/decode_giga_speechtop_g2p_gt.proc \
--ref_ocr /nfs/yangguanrou.ygr/data/ner/giga_name_test/2/giga_ner_hotwords_list.txt \
--ref2session /nfs/yangguanrou.ygr/data/ner/giga_name_test/2/utt2spk \
--rec_name base \
--rec_name hot \
--rec_file /nfs/yangguanrou.ygr/experiments_slides_wavlm/slides-finetune-wavlm_notext/asr/1760/decode_log_Giga_noname_pred.proc \
--rec_file /root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/remake/debug_pred.proc \
> /root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/giga_speechtop_g2p/texttop_g2p.log

#-m debugpy --listen 5678 --wait-for-client
