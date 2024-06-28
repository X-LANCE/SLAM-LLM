export CUDA_VISIBLE_DEVICES=0

python /nfs/yangguanrou.ygr/slidespeech/compute_wer_details/compute_wer_details.py --v 1 \
--ref /nfs/yangguanrou.ygr/experiments_gigaspeech_lora/ft_wavlm_giga1000_lora_r32-ft-char-hotwords-20240617/asr_epoch_2_step_19475/decode_test_name_beam4_filter_fw_single_gt.proc2 \
--ref_ocr /nfs/yangguanrou.ygr/data/ner/giga_name_test/2/giga_ner_hotwords_list.txt \
--ref2session /nfs/yangguanrou.ygr/data/ner/giga_name_test/2/utt2spk \
--rec_name base \
--rec_file /nfs/yangguanrou.ygr/experiments_gigaspeech_lora/ft_wavlm_giga1000_lora_r32-ft-char-hotwords-20240617/asr_epoch_2_step_19475/decode_test_name_beam4_filter_fw_single_pred.proc2 \
> /root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/lora_ft/BER/wavlm_ft_char_filter_fw_wordnum2_BER_11.76.log

#-m debugpy --listen 5678 --wait-for-client
