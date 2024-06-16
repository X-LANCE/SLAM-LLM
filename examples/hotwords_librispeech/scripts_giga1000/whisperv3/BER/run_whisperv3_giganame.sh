export CUDA_VISIBLE_DEVICES=0

python /nfs/yangguanrou.ygr/slidespeech/compute_wer_details/compute_wer_details.py --v 1 \
--ref /root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_name_gt.proc2 \
--ref_ocr /nfs/yangguanrou.ygr/data/ner/giga_name_test/2/giga_ner_hotwords_list.txt \
--ref2session /nfs/yangguanrou.ygr/data/ner/giga_name_test/2/utt2spk \
--rec_name base \
--rec_file /root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_name_pred.proc2 \
> /root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/BER/BER.log

#-m debugpy --listen 5678 --wait-for-client
