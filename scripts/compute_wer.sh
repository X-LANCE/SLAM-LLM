#cd /root/SLAM-LLM

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/decode_giga_texttop_g2p_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/decode_giga_texttop_g2p_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/text/decode_giga_speechtotext_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/text/decode_giga_speechtotext_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/text/decode_giga_texttotext_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/text/decode_giga_texttotext_pred


trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/remake/decode_giga_texttop_g2p_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/remake/decode_giga_texttop_g2p_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/remake/debug_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/remake/debug_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisperv3/decode_giga_log_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisperv3/decode_giga_log_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/decode_giga_speechtop_g2p_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/decode_giga_speechtop_g2p_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/decode_giga_speechtop_g2p_onlyctc_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/g2p/decode_giga_speechtop_g2p_onlyctc_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/text/decode_giga_speechtotext_gigadata_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/text/decode_giga_speechtotext_gigadata_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_onlyctc_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_onlyctc_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtotext_gigadata_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtotext_gigadata_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner/decode_log_Giga_noname_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner/decode_log_Giga_noname_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_addphn_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_addphn_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_addphn_bs1_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_speechtop_g2p_addphn_bs1_pred

trans=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_gt_gt
preds=/root/SLAM-LLM/scripts_all/scripts_ner_texttop/whisper_v3_encoder/decode_giga_gt_pred

trans=/nfs/yangguanrou.ygr/experiments_hubert/vicuna-7b-v1.5-hubert_xtralarge_ll60k_finetune_ls960/asr/1188/decode_librispeech_test_clean_beam4_filtered_gt
preds=/nfs/yangguanrou.ygr/experiments_hubert/vicuna-7b-v1.5-hubert_xtralarge_ll60k_finetune_ls960/asr/1188/decode_librispeech_test_clean_beam4_filtered_pred

python src/llama_recipes/utils/whisper_tn.py ${trans} ${trans}.proc
python src/llama_recipes/utils/llm_tn.py ${preds} ${preds}.proc
python src/llama_recipes/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

tail -3 ${preds}.proc.wer