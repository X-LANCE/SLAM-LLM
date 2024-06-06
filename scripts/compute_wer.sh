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

trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-1-20240523/asr_epoch_1_step_18000/decode_test_name_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-1-20240523/asr_epoch_1_step_18000/decode_test_name_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-20240522/asr_epoch_1_step_26000/decode_librispeech_test_clean_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-20240522/asr_epoch_1_step_26000/decode_librispeech_test_clean_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-1-20240523/asr_epoch_1_step_18000/decode_gigaspeech_dev_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-1-20240523/asr_epoch_1_step_18000/decode_gigaspeech_dev_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-lr5e-5-ws10000-mzy-20240525/asr_epoch_1_step_72000/decode_gigaspeech_dev_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-lr5e-5-ws10000-mzy-20240525/asr_epoch_1_step_72000/decode_gigaspeech_dev_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-lr5e-5-ws10000-mzy-20240525/asr_epoch_1_step_72000/decode_gigaspeech_test_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-lr5e-5-ws10000-mzy-20240525/asr_epoch_1_step_72000/decode_gigaspeech_test_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_gigaspeech_dev_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_gigaspeech_dev_beam4_pred


trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_phn-20240528/asr_epoch_1_step_84000/decode_gigaspeech_dev_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_phn-20240528/asr_epoch_1_step_84000/decode_gigaspeech_dev_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-remake-20240526/asr_epoch_1_step_86000/decode_gigaspeech_dev_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-remake-20240526/asr_epoch_1_step_86000/decode_gigaspeech_dev_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-remake-20240526/asr_epoch_1_step_86000/decode_gigaspeech_test_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/experiments_gigaspeech/tiaocan/vicuna-7b-v1.5-WavLM-Large-gigaspeech-remake-20240526/asr_epoch_1_step_86000/decode_gigaspeech_test_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_gigaspeech_test_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_gigaspeech_test_beam4_pred

trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_phn-20240528/asr_epoch_1_step_84000/decode_gigaspeech_test_beam4_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_phn-20240528/asr_epoch_1_step_84000/decode_gigaspeech_test_beam4_pred


trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_pred

python src/llama_recipes/utils/whisper_tn.py ${trans} ${trans}.proc
python src/llama_recipes/utils/llm_tn.py ${preds} ${preds}.proc
python src/llama_recipes/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

tail -3 ${preds}.proc.wer