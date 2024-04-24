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

python src/llama_recipes/utils/whisper_tn.py ${trans} ${trans}.proc
python src/llama_recipes/utils/llm_tn.py ${preds} ${preds}.proc
python src/llama_recipes/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer

tail -3 ${preds}.proc.wer