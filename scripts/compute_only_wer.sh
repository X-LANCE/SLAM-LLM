trans=/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_gt_tn
preds=/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/my_tn/decode_dev_whisper_1_pred_tn


trans=/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_gt_tn
preds=/nfs/yangguanrou.ygr/slides-finetune-whisperv3/asr/9760/mytn/decode_log_dev_clean_beam4_repetition_penalty1_pred_tn


trans=/root/fairseq/data/bpe_txt/normal_test_other.txt
preds=/root/fairseq/wavlm_ft_libri960_base10h_token/decode_result/ckptbest/test_other/viterbi/wavlm_ft_libri960_test_other_tokentostr.txt

trans=/root/fairseq/data/bpe_txt/normal_test_clean.txt
preds=/root/fairseq/wavlm_ft_libri960_base10h_token/decode_result/ckptbest/test_clean/viterbi/wavlm_ft_libri960_test_clean_tokentostr.txt

trans=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_gt
preds=/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_gigaspeech_dev_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_gigaspeech_dev_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/dev_viterbi_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/dev_viterbi_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/test_viterbi_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/test_viterbi_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/vitterbi_giganame_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/vitterbi_giganame_pred

trans=/root/SALMONN/SALMONN_librispeech_test_clean.log_gt
preds=/root/SALMONN/SALMONN_librispeech_test_clean.log_pred

trans=/root/SALMONN/test_other/SALMONN_librispeech_test_other.log_gt
preds=/root/SALMONN/test_other/SALMONN_librispeech_test_other.log_pred

trans=/root/SALMONN/giga_dev/SALMONN_gigaspeech_dev.log_gt
preds=/root/SALMONN/giga_dev/SALMONN_gigaspeech_dev.log_pred


trans=/root/SALMONN/giga_test/SALMONN_gigaspeech_test.log_gt
preds=/root/SALMONN/giga_test/SALMONN_gigaspeech_test.log_pred

trans=/root/SLAM-LLM/mzy/dev_viterbi_gt
preds=/root/SLAM-LLM/mzy/dev_viterbi_pred

trans=/root/SLAM-LLM/mzy/only_gigatn/test_viterbi_gt
preds=/root/SLAM-LLM/mzy/only_gigatn/test_viterbi_pred

python src/llama_recipes/utils/compute_wer.py ${trans} ${preds} ${preds}.onlywer

tail -3 ${preds}.onlywer

