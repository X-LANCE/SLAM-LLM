# trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/debug1/whisperv3_gigaspeech_dev_gt
# preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/debug1/whisperv3_gigaspeech_dev_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_gigaspeech_dev_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_gigaspeech_dev_pred

# trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_name_gt
# preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/whisperv3/whisperv3_name_pred

trans=/nfs/yangguanrou.ygr/experiments_gigaspeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_test_name_beam4_bs1_gt
preds=/nfs/yangguanrou.ygr/experiments_gigaspeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_test_name_beam4_bs1_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/dev_viterbi_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/dev_viterbi_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/test_viterbi_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/test_viterbi_pred

trans=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/vitterbi_giganame_gt
preds=/root/SLAM-LLM/examples/hotwords_librispeech/scripts_giga1000/wavlm_ft_viterbi/vitterbi_giganame_pred

trans=/nfs/yangguanrou.ygr/experiments_gigaspeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_test_name_beam4_bs1_again_gt
preds=/nfs/yangguanrou.ygr/experiments_gigaspeech/vicuna-7b-v1.5-WavLM-Large-gigaspeech-ft_char-20240527/asr_epoch_1_step_48000/decode_test_name_beam4_bs1_again_pred

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

trans=/root/SLAM-LLM/mzy/libri_4gram/dev_viterbi_gt
preds=/root/SLAM-LLM/mzy/libri_4gram/dev_viterbi_pred

trans=/root/SLAM-LLM/mzy/test_viterbi_gt
preds=/root/SLAM-LLM/mzy/test_viterbi_pred

python src/slam_llm/utils/giga_tn.py ${trans} ${trans}.proc1
python src/slam_llm/utils/giga_tn.py ${preds} ${preds}.proc1
python src/llama_recipes/utils/whisper_tn.py ${trans}.proc1 ${trans}.proc2
python src/llama_recipes/utils/llm_tn.py ${preds}.proc1 ${preds}.proc2
python src/llama_recipes/utils/compute_wer.py ${trans}.proc2 ${preds}.proc2 ${preds}.gigaproc.wer

# python src/slam_llm/utils/whisper_tn.py ${trans} ${trans}.proc
# python src/slam_llm/utils/whisper_tn.py ${preds} ${preds}.proc
# python src/slam_llm/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer


tail -3 ${preds}.gigaproc.wer