#cd /root/SLAM-LLM

# trans="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/3/decode_log_test_other_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/3/decode_log_test_other_beam4_repetition_penalty1_pred"
# trans="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/20/decode_log_test_other_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/20/decode_log_test_other_beam4_repetition_penalty1_pred"
# trans="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/20/decode_log_test_other_beam4_repetition_penalty1_bs2_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/20/decode_log_test_other_beam4_repetition_penalty1_bs2_pred"
# trans="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/10/decode_log_test_other_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-13b-v1.5-finetune-avsr-20230115/avsr/10/decode_log_test_other_beam4_repetition_penalty1_pred"
# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/30/decode_LRS3_test_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/30/decode_LRS3_test_beam4_repetition_penalty1_pred"

# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/37/decode_LRS3_test_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/37/decode_LRS3_test_beam4_repetition_penalty1_pred"

# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/15/decode_LRS3_test_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/15/decode_LRS3_test_beam4_repetition_penalty1_pred"

# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/20/decode_LRS3_test_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-asr-20230116/avsr/20/decode_LRS3_test_beam4_repetition_penalty1_pred"

# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-sota-asr-20230119/avsr/40/decode_log_test_other_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-sota-asr-20230119/avsr/40/decode_log_test_other_beam4_repetition_penalty1_pred"

# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-sota-asr-1e-3-0121/avsr/43/decode_log_test_other_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-sota-asr-1e-3-0121/avsr/43/decode_log_test_other_beam4_repetition_penalty1_pred"

# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-sota-asr-1e-3-0121/avsr/60/decode_log_test_other_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-sota-asr-1e-3-0121/avsr/60/decode_log_test_other_beam4_repetition_penalty1_pred"


# trans="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-whisper-asr-20230121/avsr/40/decode_LRS3_test_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-finetune-whisper-asr-20230121/avsr/40/decode_LRS3_test_beam4_repetition_penalty1_pred"



# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-hubert_large_ll60k-0127/asr/3594/decode_log_test_clean_beam4_repetition_penalty1_pred"
# trans= "/nfs/yangguanrou.ygr/vicuna-7b-v1.5-hubert_large_ll60k-0127/asr/3594/decode_log_test_clean_beam4_repetition_penalty1_gt"

# trans= "/nfs/yangguanrou.ygr/vicuna-7b-v1.5-hubert_large_ll60k-0127/asr/3594/decode_log_test_clean_beam4_repetition_penalty1_gt"
# preds="/nfs/yangguanrou.ygr/vicuna-7b-v1.5-hubert_large_ll60k-0127/asr/3594/decode_log_test_clean_beam4_repetition_penalty1_pred"


# python src/llama_recipes/utils/preprocess_text.py ${preds} ${preds}.proc
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc ${preds}.proc.wer
#-m debugpy --listen 5678 --wait-for-client 

trans=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-large_vox_433h-qformer1/asr/384/decode_log_test_clean_beam4_repetition_penalty1_gt
presd=/nfs/yangguanrou.ygr/vicuna-7b-v1.5-large_vox_433h-qformer1/asr/384/decode_log_test_clean_beam4_repetition_penalty1_pred

trans=/root/tmp/vicuna-7b-v1.5-large_vox_433h_bs8-20240707/vsr_epoch_6_step_5695/decode__beam4_gt
preds=/root/tmp/vicuna-7b-v1.5-large_vox_433h_bs8-20240707/vsr_epoch_6_step_5695/decode__beam4_pred

trans=/root/tmp/vicuna-7b-v1.5-large_vox_433h_bs8-20240707/vsr_epoch_6_step_5695/decode__beam4_bs1_gt
preds=/root/tmp/vicuna-7b-v1.5-large_vox_433h_bs8-20240707/vsr_epoch_6_step_5695/decode__beam4_bs1_pred

trans=/root/tmp/vicuna-7b-v1.5-large_vox_433h_bs8_con-20240712/vsr_epoch_6_step_5695/decode__beam4_bs1_gt
preds=/root/tmp/vicuna-7b-v1.5-large_vox_433h_bs8_con-20240712/vsr_epoch_6_step_5695/decode__beam4_bs1_pred

trans=/root/tmp/vicuna-7b-v1.5-large_vox_433h_1e-4-20240712/vsr_epoch_6_step_5695/decode__beam4_bs1_gt
preds=/root/tmp/vicuna-7b-v1.5-large_vox_433h_1e-4-20240712/vsr_epoch_6_step_5695/decode__beam4_bs1_pred

# 我觉得还是不要前两个了 没什么用
python src/llama_recipes/utils/whisper_tn.py ${trans} ${trans}.proc
python src/llama_recipes/utils/llm_tn.py ${preds} ${preds}.proc
python src/llama_recipes/utils/compute_wer.py ${trans}.proc ${preds}.proc ${preds}.proc.wer
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds} ${preds}.proc.wer

tail -3 ${preds}.proc.wer

# echo "-------num2word------"
# python src/llama_recipes/utils/num2word.py ${preds}.proc ${preds}.proc.words
# python src/llama_recipes/utils/compute_wer.py ${trans} ${preds}.proc.words ${preds}.proc.wer.words

# tail -3 ${preds}.proc.wer.words