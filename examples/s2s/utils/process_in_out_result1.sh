# test_out_id_path=/nfs/ya``ngguanrou.ygr/data/secap_my/remove_quater_emotion_prompt_diversity/q_id/test_id_1_25
# test_in_id_path=/nfs/yangguanrou.ygr/data/secap_my/remove_quater_emotion_prompt_diversity/q_id/test_id_75_99
# # dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_random_init_lr5e-6_50k_remove_q/s2s_epoch_20_step_1376/tts_decode_test_rp_seed_greedy_secap_test
# dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_lr5e-6_50k_remove_q/s2s_epoch_20_step_1376/tts_decode_test_rp_seed_greedy_secap_test
# python /nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/process_in_out_result1.py --dir ${dir} --test_out_id_path ${test_out_id_path} --test_in_id_path ${test_in_id_path}
# bash scripts/compute_wer_zh.sh ${dir}/split/in
# bash scripts/compute_wer_zh.sh ${dir}/split/out 


test_out_id_path=/nfs/yangguanrou.ygr/data/secap_my/remove_quater_emotion_prompt_diversity/h_id/test_id_0_49
test_in_id_path=/nfs/yangguanrou.ygr/data/secap_my/remove_quater_emotion_prompt_diversity/h_id/test_id_50_99
dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_random_init_lr5e-6_50k_remove_h/s2s_epoch_31_step_600/tts_decode_test_rp_seed_greedy_secap_test
# dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_lr5e-6_50k_remove_h/s2s_epoch_31_step_600/tts_decode_test_rp_seed_greedy_secap_test
python /nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/process_in_out_result1.py --dir ${dir} --test_out_id_path ${test_out_id_path} --test_in_id_path ${test_in_id_path}
bash scripts/compute_wer_zh.sh ${dir}/split/in
bash scripts/compute_wer_zh.sh ${dir}/split/out 