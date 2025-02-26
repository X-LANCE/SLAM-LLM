#dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/belle_pretrain_remake/s2s_epoch_2_step_31841/tts_decode_test_rp_seed_greedy_secap_test
#dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_lr5e-6_50k/s2s_epoch_15_step_354/tts_decode_test_rp_seed_greedy_secap_test
#dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_lr5e-6_50k/s2s_epoch_8_step_2677/tts_decode_test_rp_seed_greedy_secap_test
#dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_random_init_lr5e-6_50k/s2s_epoch_15_step_354/tts_decode_test_rp_seed_greedy_secap_test
dir=/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/ft_secap_qwen25_random_init_lr5e-6_50k/s2s_epoch_8_step_2677/tts_decode_test_rp_seed_greedy_secap_test
python /nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/process_in_out_result.py --dir ${dir}
bash scripts/compute_wer_zh.sh ${dir}/split/in
bash scripts/compute_wer_zh.sh ${dir}/split/out 