export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

run_dir=/data/wenxi.chen/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

clap_dir=/data/xiquan.li/models/clap
inference_data_path=/data/wenxi.chen/data/audiocaps/new_test.jsonl
output_dir=/data/wenxi.chen/cp/aac_epoch_2_step_182_audiocaps_seed42

echo "Running CLAP-Refine"

# -m debugpy --listen 6666 --wait-for-client
python ${code_dir}/utils/clap_refine.py \
    --start_beam 2 --end_beam 8 \
    --clap_ckpt $clap_dir/best_model.pt \
    --config $clap_dir/clap_config.yaml \
    --test_jsonl $inference_data_path \
    --exp_explorer $output_dir