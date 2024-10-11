export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

run_dir=/data/wenxi.chen/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

clap_dir=/data/xiquan.li/models/clap
inference_data_path=/data/wenxi.chen/data/clotho/evaluation_single.jsonl
output_dir=/data/wenxi.chen/cp/wavcaps_pt_v7_epoch4-clotho_ft-seed10086_btz4_lr8e-6-short_prompt_10w/aac_epoch_1_step_4500

echo "Running CLAP-Refine"

# -m debugpy --listen 6666 --wait-for-client
python ${code_dir}/utils/clap_refine.py \
    --start_beam 2 --end_beam 8 \
    --clap_ckpt $clap_dir/best_model.pt \
    --config $clap_dir/clap_config.yaml \
    --test_jsonl $inference_data_path \
    --exp_explorer $output_dir