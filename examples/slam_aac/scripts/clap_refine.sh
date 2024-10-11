export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

run_dir=/data/wenxi.chen/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

# -m debugpy --listen 6666 --wait-for-client
python ${code_dir}/utils/clap_refine.py \
    --start_beam 2 --end_beam 8 \
    --clap_ckpt /data/xiquan.li/models/clap/best_model.pt \
    --config /data/xiquan.li/models/clap/clap_config.yaml \
    --test_jsonl /data/xiquan.li/data/rz_cap/clotho/test_single.jsonl \
    --exp_explorer /data/wenxi.chen/models/clotho