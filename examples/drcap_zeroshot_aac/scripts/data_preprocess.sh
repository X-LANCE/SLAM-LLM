## data proprocess for DRCap, including: 
# 1. Do the text-to-text retrieval for training/validation set
# 2. Do the audio-to-text retrieval for the test set
# 3. Create text embedding support for projection-based decoding 

export CUDA_VISIBLE_DEVICES=1

root_dir=/data/xiquan.li/SLAM-LLM_new/
cd $root_dir

# -m debugpy --listen 6666 --wait-for-client

python  examples/drcap_zeroshot_aac/data_preprocess.py \
    --input_file_train examples/drcap_zeroshot_aac/data/train.jsonl \
    --input_file_val examples/drcap_zeroshot_aac/data/val.jsonl \
    --input_file_test examples/drcap_zeroshot_aac/data/test.jsonl \
    --input_file_database examples/drcap_zeroshot_aac/data/database.jsonl \
    --clap_encoder_path /data/xiquan.li/models/clap/models/best_model.pt \
    --topn 3 \
    --sim_min 0.75 \
    --sim_max 0.85 \