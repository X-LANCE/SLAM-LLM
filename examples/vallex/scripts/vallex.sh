export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export FORCE_CPU=1
export RANK=0

cd /Work20/2023/wangtianrui/codes/util_repos/SLAM-LLM


output_dir=/Work20/2023/wangtianrui/codes/test/debug

PYTHONPATH=/Work20/2023/wangtianrui/codes/util_repos/SLAM-LLM \
python examples/vallex/finetune_vallex.py \
--config-path "/Work20/2023/wangtianrui/codes/util_repos/SLAM-LLM/examples/vallex/conf" \
--config-name "vallex.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="vallex" \
++dataset_config.file="src/slam_llm/datasets/vallex_dataset.py:get_vallex_dataset" \
++dataset_config.train_data_path=/Work20/2023/wangtianrui/datas/bilibli_7min_woman/data_bin_zh \
++dataset_config.val_data_path=/Work20/2023/wangtianrui/datas/bilibli_7min_woman/data_bin_zh \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=5e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=1 \
++train_config.batching_strategy="custom" \
++train_config.output_dir=$output_dir 