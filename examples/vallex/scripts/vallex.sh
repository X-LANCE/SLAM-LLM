export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export FORCE_CPU=1

cd /home/wangtianrui/codes/SLAM-LLM-main


output_dir=/home/wangtianrui/model_save/vallex

PYTHONPATH=/home/wangtianrui/codes/SLAM-LLM-main \
python src/llama_recipes/pipeline/finetune.py \
--config-path "examples/vallex/conf" \
--config-name "vallex.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="vallex" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=1536 \
\
++dataset_config.dataset=vallex_dataset \
++dataset_config.file="src/llama_recipes/datasets/vallex_dataset.py:get_vallex_dataset" \
++dataset_config.train_data_path=/home/wangtianrui/datas/tiny_data/bilibli_7min_woman/data_bin_zh \
\
++train_config.batching_strategy=custom \
++train_config.use_fp16=true \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=1 \
++train_config.output_dir=$output_dir \
++metric=acc 