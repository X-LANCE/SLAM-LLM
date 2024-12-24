#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/SLAM-LLM
cd $run_dir
code_dir=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/SLAM-LLM/examples/sec_emotioncaps

speech_encoder_path=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/ckpt/emotion2vec_base.pt
llm_path=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/ckpt/vicuna-7b-v1.5
val_data_path=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/data/valid.jsonl

encoder_fairseq_dir=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/deps/emotion2vec/upstream

output_dir=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/out/sec-decode-$(date +"%Y%m%d-%s")

ckpt_path=/hpc_stor03/sjtu_home/ruiyang.xu/SLAM/out/sec-finetune-20241001-1727786623/sec_epoch_1_step_3000/model.pt

decode_log=$output_dir/decode_log

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=emotion2vec \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_fairseq_dir=$encoder_fairseq_dir \
++model_config.encoder_dim=768  \
++model_config.encoder_projector=q-former \
++dataset_config.dataset=speech_dataset \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.data_path=$val_data_path \
++dataset_config.inference_mode=true \
++dataset_config.input_type=raw \
++train_config.model_name=sec \
++train_config.num_epochs=1 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++log_config.log_file=$output_dir/train.log \
++ckpt_path=$ckpt_path \
++decode_log=$decode_log
"

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_sec_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args