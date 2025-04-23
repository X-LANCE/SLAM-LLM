#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/wavlm/WavLM-Large.pt
llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/vicuna-7b-v1.5
train_data_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multitask_wav.jsonl
val_data_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multitask_wav.jsonl

output_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU/examples/asr_librispeech/exp-$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=vicuna-7b-v1.5 \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=wavlm \
++model_config.normalize=true \
++dataset_config.normalize=true \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1024 \
++model_config.encoder_projector=linear \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++train_config.model_name=asr \
++train_config.num_epochs=3 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=50 \
++train_config.batch_size_training=1 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \
"




deepspeed \
    --num_gpus=8 \
    --num_nodes=1 \
    $code_dir/deepspeed_finetune_asr.py \
    $hydra_args
    # --num_gpus=2 \
    # --num_nodes=1 \
    # --master_port=29502 \

