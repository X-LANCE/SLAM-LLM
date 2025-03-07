#!/bin/bash
# export PYTHONPATH=/root/whisper:$PYTHONPATH
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1,2
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1
# set -e 
# set -r 
# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/hpc_stor01/home/yangui.fang_sx/workingspace/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech




dataset=aishell1
projector=linear
encoder_name=whisper
llm_name=Qwen2.5-7B-Instruct
use_peft=false
ckpt_path=/hpc_stor01/home/yangui.fang_sx/workingspace/project/asr_librispeech_origin/exp/whisper_Qwen2.5-7B-Instruct_aishell1_linear_lorafalse_20241202-1342/asr_epoch_4_step_2482/model.pt
if [[ $peft == true ]]
then
    ckpt_path=/hpc_stor01/home/yangui.fang_sx/workingspace/project/asr_librispeech_origin/exp/whisper_Qwen2.5-1.5B-Instruct_aishell1_linear_lorafalse_20241203-1823/asr/5/model.pt
fi

# Choose Encoder
if [[ $encoder_name == "whisper" ]]
then
    speech_encoder_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/whisper-large-v3/large-v3.pt
    encoder_dim=1280
    input_type=mel 
    mel_size=128 
elif [[ $encoder_name == "wavlm" ]]
then
    speech_encoder_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/wavlm/WavLM-Large.pt
    encoder_dim=1024
    input_type=raw
    mel_size=
else
    exit 1
fi

# Choose LLM
if [[ $llm_name == "vicuna-7b-v1.5" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/vicuna-7b-v1.5
    llm_dim=4096
    use_fp16=true
elif [[ $llm_name == "Qwen2.5-7B-Instruct" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2.5-7B-Instruct
    llm_dim=3584 
    use_fp16=true
elif [[ $llm_name == "Qwen2.5-1.5B-Instruct" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2.5-1.5B-Instruct
    llm_dim=1536 
    use_fp16=true
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2-7B
    llm_dim=3584 
    use_fp16=true
elif [[ $llm_name == "Qwen2.5-7B" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2.5-7B
    llm_dim=3584 
    use_fp16=true
else
    exit 1
fi

# Choose Train/Dev/Test
if [[ $dataset == "aishell1" ]]
then
    train_data_path=/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr_librispeech/train.jsonl
    val_data_path=/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr_librispeech/dev.jsonl
else
    exit 1
fi

output_dir=$run_dir/$code_dir/exp/${encoder_name}_${llm_name}_${dataset}_${projector}_lora${use_peft}_$(date +"%Y%m%d-%H%M")

hydra_args="hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=$encoder_name \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=$projector \
++dataset_config.dataset=$dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=$input_type \
++dataset_config.mel_size=128 \
++train_config.model_name=asr \
++train_config.num_epochs=10 \
++train_config.use_peft=$use_peft \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=12000 \
++train_config.batch_size_training=6 \
++train_config.val_batch_size=6 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=acc \

"
# ++ckpt_path=$ckpt_path \

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=$use_fp16 \
        $hydra_args \

fi