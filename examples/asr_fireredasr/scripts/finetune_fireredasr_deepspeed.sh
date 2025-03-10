#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
run_dir=/aistor/aispeech/hpc_stor01/home/pengjing00sx/SLAM-LLM/
cd $run_dir
code_dir=examples/asr_fireredasr
# multitask 
# dataset=alimeeting
# multitask_asr
dataset=aishell-1
prompt_style=normal #instruct
deepspeed_config=/aistor/aispeech/hpc_stor01/home/pengjing00sx/SLAM-LLM/examples/asr_fireredasr/conf/ds_config.json

if [[ $dataset == aishell-1 || $dataset == aishell-2 || $dataset == librispeech || $dataset == alimeeting || $dataset == gigaspeech || $dataset == wenetspeech ]]
then
    # aishell1:asr hotword 
    # aisehll2：asr hotword mt
    # librispeech:asr prevtext mt
    # alimeeting: asr_far_bf asr_near
    # gigaspeech: asr
    # wenetspeech: asr
    dataset_task=asr
fi
projector=linear
encoder_name=conformer
llm_name=Qwen2-7B-Instruct
use_peft=true
use_fp16=true
freeze_encoder=false
pad_or_trim=true
encoder_projector_ds_rate=2
# enhance
# enhance
speed_perturb=false
spec_augmentation=false
add_noise=false
add_reverb=false

if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    ckpt_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/FireRedASR-LLM-L
fi

# Choose Encoder
if [[ $encoder_name == "whisper" ]]
then
    if [[ $encoder_finetune == true ]]
    then    
        speech_encoder_path=
        mel_size=80
    else
        speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/whisper/large-v3.pt
        mel_size=128 
    fi
    encoder_dim=1280
    input_type=mel 
    
elif [[ $encoder_name == "wavlm" ]]
then
    speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/wavlm/WavLM-Large.pt
    encoder_dim=1024
    input_type=raw
    mel_size=128
elif [[ $encoder_name == "conformer" ]]
then 
    speech_encoder_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/FireRedASR/pretrained_models/FireRedASR-LLM-L/asr_encoder.pth.tar
    encoder_dim=1280
    input_type=raw
    mel_size=128
else
    exit 1
fi

# Choose LLM
if [[ $llm_name == "vicuna-7b-v1.5" ]]
then
    llm_path=
    llm_dim=4096
elif [[ $llm_name == "Qwen2.5-7B-Instruct" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B-Instruct" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/FireRedASR/pretrained_models/Qwen2-7B-Instruct
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=
    llm_dim=3584 
elif [[ $llm_name == "Qwen2.5-7B" ]]
then
    llm_path=
    llm_dim=3584 
else
    exit 1
fi

# Choose Train/Dev/Test
if [[ $dataset == aishell-1 || $dataset == aishell-2 || $dataset == librispeech || $dataset == alimeeting || $dataset == gigaspeech || $dataset == wenetspeech ]]
then
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/dev/
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/test/
elif [[  $dataset == "librispeech" ]]
then
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/dev-other/
else
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/dev/
fi
file=examples/asr_fireredasr/model/slam_fireredasr.py:model_factory
inference_mode=False
output_dir=${code_dir}/exp/${dataset}/$(date +"%Y%m%d")/${encoder_name}_${projector}_${llm_name}_lora${use_peft}_pad${pad_or_trim}_${prompt_style}_${dataset_task}_speed${speed_perturb}_specaug${spec_augmentation}-$(date +"%H%M")
hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=$encoder_name \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate  \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=$projector \
++model_config.ckpt_path=$ckpt_path \
++model_config.normalize=true \
++model_config.file=$file \
++dataset_config.llm_name=$llm_name \
++dataset_config.prompt_style=$prompt_style \
++dataset_config.normalize=true \
++dataset_config.dataset=$dataset \
++dataset_config.input_type=$input_type \
++dataset_config.speed_perturb=$speed_perturb \
++dataset_config.spec_augmentation=$spec_augmentation \
++dataset_config.add_reverb=$add_reverb \
++dataset_config.noise_file_path=$noise_file_path \
++dataset_config.mel_size=$mel_size \
++dataset_config.pad_or_trim=$pad_or_trim \
++dataset_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++train_config.model_name=mala_asr \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=$freeze_encoder \
++train_config.use_peft=$use_peft \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=5e-5 \
++train_config.validation_interval=50000 \
++train_config.batch_size_training=12 \
++train_config.val_batch_size=12 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
++train_config.inference_mode=$inference_mode \
++metric=acc \
"
if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    hydra_args+="++ckpt_path=$ckpt_path"
fi
# hydra_args+="++ckpt_path=$ckpt_path/model.pt"

# -m debugpy --listen 5678 --wait-for-client
if [[ $ASCEND_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_fireredasr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 8 \
        --master_port=29505 \
        $code_dir/finetune_fireredasr_deepspeed.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=$use_fp16 \
        ++deepspeed_config=$deepspeed_config \
        ${hydra_args}
fi
