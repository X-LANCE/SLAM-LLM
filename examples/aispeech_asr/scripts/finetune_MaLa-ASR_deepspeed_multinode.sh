#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=3600
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM/
cd $run_dir
code_dir=examples/aispeech_asr

dataset=aishell-2
prompt_style=normal #instruct
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
encoder_name=whisper
llm_name=Qwen2.5-7B-Instruct
use_peft=true
use_fp16=true
freeze_encoder=true
pad_or_trim=true
encoder_projector_ds_rate=5
speed_perturb=false
spec_augmentation=false
add_noise=false
add_reverb=false
# deepspeed_config=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/mala-asr/conf/ds_config_from_k2.json
deepspeed_config=./conf/ds_config.json
if [[ $encoder_name == "whisper" ]]
then
    encoder_finetune=false
fi
if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/aispeech_asr/exp/librispeech/20250322/whisper_linear_Qwen2.5-7B-Instruct_lorafalse_padtrue_normal_asr_speedfalse_specaugfalse-1121/mala_asr_epoch_2_step_25000_best
fi

# Choose Encoder
if [[ $encoder_name == "whisper" ]]
then
    if [[ $encoder_finetune == true ]]
    then    
        speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/whisper/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt
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
elif [[  $dataset == "librispeech" ]]
then
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/dev-other/
else
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/dev/
fi


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
++model_config.normalize=true \
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
++train_config.freeze_llm=true \
++train_config.use_peft=$use_peft \
++train_config.batching_strategy=custom \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++metric=acc \
"
if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    hydra_args+="++ckpt_path=$ckpt_path/model.pt"
fi
# hydra_args+="++ckpt_path=$ckpt_path/model.pt"


HOST_FILE="/tmp/"${JobID}                        #生成的hostfile的完整文件名，$JobID调度系统会自动生成
SSH_PORT=6666                                    #因调度系统强制普通用户身份起容器，需要将ssh端口指定为大于1024的值
 
gen_hostfile() {                                 #此函数负责生成hostfile, 已跟调度系统对接好，直接使用，不要修改
    echo "${VC_MASTER_HOSTS} slots=${GPU_PER_TASK}" > ${HOST_FILE}
    echo "${VC_WORKER_HOSTS}" | awk -F ',' -v gpu_num=$GPU_PER_TASK '{for (i=1; i<=NF; i++) print $i" slots="gpu_num}' >> ${HOST_FILE}
}
 
do_train() {
    cat $HOST_FILE                                     #训练主入口函数
    /usr/sbin/sshd -p ${SSH_PORT}                #在Rank0上后台启动sshd服务，不要修改
    deepspeed \
        --hostfile $HOST_FILE \
        --ssh_port $SSH_PORT \
        $code_dir/finetune_mala_asr_deepspeed.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++train_config.use_fp16=$use_fp16 \
        ++deepspeed_config=$deepspeed_config \
        ${hydra_args} \

        
}
 
if [ "${RANK}" = "0" ]; then                     #只在index为RANK0的POD上启动主训练脚本，其他节点由主节点通过ssh分发任务（$RANK由调度系统自动分配）
    gen_hostfile                                 #生成分布式训练需要的hostfile
    do_train                                     #启动训练
else
    /usr/sbin/sshd -D -p ${SSH_PORT}             #其他节点的task，仅前台运行sshd服务，不执行主训练脚本，不要修改
fi