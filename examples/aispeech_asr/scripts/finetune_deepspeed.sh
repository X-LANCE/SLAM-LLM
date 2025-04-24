#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
# export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=7200
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1



run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU
cd $run_dir
code_dir=examples/aispeech_asr

train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/test
dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/test
train_max_frame_length=500
eval_max_frame_length=500
multitask_prompt_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multiprompt.jsonl"
prompt_style="\{\}" # "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n" | "USER: {}\n ASSISTANT:"
projector=linear
encoder_name=whisper
llm_name=Qwen2.5-7B-Instruct
use_peft=false # For llm
use_fp16=true
freeze_encoder=true
pad_or_trim=true # For whisper

deepspeed_config=examples/aispeech_asr/conf/ds_config.json

if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/aispeech_asr/exp/slidespeech/20250414/whisper_linear_Qwen2.5-7B-Instruct_lorafalse_padtrue_normal_asr_speedfalse_specaugfalse-1515_slidespeech_text/mala_asr_epoch_2_step_7000
fi

# Choose Encoder
if [[ $encoder_name == "whisper" ]]
then
    speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/whisper/large-v3.pt
    mel_size=128 
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
elif [[ $llm_name == "Qwen2.5-1.5B-Instruct" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-1.5B-Instruct
    llm_dim=3584 
else
    exit 1
fi






output_dir=${code_dir}/exp/$(date +"%Y%m%d-%H%M")
hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=$encoder_name \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=$projector \
++dataset_config.prompt_style=$prompt_style \
++dataset_config.train_max_frame_length=$train_max_frame_length \
++dataset_config.eval_max_frame_length=$eval_max_frame_length \
++dataset_config.multitask_prompt_path=$multitask_prompt_path \
++dataset_config.input_type=$input_type \
++dataset_config.mel_size=$mel_size \
++dataset_config.pad_or_trim=$pad_or_trim \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++train_config.model_name=aispeech_asr \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=$freeze_encoder \
++train_config.freeze_llm=true \
++train_config.use_peft=$use_peft \
++train_config.batching_strategy=dynamic \
++train_config.validation_interval=1000 \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
++metric=acc \
"
if [[ $use_peft == "true" || $freeze_encoder == false ]];then
    hydra_args+="++ckpt_path=$ckpt_path/model.pt"
fi


deepspeed \
    --num_nodes 1 \
    --num_gpus 8 \
    $code_dir/finetune_deepspeed.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=$use_fp16 \
    ++deepspeed_config=$deepspeed_config \
    ${hydra_args}
