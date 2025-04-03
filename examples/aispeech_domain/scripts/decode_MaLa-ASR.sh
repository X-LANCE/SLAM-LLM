#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=1
# export ASCEND_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
# export OPENBLAS_NUM_THREADS=1
# export GOTO_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# export CUDA_LAUNCH_BLOCKING=1
set -e 
run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM/
cd $run_dir
code_dir=examples/aispeech_asr

dataset=aishell-1
prompt_style=instruct  # normal #instruct
if [[ $dataset == aishell-1 || $dataset == aishell-2 || $dataset == librispeech-clean || $dataset == librispeech-other || $dataset == alimeeting || $dataset == gigaspeech ]]
then
    # aishell-1:asr hotword
    # aishell-2:asr hotword mt
    # librispeech:asr prevtext mt
    # alimeeting: asr_far_bf asr_near asr_far_gss
    dataset_task=asr
fi
projector=linear
encoder_name=whisper
sentence=connect
llm_name=Qwen2.5-7B-Instruct
use_peft=true
use_fp16=true
pad_or_trim=true
encoder_projector_ds_rate=5
ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/aispeech_asr/exp/multitask_asr/20250307/whisper_linear_Qwen2.5-7B-Instruct_loratrue_padtrue_instruct__speedfalse_specaugfalse-1718/mala_asr_epoch_1_step_160000/

if [[ $encoder_name == "whisper" ]]
then
    encoder_finetune=false
fi

# Choose Encoder
if [[ $encoder_name == "whisper" ]]
then
    if [[ $encoder_finetune == true ]]
    then    
        speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/whisper-Pt/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt
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
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/vicuna-7b-v1.5
    llm_dim=4096
elif [[ $llm_name == "Qwen2.5-7B-Instruct" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2-7B
    llm_dim=3584 
elif [[ $llm_name == "Qwen2.5-7B" ]]
then
    llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B
    llm_dim=3584 
else
    exit 1
fi

if [[ $dataset == "aishell-1" || $dataset == "aishell-2" || $dataset == "alimeeting" || $dataset == gigaspeech ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/test/
elif [[  $dataset == "librispeech-other" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/test-other/
elif [[  $dataset == "librispeech-clean" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/test-clean/
elif [[  $dataset == "wenetspeech_test_net" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/wenetspeech/asr/test_net/
elif [[  $dataset == "wenetspeech_test_meeting" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/wenetspeech/asr/test_meeting/  
else
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/test/
fi

decode_log=$ckpt_path/decode_${dataset}_${dataset_task}_${prompt_style}
python $code_dir/inference_mala_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    hydra.run.dir=$ckpt_path \
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.normalize=true \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.encoder_projector=$projector \
    ++dataset_config.llm_name=$llm_name \
    ++dataset_config.prompt_style=$prompt_style \
    ++dataset_config.dataset=$dataset \
    ++dataset_config.pad_or_trim=$pad_or_trim \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.input_type=$input_type \
    ++dataset_config.mel_size=$mel_size \
    ++dataset_config.inference_mode=true \
    ++train_config.model_name=mala_asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=8 \
    ++train_config.num_workers_dataloader=0 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/model.pt  


python /aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/tools/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer 
python /aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/tools/pyResults/pyResults.py ${decode_log}_gt ${decode_log}_pred > ${decode_log}_ser 
python "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/SLAM-LLM/examples/mala_asr_slidespeech/slam_llm/utils/compute_wer.py"  ${decode_log}_gt ${decode_log}_pred ${decode_log}_ser
