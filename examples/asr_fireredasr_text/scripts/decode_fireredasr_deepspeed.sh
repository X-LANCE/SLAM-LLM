#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export ASCEND_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
set -e 
run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM/
cd $run_dir
code_dir=examples/asr_fireredasr_text
dataset=alimeeting
ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM/examples/asr_fireredasr_text
# ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/asr_fireredasr_text/exp/alimeeting/20250320/conformer_linear_Qwen2-7B-Instruct_loratrue_padtrue_normal_asr_far_bf_speedfalse_specaugfalse-1513/mala_asr_epoch_2_step_4000
prompt_style=normal  # normal #instruct
if [[ $dataset == aishell-1 || $dataset == aishell-2 || $dataset == librispeech-clean || $dataset == librispeech-other || $dataset == alimeeting || $dataset == slidespeech ]]
then
    # aishell-1:asr hotword
    # aishell-2:asr hotword mt
    # librispeech:asr prevtext mt
    # alimeeting: asr_far_bf asr_near
    # slidespeech: asr domain
    dataset_task=asr_far_bf
fi
projector=linear
encoder_name=conformer
llm_name=Qwen2-7B-Instruct
use_peft=false
use_fp16=true
pad_or_trim=true
encoder_projector_ds_rate=2
deepspeed_config=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM/examples/asr_fireredasr_text/conf/ds_config.json
prompt_style=normal #instruct
# Choose Encoder， 这个还是一点用也没有
if [[ $encoder_name == "whisper" ]]
then
    if [[ $encoder_finetune == true ]]
    then    
        speech_encoder_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/whisper-Pt/whisper-large-v2-multi-hans-zh-epoch-3-avg-10.pt
        mel_size=80
    else
        speech_encoder_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/whisper-large-v3/large-v3.pt
        mel_size=128 
    fi
    encoder_dim=1280
    input_type=mel 
    
elif [[ $encoder_name == "wavlm" ]]
then
    speech_encoder_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/wavlm/WavLM-Large.pt
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

# Choose LLM, 这个一点用也没有
if [[ $llm_name == "vicuna-7b-v1.5" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/vicuna-7b-v1.5
    llm_dim=4096
elif [[ $llm_name == "Qwen2.5-7B-Instruct" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2.5-7B-Instruct
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2-7B
    llm_dim=3584 
elif [[ $llm_name == "Qwen2.5-7B" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2.5-7B
    llm_dim=3584 
elif [[ $llm_name == "Qwen2-7B-Instruct" ]]
then 
    llm_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/FireRedASR/pretrained_models/Qwen2-7B-Instruct
    llm_dim=3584
else
    exit 1
fi

if [[ $dataset == "aishell-1" || $dataset == "aishell-2" || $dataset == "alimeeting" || slidespeech ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${dataset_task}/test/
elif [[ $dataset == "librispeech-other" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/test-other/
elif [[ $dataset == "librispeech-clean" ]]
then
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/librispeech/${dataset_task}/test-clean/
else
    test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/test/
fi
decode_log=$ckpt_path/decode_${dataset}_${dataset_task}_${prompt_style}
decode_log=./decode_${dataset}_${dataset_task}_${prompt_style}
# -m debugpy --listen 5678 --wait-for-client
deepspeed \
    --num_nodes 1 \
    --num_gpus 8 \
    $code_dir/inference_fireredasr_deepspeed.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    hydra.run.dir=$ckpt_path \
    ++model_config.llm_name=$llm_name \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.normalize=true \
    ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
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
    ++train_config.model_name=firered_asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=8 \
    ++train_config.num_workers_dataloader=8 \
    ++train_config.output_dir=$output_dir \
    ++train_config.inference_mode=true \
    ++decode_log=$decode_log \
    # ++ckpt_path=$ckpt_path/model.pt


python /aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/tools/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer 
python /aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/tools/pyResults/pyResults.py ${decode_log}_gt ${decode_log}_pred > ${decode_log}_ser 
python "/hpc_stor01/home/yangui.fang_sx/workingspace/SLAM-LLM/examples/mala_asr_slidespeech/slam_llm/utils/compute_wer.py"  ${decode_log}_gt ${decode_log}_pred ${decode_log}_ser
