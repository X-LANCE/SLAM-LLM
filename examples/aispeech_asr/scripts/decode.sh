#!/bin/bash
set -e 
run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/github/SLAM-LLM-NPU
cd $run_dir
code_dir=examples/aispeech_asr

prompt_style="\{\}" # "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n" | "USER: {}\n ASSISTANT:"
projector=linear
encoder_name=whisper
llm_name=Qwen2.5-7B-Instruct
use_peft=false
use_fp16=false
pad_or_trim=true
encoder_projector_ds_rate=5
eval_max_frame_length=1000
ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/aispeech_asr/exp/librispeech/20250322/whisper_linear_Qwen2.5-7B-Instruct_lorafalse_padtrue_normal_asr_speedfalse_specaugfalse-1121/mala_asr_epoch_2_step_25000_best
test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/aishell-1/asr/test


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

decode_log=$ckpt_path/decode
python \
    $code_dir/inference_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    hydra.run.dir=$ckpt_path \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.normalize=true \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.encoder_projector=$projector \
    ++dataset_config.prompt_style=$prompt_style \
    ++dataset_config.dataset=$dataset \
    ++dataset_config.pad_or_trim=$pad_or_trim \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.input_type=$input_type \
    ++dataset_config.mel_size=$mel_size \
    ++dataset_config.inference_mode=true \
    ++train_config.model_name=aispeech_asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=dynamic \
    ++train_config.num_epochs=1 \
    ++train_config.num_workers_dataloader=0 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/model.pt  



