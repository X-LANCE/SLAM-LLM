#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
run_dir=/hpc_stor01/home/yangui.fang_sx/workingspace/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech
dataset=aishell2
projector=linear
encoder_name=whisper
llm_name=Qwen2.5-7B-Instruct
use_peft=false
ckpt_path=/hpc_stor01/home/yangui.fang_sx/workingspace/project/asr_librispeech_origin/exp/whisper_Qwen2.5-7B-Instruct_aishell1_linear_lorafalse_20241202-1342/asr_epoch_4_step_2482/
output_dir=$ckpt_path

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
elif [[ $llm_name == "Qwen2-7B" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2-7B
    llm_dim=3584 
    use_fp16=true
elif [[ $llm_name == "Qwen2.5-1.5B-Instruct" ]]
then
    llm_path=/hpc_stor01/home/yangui.fang_sx/workingspace/model/Qwen2.5-1.5B-Instruct
    llm_dim=1536 
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
    val_data_path=/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr_librispeech/test.jsonl
else
    exit 1
fi


decode_log=$ckpt_path/decode_${dataset}_beam4
# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=$llm_name \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=128 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=6 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \

python /hpc_stor01/home/yangui.fang_sx/workingspace/tools/pyResults/pyResults.py ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer
# python "/hpc_stor01/home/yangui.fang_sx/workingspace/tools/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer"
# python "/hpc_stor01/home/yangui.fang_sx/workingspace/github/SLAM-LLM/examples/mala_asr_slidespeech/slam_llm/utils/compute_wer.py  ${decode_log}_gt ${decode_log}_pred ${decode_log}_ser"