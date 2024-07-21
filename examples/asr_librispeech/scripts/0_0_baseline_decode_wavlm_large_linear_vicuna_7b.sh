#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-sxm2:1   # --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=log/%j.output
#SBATCH --error=log/%j.error


#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

module load anaconda3/2022.05
module load ffmpeg/20190305 
module unload cuda/11.2  
module load cuda/11.8  

source activate /work/van-speech-nlp/jindaznb/slamenv/

test_speaker="librispeech"
encoder_name="wavlm"

run_dir=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=${run_dir}/models/WavLM-Large.pt
llm_path=${run_dir}/models/vicuna-7b-v1.5

timestamp=$(date +"%Y%m%d%H%M%S")
output_dir=${run_dir}/out/vicuna-librispeech-linear-steplrwarmupkeep1e-4-whisper-largev3

ckpt_path=${run_dir}/models
split=other
# val_data_path=/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/slam-llm/examples/asr_librispeech/data/${test_speaker}_${split}.jsonl
val_data_path=${run_dir}/data/loaded_librispeech_test_${split}.jsonl
decode_log=${run_dir}/out/decode_${test_speaker}_${split}_${encoder_name}_beam4

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
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
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/wavlm_linear_model.pt \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
