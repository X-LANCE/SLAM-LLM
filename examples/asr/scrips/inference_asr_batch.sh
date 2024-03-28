#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/work/SLAM-LLM
cd $run_dir
code_dir=examples/asr/

speech_encoder_path=/cxgroup/model/whisper/large-v3.pt

# llm_path=/cxgroup/model/Llama-2-7b-chat-hf
llm_path=/cxgroup/model/vicuna-7b-v1.5

output_dir=/work/exps/vicuna-7b-v1.5-mls-french-linear-lora-32-steplrwarmupkeep1e-4-whisper-largev3-20240313-test
ckpt_path=$output_dir/asr/1
val_data_path=data/mls/french_test.jsonl
decode_log=$ckpt_path/long_prompt_french_test_beam4

# -m debugpy --listen 5678 --wait-for-client
# python $code_dir/inference_asr_batch.py \

# prompt="Transcrire la parole en français. Sortez la transcription directement sans contenu redondant. Assurez-vous que la sortie n'est pas dupliquée. "
# python -m debugpy --listen 5678 --wait-for-client $code_dir/inference_asr_batch.py \
python $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=128 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=4 \
        ++train_config.num_workers_dataloader=4 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        # ++dataset_config.prompt="${prompt}" \
        # ++peft_ckpt=$ckpt_path \
        # ++train_config.use_peft=true \
        # ++train_config.peft_config.r=32 \
        # ++dataset_config.normalize=true \
        # ++model_config.encoder_projector=q-former \
        # ++dataset_config.fix_length_audio=64 \
        # --peft_ckpt $peft_ckpt \
        # ++ckpt_path=$ckpt_path/model.pt \
        # --use_peft --peft_method lora \

python src/slam_llm/utils/whisper_tn.py ${decode_log}_gt ${decode_log}_gt.proc
python src/slam_llm/utils/whisper_tn.py ${decode_log}_pred ${decode_log}_pred.proc
python src/slam_llm/utils/compute_wer.py ${decode_log}_gt.proc ${decode_log}_pred.proc ${decode_log}.proc.wer
# python src/slam_llm/utils/compute_wer.py ${decode_log}_gt ${decode_log}_pred ${decode_log}_wer
