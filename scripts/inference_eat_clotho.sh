#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

cd /root/SLAM-LLM


audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M.pt # finetune 定长
# audio_encoder_path=/root/models/EAT/EAT-base_epoch30_finetune_AS2M_v1.pt # finetune 变长


llm_path=/root/models/vicuna-7b-v1.5
num_beams=4


output_dir=/root/exps/wavcaps_btz8_pt2_clotho_ft_seed42_btz4_lr1e-5
ckpt_path=$output_dir/aac/2
# peft_ckpt=/nfs/maziyang.mzy/exps/llama-2-hf-finetune-asr-ds5-proj2048-lr1e-4-whisper-lora-prompt-paddinglr-20240102/asr/4
# val_data_path=/root/data/Rag/clotho_jsonl/evaluation_rag_as_ac_cl.jsonl
val_data_path=/root/data/Clotho_v2/evaluation_single.jsonl

decode_log=$ckpt_path/decode_log_test_clean_beam${num_beams}_repetition_penalty1
# decode_log=$ckpt_path/test2_nucleus

# -m debugpy --listen 6666 --wait-for-client
python src/llama_recipes/pipeline/inference_batch.py \
    --config-path "/root/SLAM-LLM/scripts/conf" \
    --config-name "aac_vicuna_lora.yaml" \
    hydra.run.dir=$ckpt_path \
    model_config.llm_name="vicuna-7b-v1.5" \
    model_config.llm_path=$llm_path \
    model_config.llm_dim=4096 \
    model_config.encoder_name=eat \
    model_config.encoder_path=$audio_encoder_path \
    model_config.encoder_dim=768 \
    model_config.encoder_projector=linear \
    model_config.encoder_projector_ds_rate=5 \
    +model_config.normalize=true \
    dataset_config.dataset=audio_dataset \
    dataset_config.prompt="Describe the audio you hear." \
    dataset_config.val_data_path=$val_data_path \
    dataset_config.fbank_mean=-4.268 \
    dataset_config.fbank_std=4.569 \
    dataset_config.model_name=eat \
    dataset_config.inference_mode=true \
    +dataset_config.normalize=true \
    +dataset_config.input_type=mel \
    train_config.model_name=aac \
    train_config.batching_strategy=custom \
    train_config.num_epochs=1 \
    train_config.val_batch_size=4 \
    train_config.num_workers_dataloader=4 \
    train_config.output_dir=$output_dir \
    +decode_log=$decode_log \
    train_config.freeze_encoder=true \
    train_config.freeze_llm=true \
    train_config.use_peft=true \
    train_config.peft_config.peft_method=lora \
    +ckpt_path=$ckpt_path/model.pt \
    +peft_ckpt=$ckpt_path \
    dataset_config.fixed_length=true \
    dataset_config.target_length=1024 \
    model_config.num_beams=$num_beams \

    # dataset_config.use_rag=true \
    # model_config.do_sample=true \
    # model_config.top_p=0.95 \
    # model_config.temperature=0.5
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
# --peft_ckpt $peft_ckpt \
# --use_peft --peft_method lora \