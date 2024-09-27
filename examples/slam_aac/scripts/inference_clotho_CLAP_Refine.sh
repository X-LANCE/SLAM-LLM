#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

run_dir=/data/wenxi.chen/SLAM-LLM
cd $run_dir
code_dir=examples/slam_aac

audio_encoder_path=/data/xiquan.li/models/EAT-base_epoch30_ft.pt
llm_path=/data/xiquan.li/models/vicuna-7b-v1.5

seed=42
encoder_projector_ds_rate=5

inference_data_path=/data/wenxi.chen/data/clotho/evaluation_single.jsonl
output_dir=/data/wenxi.chen/cp/wavcaps_pt_v7_epoch4-clotho_ft-seed10086_btz4_lr8e-6-short_prompt_10w/aac_epoch_1_step_4500


beam_range=(2 3 4 5 6 7 8)

for num_beams in "${beam_range[@]}"; do
    decode_log=$output_dir/decode_beam${num_beams}

    if [ -f "$decode_log" ]; then
        echo "Decode log $decode_log already exists, skipping this beam size..."
        continue
    fi

    echo "Running inference with num_beams=$num_beams"

    python $code_dir/inference_aac_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$output_dir \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=eat \
        ++model_config.encoder_path=$audio_encoder_path \
        ++model_config.encoder_dim=768 \
        ++model_config.encoder_projector=linear \
        ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++model_config.normalize=true \
        ++dataset_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
        ++dataset_config.dataset=audio_dataset \
        ++dataset_config.val_data_path=$inference_data_path \
        ++dataset_config.fbank_mean=-4.268 \
        ++dataset_config.fbank_std=4.569 \
        ++dataset_config.model_name=eat \
        ++dataset_config.inference_mode=true \
        ++dataset_config.normalize=true \
        ++dataset_config.input_type=mel \
        ++dataset_config.fixed_length=true \
        ++dataset_config.target_length=1024 \
        ++train_config.model_name=aac \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=8 \
        ++train_config.num_workers_dataloader=8 \
        ++train_config.output_dir=$output_dir \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=false \
        ++train_config.use_peft=true \
        ++ckpt_path=$output_dir/model.pt \
        ++peft_ckpt=$output_dir \
        ++decode_log=$decode_log \
        ++model_config.num_beams=$num_beams
done

# note: to inference model trained the linear layer only, you could set '++train_config.use_peft=false' and 'train_config.freeze_llm=true'
# bash /data/wenxi.chen/SLAM-LLM/examples/slam_aac/scripts/inference_clotho_CLAP_Refine.sh