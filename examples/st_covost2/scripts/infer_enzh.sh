

export MASTER_ADDR=localhost
export MASTER_PORT=12345

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline


your_code=/code

your_data=/userhome






source=covost_enzh

encoder_path_hf=${your_data}/speech/models/whisper-large-v3
checkpoint_dir=${your_data}/speech/models/cotst
llm_path=${your_data}/speech/models/Qwen2-7B

val_data_path=${your_data}/speech/data/qwen/test_spt_0926.jsonl

run_dir=${your_code}/SLAM-LLM
cd $run_dir
code_dir=examples/st_covost2


ckpt_name=$checkpoint_dir/model.pt
echo $ckpt_name


decode_log=$checkpoint_dir/$source.jsonl

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/inference_asr_batch.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++fsdp_config.pure_bf16=true \
        ++model_config.llm_name="vicuna-7b-v1.5" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=3584 \
        ++model_config.query_len=80 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_path_hf=$encoder_path_hf \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=q-former \
        ++dataset_config.dataset=st_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.fix_length_audio=80 \
        ++dataset_config.mel_size=128 \
        ++dataset_config.inference_mode=true \
        ++dataset_config.source=$source \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=12 \
        ++train_config.num_workers_dataloader=16 \
        ++train_config.output_dir=$checkpoint_dir \
        ++log_config.decode_log=$decode_log \
        ++model_config.ckpt_path=$ckpt_name \
        $hydra_args
fi

python ${your_code}/SLAM-LLM/examples/st_covost2/test_werbleu.py --file $decode_log --task st