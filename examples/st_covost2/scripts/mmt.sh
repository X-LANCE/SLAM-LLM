export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
# export HYDRA_FULL_ERROR=1

your_code=/code

your_data=/userhome




run_dir=${your_code}/SLAM-LLM
cd $run_dir
code_dir=examples/st_covost2

# speech_encoder_path=${your_data}/speech/models/whisper/large-v3.pt
encoder_path_hf=${your_data}/speech/models/whisper-large-v3
llm_path=${your_data}/speech/models/Qwen2-7B

train_data_path=${your_data}/speech/data/qwen/train_spt_0926.jsonl
val_data_path=${your_data}/speech/data/qwen/dev_spt_0926.jsonl



source=covost_enenzh


checkpoint_dir=${your_data}/speech/data/qwen/asr-pretrain
output_dir=${your_data}/speech/data/qwen/mmt




# 查找以asr_epoch_开头的目录，提取epoch和step，并找出最大的epoch和step
max_epoch=$(ls -d ${checkpoint_dir}/asr_epoch_*_step_* | sed -n 's/.*asr_epoch_\([0-9]*\)_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)
max_step=$(ls -d ${checkpoint_dir}/asr_epoch_${max_epoch}_step_* | sed -n 's/.*asr_epoch_[0-9]*_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)

# 构建最终的路径
final_path="${checkpoint_dir}/asr_epoch_${max_epoch}_step_${max_step}"


ckpt_name=$final_path/model.pt

# 使用find命令搜索所有.pt文件，并获取最后修改日期最晚的文件






hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=Qwen \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=3584 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_path_hf=$encoder_path_hf \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=q-former \
++model_config.query_len=80 \
++dataset_config.dataset=st_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128  \
++dataset_config.fix_length_audio=80 \
++dataset_config.source=$source \
++train_config.model_name=asr \
++train_config.num_epochs=10 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.gradient_accumulation_steps=1 \
++train_config.warmup_steps=1000 \
++train_config.total_steps=1000000 \
++train_config.lr=1e-4 \
++train_config.batch_size_training=2 \
++train_config.val_batch_size=8 \
++train_config.num_workers_dataloader=16 \
++train_config.output_dir=$output_dir \
++metric=acc \
++train_config.use_fp16=false \
"


# 



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
        --master_port=29504 \
        $code_dir/finetune_asr.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_fsdp=false \
        ++train_config.enable_ddp=true \
        ++fsdp_config.pure_bf16=true \
        ++log_config.use_wandb=false \
        ++log_config.wandb_project_name=SLAM \
        ++train_config.validation_interval=2000 \
        ++log_config.wandb_exp_name=asr \
        ++train_config.use_peft=false \
        $hydra_args
fi
        