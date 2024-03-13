# SLAM-LLM: **S**peech, **L**anguage, **A**udio, **M**usic Processing with Large Language Model

# News
- [Update Mar. 13, 2024] Please join [slack](https://join.slack.com/t/slam-llm/shared_invite/zt-2cxmm7fue-tEKmZcL1hB8s2R2GQdTTiA). We will sync our updates here.


# Table of Contents
1. [Setup](#setup)
2. [Fine-tuning](#fine-tuning)
    - [Single GPU](#single-gpu)
    - [Multi GPU One Node](#multiple-gpus-one-node)
    - [Multi GPU Multi Node](#multi-gpu-multi-node)
3. [Inference](#inference)
    - [Batch Inference](#batch-inference)
    - [Real-time Inference](#real-time-inference)
4. [License and Acceptable Use Policy](#license)
5. [Citation](#citation)

# Setup

## Installation
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout tags/v4.35.2
pip install -e .
cd ..
git clone https://github.com/huggingface/peft.git
cd peft
git checkout tags/0.6.0
pip install -e .
cd ..
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
git clone git@github.com:ddlBoJack/SLAM-LLM.git
cd SLAM-LLM
pip install  -e .
```

**For more in depth information checkout the following:**

* [Single GPU Fine-tuning](./docs/single_gpu.md)
* [Multi-GPU Fine-tuning](./docs/multi_gpu.md)
* [LLM Fine-tuning](./docs/LLM_finetuning.md)
* [Adding custom datasets](./docs/Dataset.md)
* [Inference](./docs/inference.md)
* [FAQs](./docs/FAQ.md)

# Fine-tuning

We take Automatic Speech Recognition (ASR) with Large Language Models (LLM) as an example to demonstrate the fine-tuning process. The same process can be applied to other tasks in [example](./examples)(TODO) and [scripts](./scripts) folder. 

## Single and Multi GPU Finetune

If you want to dive right into single or multi GPU fine-tuning, run the examples below on a single GPU like A10, T4, V100, A100 etc.
All the parameters in the examples and recipes below need to be further tuned to have desired results based on the model, method, data and task at hand.

### Single GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
cd /root/SLAM-LLM

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
output_dir=/nfs/maziyang.mzy/exps/finetune-asr-whisper-largev2-vicuna-7b-v1.5-linear-ds5-proj2048-stlr-lora

python src/slam-llm/pipeline/finetune.py \
--config-path "scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=speech_dataset \
++dataset_config.prompt="Transcribe speech to text. " \
++dataset_config.train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
++dataset_config.val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other_filtered.jsonl \
++dataset_config.input_type=mel \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++log_config.log_file=/$output_dir/train.log \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir \
++log_config.wandb_entity_name=zym22 \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=${0##*/%.*} \
++log_config.log_interval 5 \
++metric=acc \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
```

Here we make use of Parameter Efficient Methods (PEFT) as described in the next section. To run the command above make sure to pass the `peft_method` arg which can be set to `lora`, `llama_adapter` or `prefix`.

**Note** if you are running on a machine with multiple GPUs please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`


### Multiple GPUs One Node with DDP:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
cd /root/SLAM-LLM

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
output_dir=/nfs/maziyang.mzy/exps/finetune-asr-whisper-largev2-vicuna-7b-v1.5-linear-ds5-proj2048-stlr-lora

torchrun \
--nnodes 1 \
--nproc_per_node 4 \
src/llama_recipes/pipeline/finetune.py \
--config-path "scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$output_dir \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=speech_dataset \
++dataset_config.prompt="Transcribe speech to text. " \
++dataset_config.train_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_train_960h.jsonl \
++dataset_config.val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_dev_other.jsonl \
++dataset_config.input_type=mel \
++train_config.model_name=asr \
++train_config.enable_fsdp=false \
++train_config.enable_ddp=true \
++train_config.use_fp16=true \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=100000 \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++log_config.log_file=/$output_dir/train.log \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir \
++log_config.wandb_entity_name=zym22 \
++log_config.wandb_project_name=slam-llm \
++log_config.wandb_exp_name=${0##*/%.*} \
++log_config.log_interval 5 \
++metric=acc \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
```
If you want to run with FSDP, you can set `++train_config.enable_fsdp=true` and `++train_config.enable_ddp=false`.

### Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up the fine-tuning job. This has been enabled in `optimum` library from HuggingFace as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

### Fine-tuning using FSDP on 70B Model

If you are interested in running full parameter fine-tuning on the 70B model, you can enable `low_cpu_fsdp` mode as the following command. This option will load model on rank0 only before moving model to devices to construct FSDP. This can dramatically save cpu memory when loading large models like 70B (on a 8-gpu node, this reduces cpu memory from 2+T to 280G for 70B model). This has been tested with `BF16` on 16xA100, 80GB GPUs.

### Multi GPU Multi Node:

```bash

sbatch multi_node.slurm
# Change the num nodes and GPU per nodes in the script before running.

```
You can read more about our fine-tuning strategies [here](./docs/LLM_finetuning.md).

# Inference

Once you have fine-tuned the model(for example, whisper + vicuna + linear + lora), you can use the following command to run inference on the fine-tuned model.

## Batch Inference

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
cd /root/SLAM-LLM

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
output_dir=/nfs/maziyang.mzy/exps/finetune-asr-whisper-largev2-vicuna-7b-v1.5-linear-ds5-proj2048-stlr-lora
ckpt_path=$output_dir/asr/2
decode_log=$ckpt_path/decode_log_test_clean_beam4

python src/llama_recipes/pipeline/inference_batch.py \
--config-path "scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
hydra.run.dir=$ckpt_path \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++dataset_config.dataset=speech_dataset \
++dataset_config.prompt="Transcribe speech to text. " \
++dataset_config.val_data_path=/nfs/maziyang.mzy/data/librispeech/librispeech_test_clean.jsonl \
++dataset_config.input_type=mel \
++dataset_config.inference_mode=true \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++train_config.batching_strategy=custom \
++train_config.num_epochs=1 \
++train_config.val_batch_size=4 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++ckpt_path=$ckpt_path/model.pt \
++peft_ckpt=$ckpt_path \
++decode_log=$decode_log \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
```

## Real-time Inference

```bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
cd /root/SLAM-LLM

speech_encoder_path=/nfs/maziyang.mzy/models/Whisper/large-v2.pt
llm_path=/nfs/maziyang.mzy/models/vicuna-7b-v1.5
output_dir=/nfs/maziyang.mzy/exps/finetune-asr-whisper-largev2-vicuna-7b-v1.5-linear-ds5-proj2048-stlr-lora
ckpt_path=$output_dir/asr/2

python src/llama_recipes/pipeline/inference.py \
--config-path "scripts/conf" \
--config-name "asr_vicuna_lora.yaml" \
++model_config.llm_name="vicuna-7b-v1.5" \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=4096 \
++model_config.encoder_name=whisper \
++model_config.encoder_ds_rate=2 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=linear \
++model_config.encoder_projector_ds_rate=5 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=false \
++train_config.use_peft=true \
++train_config.peft_config.peft_method=lora \
++ckpt_path=$ckpt_path/model.pt \
++peft_ckpt=$ckpt_path \
++decode_log=$decode_log \
# ++model_config.encoder_projector=q-former \
# ++dataset_config.fix_length_audio=64 \
```

# License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)

# Citation

```
@article{ma2024embarrassingly,
  title={An Embarrassingly Simple Approach for LLM with Strong ASR Capacity},
  author={Ma, Ziyang and Yang, Guanrou and Yang, Yifan and Gao, Zhifu and Wang, Jiaming and Du, Zhihao and Yu, Fan and Chen, Qian and Zheng, Siqi and Zhang, Shiliang and others},
  journal={arXiv preprint arXiv:2402.08846},
  year={2024}
}
```
