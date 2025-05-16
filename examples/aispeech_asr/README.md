# AISPEECH_ASR

## Overview

This example is designed for large-scale industrial data training, suitable for datasets on the order of 100,000 hours. Its main features include:
- **Support for multi-task training**: Designed to support tasks such as ASR and ST through a unified data format.
- **Dynamic prompt selection**: Supports random selection from multiple prompts.
- **Iterative dataset**: Uses an iterative dataset format to reduce startup time for large datasets.
- **Deepspeed training**: Supports DeepSpeed training to significantly reduce memory usage.
- **Multi-machine multi-GPU inference**: Supports distributed inference across multiple machines and GPUs to reduce evaluation time.
- **Dynamic frame batching**: Dynamically combines frames based on audio size rather than using a fixed batch size, significantly reducing training and evaluation time (reduces training time by 3/4 for 100,000 hours of data).

This example is modified from `mala_asr_slidespeech`.

## Model Architecture

The model architecture can be dynamically selected within the scope supported by SLAM-LMM. Below are some recommended configurations:
- **Encoder**: WavLM, Whisper
- **Projector**: Linear
- **LLM**: Qwen2.5-7B-Instruct, Vicuna1.5-7B

## Data Preparation

The following two files are required:
- `multitask.jsonl`
- `multiprompt.jsonl`

### multitask.jsonl

The format of this file is as follows, where `path` supports both ark format and wav files:
```json
{"key": "BAC009S0002W0122", "task": "ASR", "target": "而对楼市成交抑制作用最大的限购", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/train/data/data_wav.1.ark:17"}
{"key": "BAC009S0002W0123", "task": "ASR", "target": "也成为地方政府的眼中钉", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/train/data/data_wav.1.ark:191758"}
{"key": "BAC009S0002W0124", "task": "ASR", "target": "自六月底呼和浩特市率先宣布取消限购后", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/train/data/data_wav.1.ark:315339"}
{"key": "BAC009S0764W0238", "task": "hotword", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/test/data/data_wav.1.ark:17343733", "target": "形成一批具有国际竞争力的中国企业", "hotword": "中国"}
```

### multiprompt.jsonl

The format of this file is as follows:
```json
{"task": "ASR", "prompt": "Transcribe speech to text."}
{"task": "ASR", "prompt": "请识别语音."}
{"task": "ZH2EN", "prompt": "请识别语音并翻译为英文:"}
{"task": "EN2ZH", "prompt": "请识别语音并翻译为中文:"}
{"task": "prevtext", "prompt": "Transcribe speech to text, below are the previous historical transcription texts:{}."}
{"task": "hotword", "prompt": "Transcribe speech to text, follow words may occur:{}."}
```

### Notes
- If multiple prompts are provided, one will be selected dynamically.
- For additional information (e.g., hotwords), include the task-named information in `multitask.jsonl` and use `{}` in the prompt to inject this information. Additionally, update the `append_info_tasks` in the `aispeech_config` file:
  ```python
  append_info_tasks: List = field(default_factory=lambda: ["hotword"])
  ```

## Training a New Model

### Script Preparation

Prepare and modify the following content in `scripts/finetune_deepspeed.sh` or `scripts/finetune_torchrun.sh` (Deepspeed is recommended):
```bash
run_dir=  # Directory to save the model
train_scp_file_path=  # Path to training data
dev_scp_file_path=  # Path to validation data
train_max_frame_length=1500  # Maximum frame length for training
eval_max_frame_length=1000  # Maximum frame length for evaluation
multitask_prompt_path=  # Path to multitask.jsonl
projector=linear  # Type of projector
encoder_name=whisper  # Name of the encoder
llm_name=Qwen2.5-7B-Instruct  # Name of the LLM
use_peft=false  # Whether to use PEFT (for LLM)
use_fp16=true  # Whether to use FP16
freeze_encoder=true  # Whether to freeze the encoder
pad_or_trim=true  # Whether to use pad_or_trim (for Whisper)
deepspeed_config=  # Path to DeepSpeed configuration file
```

Typically, we first train the projector and then fine-tune the LoRA. For projector training, set:
```bash
use_peft=false
```

For LoRA training, set (with `ckpt_path` pointing to the model saved in the previous step):
```bash
use_peft=true
if [[ $use_peft == "true" ]]; then
    ckpt_path=  
fi
```
### Deepspeed
When using `bf16`/`fp16` for training, deepspeed saves about 20GB of GPU memory compared to `torchrun` when training a 7B model. For 7B models, it's recommended to use `zero-0`/`1`/`2`, while for extremely large models, `zero-3` can be used, though communication may become a bottleneck.

```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu"
        }
    }
}
```
Note that when using `zero-0`/`1`/`2`/`3`, the DeepSpeed model is saved as `pytorch_model.bin`
If you use bf16/fp16 training in DeepSpeed and encounter NaN in train/eval loss, check the autocast in `src/slam_llm/utils/deepspeed_utils.py`:

```python
with autocast()  # original code
with autocast(dtype=torch.bfloat16) # must work
with autocast(dtype=torch.float16) 
```
## Decoding

- **Single-machine single-GPU decoding**: Refer to `scripts/decode.sh`
- **Single-machine multi-GPU decoding**: Refer to `scripts/decode_deepspeed.sh`

## Multi-Machine Multi-GPU Support

Multi-machine multi-GPU training can be supported with minor modifications to the `finetune_deepspeed.sh` or `scripts/decode_deepspeed.sh` scripts. Due to environment-specific requirements, this example does not include dedicated scripts for multi-machine multi-GPU setups.
