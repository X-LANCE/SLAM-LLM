# AISPEECH_ASR

## 概述

这是为工业界大规模数据训练准备的示例，适用于10万小时量级的数据训练，主要特点如下：
- **多任务训练支持**：通过设计数据格式，支持包括ASR、ST等多种任务。
- **动态Prompt选择**：支持在多个Prompt中随机选择。
- **迭代式dataset**：采用迭代形式的dataset，减少大数据量时的启动时间。
- **Deepspeed训练**：支持Deepspeed训练，显著减少内存使用。
- **多机多卡推理**：支持多机多卡推理，减少评估时间。
- **动态帧数组合**：根据每个音频大小动态组合合适的帧数进行训练，而非使用固定的batch_size，大大减少了训练和评估时间（在10万小时量级的数据上，训练时间减少了3/4）。
- **昇腾NPU适配**：适配支持昇腾NPU。

本示例基于`mala_asr_slidespeech`进行修改。

## 模型架构

可以根据需要，在SLAM—LMM支持的范围内动态选择模型架构。以下是一些推荐的模型配置：
- **Encoder**：WavLM, Whisper
- **Projector**：Linear
- **LLM**：Qwen2.5-7B-Instruct, Vicuna1.5-7B

## 数据准备

需要准备以下两个文件：
- `multitask.jsonl`
- `multiprompt.jsonl`

### multitask.jsonl

该文件的内容格式如下，其中`path`支持ark格式和wav文件：
```json
{"key": "BAC009S0002W0122", "task": "ASR", "target": "而对楼市成交抑制作用最大的限购", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/train/data/data_wav.1.ark:17"}
{"key": "BAC009S0002W0123", "task": "ASR", "target": "也成为地方政府的眼中钉", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/train/data/data_wav.1.ark:191758"}
{"key": "BAC009S0002W0124", "task": "ASR", "target": "自六月底呼和浩特市率先宣布取消限购后", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/train/data/data_wav.1.ark:315339"}
{"key": "BAC009S0764W0238", "task": "hotword", "path": "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/test/data/data_wav.1.ark:17343733", "target": "形成一批具有国际竞争力的中国企业", "hotword": "中国"}
```

### multiprompt.jsonl

该文件的内容格式如下：
```json
{"task": "ASR", "prompt": "Transcribe speech to text."}
{"task": "ASR", "prompt": "请识别语音."}
{"task": "ZH2EN", "prompt": "请识别语音并翻译为英文:"}
{"task": "EN2ZH", "prompt": "请识别语音并翻译为中文:"}
{"task": "prevtext", "prompt": "Transcribe speech to text, below are the previous historical transcription texts:{}."}
{"task": "hotword", "prompt": "Transcribe speech to text, follow words may occur:{}."}
```

### 注意事项
- 如果有多条Prompt，会动态选择其中一条。
- 如果有额外信息（如热词），请在`multitask.jsonl`中提供与任务同名的信息，并在Prompt中使用`{}`注入该信息。同时，修改`aispeech_config`文件中的`append_info_tasks`：
  ```python
  append_info_tasks: List = field(default_factory=lambda: ["hotword"])
  ```

## 训练新模型

### 脚本准备

在`scripts/finetune_deepspeed.sh`或`scripts/finetune_torchrun.sh`中准备并修改以下内容（推荐使用Deepspeed）：
```bash
run_dir=  # 模型保存目录
train_scp_file_path=  # 训练数据路径
dev_scp_file_path=  # 验证数据路径
train_max_frame_length=1500  # 训练时的最大帧长度
eval_max_frame_length=1000  # 评估时的最大帧长度
multitask_prompt_path=  # multitask.jsonl文件路径
prompt_style="\{\}"  # Prompt样式，可选格式如"<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"或"USER: {}\n ASSISTANT:"
projector=linear  # Projector类型
encoder_name=whisper  # Encoder名称
llm_name=Qwen2.5-7B-Instruct  # LLM名称
use_peft=false  # 是否使用PEFT（对于LLM）
use_fp16=true  # 是否使用FP16
freeze_encoder=true  # 是否冻结Encoder
pad_or_trim=true  # 是否使用pad_or_trim（对于Whisper）
deepspeed_config=  # DeepSpeed配置文件路径
```

通常，我们首先训练Projector，然后再训练LoRA。训练Projector时，设置如下：
```bash
use_peft=false
```

训练LoRA时，设置如下（`ckpt_path`是上一步训练保存的模型路径）：
```bash
use_peft=true
if [[ $use_peft == "true" ]]; then
    ckpt_path=  # 如果是DDP训练，直接写入保存的pt文件路径；如果是Deepspeed训练，需将mp_rank_00_model_states.pt文件转化为model.pt，可使用`scripts/transcribe_deepspeed_to_pt.py`脚本
fi
```

## 解码

- **单机单卡解码**：参考`scripts/decode.sh`
- **单机多卡解码**：参考`scripts/decode_deepspeed.sh`

## 多机多卡支持
简单修改脚本finetune_deepspeed.sh 或者scripts/decode_deepspeed.sh`后可以支持多机多卡训练，因为环境不同所做的修改也不同，本实例就不放出多机多卡的脚本了