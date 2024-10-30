# s2s-dev

[English Version](README.md)

## 演示
参考 `examples/s2s/demo` 目录中的简单示例，了解 TTS 和 S2S 任务（包括域内和域外示例）。

## 环境设置
在设置好 SLAM-LLM 环境后，可以使用以下命令设置环境：
```bash
cd SLAM-LLM
pip install -r requirements.txt
```

或者使用我们提供的 Docker 镜像：
```bash
docker pull worstchan/slam-omni:v0
docker run -it --gpus all --name slam-omni worstchan/slam-omni:v0 /bin/bash
```

## 数据准备
可以使用以下命令从 Hugging Face Hub 下载 [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) 数据集：
```python
from datasets import load_dataset
ds = load_dataset("gpt-omni/VoiceAssistant-400K")
```

使用以下命令将数据集保存到磁盘：
```python
save_path = "/path/to/save/directory"
ds.save_to_disk(save_path)
```

如果将数据集保存到磁盘，可以使用以下命令加载：
```python
from datasets import load_from_disk
ds = load_from_disk(save_path)
```

## 训练

### 预训练
使用 VoiceAssistant-400K 数据集进行 **TTS** 任务的预训练，可以运行以下命令：
```bash
bash ./examples/s2s/scripts/pretrain/pretrain_tts.sh
```

### 微调
使用 VoiceAssistant-400K 数据集进行 **S2S** 任务的微调，可以运行以下命令：
```bash
# 使用 SNAC 编解码器进行微调
bash ./examples/s2s/scripts/finetune_s2s.sh

# 使用 CosyVoice 编解码器进行微调
bash ./examples/s2s/scripts/finetune/finetune_s2s_cosyvoice.sh

# 使用 CosyVoice 编解码器和分组策略进行微调
bash ./examples/s2s/scripts/finetune/finetune_s2s_cosyvoice_group.sh
```

## 推理
使用预训练模型进行文本和语音生成（即执行 S2S 任务），可以运行以下命令：
```bash
# 使用 SNAC 编解码器进行推理
bash ./examples/s2s/scripts/inference/inference_s2s.sh

# 使用 CosyVoice 编解码器进行推理
bash ./examples/s2s/scripts/inference/inference_s2s_cosyvoice.sh

# 使用 CosyVoice 编解码器和分组策略进行推理
bash ./examples/s2s/scripts/inference/inference_s2s_cosyvoice_group.sh
```

使用 TTS 预训练模型进行 TTS 推理任务，可以运行以下命令：
```bash
bash ./examples/s2s/scripts/inference_tts.sh
```

### 注意
- 当前推理仅支持 **单个** 输入，不支持批量输入。
- 提供两种推理模式：`仅文本` 和 `文本 & 语音`。可以在推理脚本中设置 `decode_text_only` 参数选择所需模式。
- 如果在推理过程中使用 CosyVoice 编解码器，可以通过设置 `audio_prompt_path` 自由选择输出语音音调。我们在 `prompt` 目录中提供了一些可选语音。如果未指定，将使用默认语音。

## 在线推理
我们还提供了 S2S 任务的在线推理脚本。只需输入 wav 文件，脚本将生成文本和语音输出。可以运行以下命令：
```bash
# 使用 SNAC 编解码器进行在线推理
bash ./examples/s2s/scripts/inference/inference_s2s_online.sh

# 使用 CosyVoice 编解码器进行在线推理
bash ./examples/s2s/scripts/inference/inference_s2s_online_cosyvoice.sh
```

此外，我们还提供了 S2S 任务的流式在线推理脚本。可以运行以下命令：
```bash
# 使用 SNAC 编解码器进行流式在线推理
bash ./examples/s2s/scripts/inference/inference_s2s_online_stream.sh
```

## TODO
- [ ] 添加更多数据集
- [ ] 添加评估脚本
