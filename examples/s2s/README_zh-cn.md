# SLAM-Omni

[English Version](README.md)

## 环境设置

在完成 SLAM-LLM 的环境设置后，运行以下命令设置 SLAM-Omni 的环境：
```bash
cd ./examples/s2s
pip install -r requirements.txt
```

或者使用我们提供的 Docker 镜像：
```bash
docker pull worstchan/slam-omni:v0
docker run -it --gpus all --name slam-omni worstchan/slam-omni:v0 /bin/bash
```

## 数据准备

我们的项目支持两种数据格式：**Parquet** 和 **JSONL**。我们提供的开源语音到语音数据集以 **Parquet** 格式存储在 Hugging Face Hub 上。您可以参考 [此笔记本](./demo/demo_data/demo.ipynb) 获取如何使用这些数据集的示例。

### Parquet

您可以使用以下命令直接从 Hugging Face Hub 下载 [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) 数据集：
```python
from datasets import load_dataset
ds = load_dataset("gpt-omni/VoiceAssistant-400K")
```

### JSONL

我们也支持 **JSONL** 格式的数据，这种格式在结构上更简洁。以下是一个 JSONL 文件的简单示例：
```jsonl
{"key": "1", "source_wav": "/xxx/1.wav", "source_text": "Can you recommend some Chinese food for me?", "target_wav": "/xxx/1.wav", "target_text": "Sure! I recommend trying dumplings, Peking duck, and mapo tofu for a mix of flavors and textures in Chinese cuisine. These dishes offer a good balance of savory, spicy, and crispy elements."}
```

## 模型训练

### S2S 预训练（不推荐）/ TTS

如果要在 SLAM-Omni 框架中基于 **TTS** 任务预训练 S2S 模型（不推荐）或直接训练一个 TTS 模型，可以运行以下命令：
```bash
bash ./examples/s2s/scripts/pretrain/pretrain_tts.sh
```

### 微调

我们提供了三种微调选项，包括 **SLAM-Omni** 和 **Mini-Omni** 建模方式。您可以运行以下命令对模型进行微调：
```bash
# 使用分组策略的微调（推荐）
bash ./examples/s2s/scripts/finetune/finetune_s2s_group.sh

# 不使用分组策略的微调
bash ./examples/s2s/scripts/finetune/finetune_s2s.sh

# Mini-Omni 框架
bash ./examples/s2s/scripts/finetune/mini-omni/finetune_s2s.sh
```

## 推理

我们提供了在线推理和批量推理脚本。更多使用提示和详细信息，请参考 [./examples/s2s/scripts/inference/README.md](./scripts/inference/README.md)。

### 在线推理

我们为 S2S 任务提供了多种在线推理选项。只需输入一个 wav 文件，脚本即可使用训练好的模型生成文本和语音输出。

```bash
# 多轮推理（推荐）
bash ./examples/s2s/scripts/inference/inference_s2s_online_multi-round.sh

# 单轮推理
bash ./examples/s2s/scripts/inference/inference_s2s_online.sh

# Mini-Omni 框架（单轮推理 + 非流式 / 流式）
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online.sh
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online_stream.sh
```

### 批量推理

使用训练好的模型执行批量推理任务时，请确保数据格式与训练时保持一致（**Parquet** 或 **JSONL**）。然后运行以下命令：

```bash
# SLAM-Omni 框架
bash ./examples/s2s/scripts/inference/inference_s2s_batch.sh

# Mini-Omni 框架
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_snac.sh
```

您还可以使用 TTS 预训练模型执行 TTS 推理任务：
```bash
bash ./examples/s2s/scripts/inference_tts.sh
```

<!-- ## 模型评估
待完成

## Gradio Demo
待完成 -->

## 致谢

- 部分代码借鉴自 [Mini-Omni](https://github.com/gpt-omni/mini-omni)，用于基于 SNAC token 的建模。
- 部分代码借鉴自 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)，用于 codec vocoder。

## TODO

- [ ] 添加模型评估脚本
- [ ] 添加 Gradio demo

