# s2s-dev

## Demo
You can refer to the `examples/s2s/demo` directory for a simple demo of TTS and S2S tasks.

## Environment Setup
You can set up the environment using the following command once you have setup the environment for SLAM-LLM:
```bash
pip install -r requirements.txt
```

Or you can use our provided docker image:
```bash
docker pull worstchan/slam-omni:v0
docker run -it --gpus all --name slam-omni worstchan/slam-omni:v0 /bin/bash
```

## Data Preparation
You can download the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) dataset from the Hugging Face Hub using the following command:
```python
from datasets import load_dataset
ds = load_dataset("gpt-omni/VoiceAssistant-400K")
```

You can save the dataset to disk using the following command:
```python
save_path = "/path/to/save/directory"
ds.save_to_disk(save_path)
```

If you save the dataset to disk, you can load it using the following command:
```python
from datasets import load_from_disk
ds = load_from_disk(save_path)
```

## Training

### Pre-training
To pre-train the model with **TTS** task using the VoiceAssistant-400K dataset, you can run the following command:
```bash
bash ./examples/s2s/scripts/pretrain_tts.sh
```

### Fine-tuning
To fine-tune the model with **S2S** task using the VoiceAssistant-400K dataset, you can run the following command:
```bash
bash ./examples/s2s/scripts/finetune_s2s.sh
```


## Inference
To generate the text and speech (i.e., to perform S2S task) using the pre-trained model given the speech input, you can run the following command:
```bash
bash ./examples/s2s/scripts/inference_s2s.sh
```

<!-- <!-- 你也可以使用 TTS 预训练的模型执行 TTS 推理任务，使用以下命令： -->
You can also use the TTS pre-trained model to perform TTS inference tasks using the following command:
```bash
bash ./examples/s2s/scripts/inference_tts.sh
```

<!-- 要注意我们目前的推理只支持单个输入，暂不支持批量输入。 -->
Please note that our current inference only supports **single** input and does not support batch input.


## TODO
- [ ] Add ASR pre-training
- [ ] Release baseline pre-trained models
- [ ] Add more datasets
- [ ] Add evaluation scripts