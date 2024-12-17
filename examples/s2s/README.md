# SLAM-Omni
(*Reproduction of the paper SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training.*)

## Environment Setup
Set up the environment using the following command after setting up the environment for SLAM-LLM:
```bash
cd ./examples/s2s
pip install -r requirements.txt
```

Or you can use our provided docker image:
```bash
docker pull worstchan/slam-omni:v0
docker run -it --gpus all --name slam-omni worstchan/slam-omni:v0 /bin/bash
```

## Data Preparation

Our project supports two data formats: **Parquet** and **JSONL**. The open-source speech-to-speech dataset we provided is stored in **Parquet** format on the Hugging Face Hub.  You can find examples of how to use these datasets in [this notebook](./demo/demo_data/demo.ipynb).

### Parquet
You can directly download the [VoiceAssistant-400K](https://huggingface.co/datasets/worstchan/VoiceAssistant-400K-SLAM-Omni) dataset from the Hugging Face Hub using the following commands:
```python
from datasets import load_dataset
ds = load_dataset("worstchan/VoiceAssistant-400K-SLAM-Omni")
```

### JSONL
We also support data in JSONL format, which offers a more concise structure. Below is a simple example of a JSONL file:  
```jsonl
{"key": "1", "source_wav": "/xxx/1.wav", "source_text": "Can you recommend some Chinese food for me?", "target_wav": "/xxx/1.wav", "target_text": "Sure! I recommend trying dumplings, Peking duck, and mapo tofu for a mix of flavors and textures in Chinese cuisine. These dishes offer a good balance of savory, spicy, and crispy elements."}
```

## Checkpoints
We reproduced the single-stage fine-tuning results of SLAM-Omni on the VoiceAssistant-400K dataset with a group size of $ G = 3 $. You can download the [checkpoint](url) from Google Drive.

## Training

You can pre-train the S2S model using TTS or ASR tasks with our provided scripts, but we recommend going directly to fine-tuning. You may also directly train a TTS or ASR model under the SLAM-Omni framework. For more details, please refer to the [pre-training README](./scripts/pretrain).

### Fine-tuning
We provide three fine-tuning options, covering both **SLAM-Omni** modeling and **Mini-Omni** modeling. You can use the commands below to fine-tune the model:
```bash
# finetune with grouping strategy (Recommended)
bash ./examples/s2s/scripts/finetune/finetune_s2s_group.sh

# finetune without grouping
bash ./examples/s2s/scripts/finetune/finetune_s2s.sh

# Mini-Omni framework
bash ./examples/s2s/scripts/finetune/mini-omni/finetune_s2s.sh
```

## Inference
We provide online and batch inference scripts. You can use the trained model or the [checkpoint](url) we provide for inference. For more tips and details, please refer to [./examples/s2s/scripts/inference/README.md](./scripts/inference/README.md).


### Online Inference
We provide multiple options for online inference in the S2S task. Simply input a wav file, and the script will generate both text and speech outputs with the trained model.

```bash
# Multi-turn (Recommended)
bash ./examples/s2s/scripts/inference/inference_s2s_online_multi-round.sh

# Single-turn
bash ./examples/s2s/scripts/inference/inference_s2s_online.sh

# Mini-Omni framework (Single-turn + non-streaming/streaming)
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online.sh
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online_stream.sh
```

You can also use a TTS pre-trained model to perform TTS inference tasks with the following command:
```bash
bash ./examples/s2s/scripts/inference/inference_tts_online.sh
```

### Batch Inference

To perform batch inference on the S2S task using trained models, ensure the data format matches the one used during training (either **Parquet** or **JSONL**). Then, run the following commands:

```bash
# SLAM-Omni framework
bash ./examples/s2s/scripts/inference/inference_s2s_batch.sh

# Mini-Omni framework
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_snac.sh
```


<!-- ## Evaluation
TBD

## Gradio Demo
TBD -->


## Acknowledgement
- We borrow some code from [Mini-Omni](https://github.com/gpt-omni/mini-omni) for the modeling based on the SNAC token.
- We borrow some code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the codec vocoder.

## TODO
- [ ] Add evaluation scripts
- [ ] Add Gradio demo