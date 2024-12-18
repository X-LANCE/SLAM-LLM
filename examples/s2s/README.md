# SLAM-Omni
(*Reproduction of the paper SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training.*)

## Environment Setup
Set up the environment using the following commands after preparing the SLAM-LLM environment:
```bash
pip install -r ./examples/s2s/requirements.txt
```

Alternatively, you can use our provided Docker image:
```bash
docker pull worstchan/slam-omni:v0
docker run -it --gpus all --name slam-omni worstchan/slam-omni:v0 /bin/bash
```

## Data Preparation

Our project supports two data formats: **Parquet** and **JSONL**. The open-source speech-to-speech dataset we provide is stored in **Parquet** format on the Hugging Face Hub.  Examples of dataset usage are available in [this notebook](./demo/demo_data/demo.ipynb).

### Parquet
Download the [VoiceAssistant-400K](https://huggingface.co/datasets/worstchan/VoiceAssistant-400K-SLAM-Omni)  dataset from the Hugging Face Hub:
```python
from datasets import load_dataset
ds = load_dataset("worstchan/VoiceAssistant-400K-SLAM-Omni")
```

### JSONL
We also support JSONL format for its concise structure. Below is an example:
```jsonl
{"key": "1", "source_wav": "/xxx/1.wav", "source_text": "Can you recommend some Chinese food for me?", "target_wav": "/xxx/1.wav", "target_text": "Sure! I recommend trying dumplings, Peking duck, and mapo tofu for a mix of flavors and textures in Chinese cuisine. These dishes offer a good balance of savory, spicy, and crispy elements."}
```

## Checkpoints
We reproduced the single-stage fine-tuning results of SLAM-Omni with a group size of **3**. The following checkpoints are available for download:
- [SLAM-Omni (Single-Round Dialogue)](https://drive.google.com/drive/folders/1ZmM1h5ZTvS-piuN-msmctmZdi51GWLAu?usp=sharing)
- [SLAM-Omni (Multi-Round Dialogue, to be released)](url)

The single-round dialogue model is trained on the VoiceAssistant-400K dataset, while the multi-round dialogue model is trained on a combination of VoiceAssistant-400K and UltraChat (not yet released) datasets.


## Training

You can pre-train the S2S model using TTS or ASR tasks with our provided scripts, though we recommend proceeding directly to fine-tuning. Alternatively, you may directly train a TTS or ASR model under the SLAM-Omni framework. For detailed instructions, refer to the [pre-training README](./scripts/pretrain).

### Fine-tuning
We provide three fine-tuning options, supporting both **SLAM-Omni** and **Mini-Omni** modeling. Use the commands below:
```bash
# Fine-tune with grouping strategy (Recommended)
bash ./examples/s2s/scripts/finetune/finetune_s2s_group.sh

# Fine-tune without grouping
bash ./examples/s2s/scripts/finetune/finetune_s2s.sh

# Mini-Omni framework
bash ./examples/s2s/scripts/finetune/mini-omni/finetune_s2s.sh
```

## Inference
We provide scripts for both online and batch inference. You can use the trained model or the provided checkpoints for inference. For detailed guidance, refer to [inference README](./scripts/inference/README.md).



### Online Inference
Run the following commands for online inference in the S2S task. Provide a wav file as input, and the script will generate text and speech outputs:

```bash
# Multi-turn (Recommended)
bash ./examples/s2s/scripts/inference/inference_s2s_online_multi-round.sh

# Single-turn
bash ./examples/s2s/scripts/inference/inference_s2s_online.sh

# Mini-Omni framework (Single-turn + non-streaming/streaming)
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online.sh
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online_stream.sh
```

To perform TTS inference with a pre-trained model, use:
```bash
bash ./examples/s2s/scripts/inference/inference_tts_online.sh
```

### Batch Inference

For batch inference in the S2S task, ensure the data format matches the training format (**Parquet** or **JSONL**). Run the following commands:

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
- We borrow some code from [Mini-Omni](https://github.com/gpt-omni/mini-omni) for SNAC token-based modeling.
- We borrow some code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the vocoder.