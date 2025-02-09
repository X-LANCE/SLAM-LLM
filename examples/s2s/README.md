# SLAM-Omni
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![arXiv](https://img.shields.io/badge/arXiv-2412.15649-B31B1B.svg)](https://arxiv.org/abs/2412.15649) [![GitHub Demo Page](https://img.shields.io/badge/Github-Demo%20Page-orange.svg)](https://slam-omni.github.io/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

(*Reproduction of the [paper](https://arxiv.org/abs/2412.15649) SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training.*)

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

Our project supports two data formats: **Parquet** and **JSONL**. The open-source datasets are available on the Hugging Face Hub in **Parquet** format. Examples usage is provided in [this notebook](./demo/demo_data/demo.ipynb).

### Supported Datasets
We provide three re-synthesized datasets for SLAM-Omni training: 
- [VoiceAssistant-400K](https://huggingface.co/datasets/worstchan/VoiceAssistant-400K-SLAM-Omni): Single-round English dialogue dataset. 
- [UltraChat-300K](https://huggingface.co/datasets/worstchan/UltraChat-300K-SLAM-Omni): Multi-round English dialogue dataset. 
- [Belle_1.4M](https://huggingface.co/datasets/worstchan/Belle_1.4M-SLAM-Omni): Multi-round Chinese dialogue dataset.

#### Usage
You can load any of these datasets using the following code:
```python
from datasets import load_dataset

# Replace "DATASET_NAME" with one of the following:
# - "worstchan/VoiceAssistant-400K-SLAM-Omni"
# - "worstchan/UltraChat-300K-SLAM-Omni"
# - "worstchan/Belle_1.4M-SLAM-Omni"

ds = load_dataset("DATASET_NAME")
```

### JSONL
We also support JSONL format for its concise structure. Below is an example:
```jsonl
{"key": "1", "source_wav": "/xxx/1.wav", "source_text": "Can you recommend some Chinese food for me?", "target_token": [742, 383, 455, ..., 180, 619], "target_text": "Sure! I recommend trying dumplings, Peking duck, and mapo tofu for a mix of flavors and textures in Chinese cuisine. These dishes offer a good balance of savory, spicy, and crispy elements."}
```

🔔**Update**:
We now use `target_token` to replace the `target_wav` field. When using your own data, you need to generate the corresponding audio response tokens yourself (e.g., using [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) tokens in SLAM-Omni).

## Checkpoints
We reproduced the single-stage fine-tuning results of SLAM-Omni with a group size of **3**. The following checkpoints are available for download:
- [Single-Round Dialogue (English)](https://drive.google.com/drive/folders/1ZmM1h5ZTvS-piuN-msmctmZdi51GWLAu?usp=sharing): Trained on VoiceAssistant-400K.
- [Multi-Round Dialogue (English)](https://drive.google.com/drive/folders/1xBNrqR2LWC0uEjezjx4aUgdsbstisboS?usp=sharing): Trained on VoiceAssistant-400K and UltraChat-300K.
- [Multi-Round Dialogue (Chinese)](https://drive.google.com/drive/folders/1sExIp-UDdL37gb-mh9YlhuDIib0-wUVP?usp=sharing): Trained on Belle_1.4M.


## Training

You can pre-train the S2S model using TTS or ASR tasks with our provided scripts, though we recommend proceeding directly to fine-tuning. Alternatively, you may directly train a TTS or ASR model under the SLAM-Omni framework. For detailed instructions, refer to the [pre-training README](./scripts/pretrain/README.md).

### Fine-tuning
We provide two primary fine-tuning options for **SLAM-Omni** modeling:
```bash
# Fine-tune with grouping strategy (Recommended)
bash ./examples/s2s/scripts/finetune/finetune_s2s_group.sh

# Fine-tune without grouping
bash ./examples/s2s/scripts/finetune/finetune_s2s.sh
```

We also include scripts for reproducing [Mini-Omni](https://github.com/gpt-omni/mini-omni). Note that this requires the original [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) dataset for training:
```bash
bash ./examples/s2s/scripts/finetune/mini-omni/finetune_s2s.sh
```

#### Note💫
Our framework theoretically supports **all codec-based spoken dialogue model training**. Simply re-synthesize the target tokens (e.g., CosyVoice2 tokens) during training for compatibility.

## Inference
We provide scripts for both **online** and **batch** inference. You can use the trained model or the provided checkpoints for inference. For detailed guidance, refer to [inference README](./scripts/inference/README.md).



### Online Inference
Run the following commands for real-time inference:

```bash
# Multi-turn (Recommended)
bash ./examples/s2s/scripts/inference/inference_s2s_online_multi-round.sh

# Single-turn
bash ./examples/s2s/scripts/inference/inference_s2s_online.sh
```

For Mini-Omni modeling, use the following commands:
```bash
# Single-turn non-streaming
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online.sh

# Single-turn streaming
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_online_stream.sh
```


### Batch Inference

For batch inference, ensure the data format matches the training format (**Parquet** or **JSONL**). Use the following commands:

```bash
# SLAM-Omni framework
bash ./examples/s2s/scripts/inference/inference_s2s_batch.sh

# Mini-Omni framework
bash ./examples/s2s/scripts/inference/mini-omni/inference_s2s_batch.sh
```

## TODO
- [ ] Add evaluation scripts.
- [ ] Add streaming inference scripts for SLAM-Omni.


<!-- ## Gradio Demo -->

## Citation
SLAM-Omni:
```bibtex
@article{chen2024slam,
  title={SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training},
  author={Chen, Wenxi and Ma, Ziyang and Yan, Ruiqi and Liang, Yuzhe and Li, Xiquan and Xu, Ruiyang and Niu, Zhikang and Zhu, Yanqiao and Yang, Yifan and Liu, Zhanxun and others},
  journal={arXiv preprint arXiv:2412.15649},
  year={2024}
}
```
Mini-Omni:
```bibtex
@article{xie2024mini,
  title={Mini-omni: Language models can hear, talk while thinking in streaming},
  author={Xie, Zhifei and Wu, Changqiao},
  journal={arXiv preprint arXiv:2408.16725},
  year={2024}
}
```

## Acknowledgement
- We borrow some code from [Mini-Omni](https://github.com/gpt-omni/mini-omni) for SNAC-based modeling.
- We borrow some code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for the vocoder.


## License
Our code is released under MIT License. The Chinese dialogue model is licensed under GPL-3.0 due to its use of Belle data and is intended for research purposes only.