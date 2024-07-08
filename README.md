<div align="center">
    <h1>
    SLAM-LLM
    </h1>
    <p>
    <b>SLAM-LLM</b> is a deep learning toolkit that allows researchers and
developers to train custom multimodal large language model (MLLM), focusing on <b>S</b>peech, <b>L</b>anguage, <b>A</b>udio, <b>M</b>usic processing. We provide detailed recipes for training and high-performance checkpoints for inference. <br>
    </p>
    <p>
    <img src="docs/logo.jpg" alt="SLAM-LLM Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/Cuda-11.8+-orange" alt="version"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/PyTorch-2.01+-brightgreen" alt="python"></a>
    <a href="https://github.com/ddlBoJack/SLAM-LLM"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>

# Table of Contents
1. [News](#news)
2. [Installation](#installation)
3. [Uasge](#uasge)
    - [List of Recipes](#list-of-recipes)
    - [Configuration Priority](#configuration-priority)
4. [Features](#features)
5. [Acknowledge](#acknowledge)
6. [Citation](#citation)

# News
- [Update Jun. 12, 2024] Recipes for [MaLa-ASR](examples/mala_asr_slidespeech/README.md) has been supported. 
- **[CALL FOR EXAMPLE]** We sincerely invite developers and researchers to develop new applications, conduct academic research based on SLAM-LLM, and pull request your examples! We also acknowledge engineering PR (such as improving and speeding up multi-node training). 
- [Update May. 22, 2024] Please join [slack](https://join.slack.com/t/slam-llm/shared_invite/zt-2mc0pkhhs-5jjOi8Cwc8R1Xc8IQmykDA) or [WeChat group](./docs/Wechat.jpg). We will sync our updates and Q&A here. 
- [Update May. 21, 2024] Recipes for [Spatial Audio Understanding](examples/seld_spatialsoundqa/README.md) has been supported. 
- [Update May. 20, 2024] Recipes for [music caption (MC)](examples/mc_musiccaps/README.md) has been supported. 
- [Update May. 8, 2024] Recipes for [visual speech recognition (VSR)](examples/vsr_LRS3/README.md) has been supported. 
- [Update May. 4, 2024] Recipes for [zero-shot text-to-speech (TTS)](examples/vallex/README.md) has been supported. 
- [Update Apr. 28, 2024] Recipes for [automated audio captioning (AAC)](examples/aac_audiocaps/README.md) has been supported. 
- [Update Mar. 31, 2024] Recipes for [automatic speech recognition (ASR)](examples/asr_librispeech/README.md) has been supported. 

# Installation
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout tags/v4.35.2
pip install -e .
cd ..
git clone https://github.com/huggingface/peft.git
cd peft
git checkout tags/v0.6.0
pip install -e .
cd ..
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/ddlBoJack/SLAM-LLM.git
cd SLAM-LLM
pip install  -e .
```

For some examples, you may need to use `fairseq`, the command line is as follows:
```
# you need to install fairseq before SLAM-LLM
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
We also provide a docker image for convenience:
```shell
# build docker image
docker build -t slam-llm:latest .

# run docker image with gpu
docker run -it --gpus all --name slam --shm-size=256g slam-llm:latest /bin/bash
```
# Usage
## List of Recipes
We provide reference implementations of various LLM-based speech, audio, and music tasks: 
- **Speech Task**
    - [Automatic Speech Recognition (ASR)](examples/asr_librispeech/README.md)
    - [Text-to-Speech (TTS)](examples/vallex/README.md)
    - [Visual Speech Recognition (VSR)](examples/vsr_LRS3/README.md)
- **Audio Task**
    - [Automated Audio Captioning (AAC)](examples/aac_audiocaps/README.md)
    - [Spatial Audio Understanding](examples/seld_spatialsoundqa/README.md)
- **Music Task**
    - [Music Caption (MC)](examples/mc_musiccaps/README.md)

## Configuration Priority
We provide hierarchical configuration inheritance relationships as follows:
```
command-line (shell file) > Hydra configuration (yaml file) > dataclass configuration (Python file)
```

# Features
- Easily extend to new models and tasks.
- Detailed recipes for training and high-performance checkpoints for inference.
- Mixed precision training which trains faster with less GPU memory on NVIDIA tensor cores. 
- Multi-GPU training with data and model parallel, supporting [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [deepspeed](https://github.com/microsoft/DeepSpeed) (still need to be improved).  
- Flexible configuration based on [Hydra](https://github.com/facebookresearch/hydra) and [dataclass](https://docs.python.org/3/library/dataclasses.html) allowing a combination of code, command-line and file based configuration. 

# Acknowledge
- We borrow code from [Llama-Recipes](https://github.com/meta-llama/llama-recipes) for the training process. 
- We borrow code from [Fairseq](https://github.com/facebookresearch/fairseq) for deepspeed configuration. 
- We thank the contributors for providing diverse recipes. 

## Citation

```
@article{ma2024embarrassingly,
  title={An Embarrassingly Simple Approach for LLM with Strong ASR Capacity},
  author={Ma, Ziyang and Yang, Guanrou and Yang, Yifan and Gao, Zhifu and Wang, Jiaming and Du, Zhihao and Yu, Fan and Chen, Qian and Zheng, Siqi and Zhang, Shiliang and others},
  journal={arXiv preprint arXiv:2402.08846},
  year={2024}
}
```

