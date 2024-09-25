FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

USER root

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/cwx-worst-one/SLAM-LLM.git"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim ninja-build\
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && pip install --upgrade pip==24.0.0

RUN pip install --no-cache-dir packaging editdistance gpustat wandb einops debugpy tqdm soundfile matplotlib scipy sentencepiece pandas \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace

RUN git clone https://github.com/huggingface/transformers.git \
    && cd transformers \
    && git checkout tags/v4.35.2 \
    && pip install --no-cache-dir -e .

RUN git clone https://github.com/huggingface/peft.git \
    && cd peft \
    && git checkout tags/v0.6.0 \
    && pip install --no-cache-dir -e .

RUN git clone https://github.com/pytorch/fairseq \
    && cd fairseq \
    && pip install --no-cache-dir --editable ./

RUN git clone https://github.com/cwx-worst-one/SLAM-LLM.git \
    && cd SLAM-LLM \
    && pip install --no-cache-dir -e .

ENV SHELL=/bin/bash

WORKDIR /workspace/SLAM-LLM