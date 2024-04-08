FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# python 3.10

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        vim \
    	libssl-dev \
        autoconf \
        automake \
        bzip2 \
        ca-certificates \
        ffmpeg \
        g++ \
        gfortran \
        git \
        libtool \
        make \
        patch \
        sox \
        subversion \
        unzip \
        valgrind \
        wget \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY ./ /SLAM-LLM
RUN cd /SLAM-LLM && \
    pip install -e .

RUN pip install git+https://github.com/openai/whisper.git
RUN pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

ENV PYTHONPATH /SLAM-LLM/src:$PYTHONPATH

WORKDIR /SLAM-LLM