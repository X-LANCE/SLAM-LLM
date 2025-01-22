## Pre-training Guide

This directory contains scripts for pre-training the S2S model using either TTS or ASR tasks. However, based on experimental findings and results from the paper, we recommend skipping the pre-training phase and proceeding directly to S2S fine-tuning for optimal performance.

### Data Preparation
Ensure your data is in the required format before starting pre-training:
- **Parquet**: Load datasets directly from the Hugging Face Hub (e.g., [VoiceAssistant-400K](https://huggingface.co/datasets/worstchan/VoiceAssistant-400K-SLAM-Omni) ).
- **JSONL**: Refer to the main README for examples of the required data structure.

### Tasks
- **TTS**: Learn to generate target speech from given text. (Supports zero-shot TTS!)
  ```bash
  bash ./examples/s2s/scripts/pretrain/pretrain_tts.sh
  ```

- **ASR**: Learn to generate text transcripts from given speech.
  ```bash
  bash ./examples/s2s/scripts/pretrain/pretrain_asr.sh
  ```

### Inference

For testing, you can perform online inference using the pre-trained models:
- TTS
  ```bash
  bash ./examples/s2s/scripts/inference/inference_tts_online.sh
  ```
- ASR
  ```bash
  bash ./examples/s2s/scripts/inference/inference_asr_online.sh
  ```