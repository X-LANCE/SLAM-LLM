## Pre-training Instructions

This directory contains scripts for S2S model pre-training using either TTS or ASR tasks. 

### Data Preparation
Before starting pre-training, ensure your data is prepared in either Parquet or JSONL format:
- **Parquet**: Directly load from the Hugging Face Hub (e.g., VoiceAssistant-400K).
- **JSONL**: Follow the main README for data structure examples.

### Tasks
- **TTS**: Learn to generate target speech from given text. (we support zero-shot TTS!)
  ```bash
  bash ./examples/s2s/scripts/pretrain/pretrain_tts.sh
  ```

- **ASR**: Learn to generate text transcripts from given speech.
  ```bash
  bash ./examples/s2s/scripts/pretrain/pretrain_asr.sh
  ```

### Inference

You can use the following command to perform online inference with the pre-trained TTS model:
```bash
# TTS
bash ./examples/s2s/scripts/inference/inference_tts_online.sh

# ASR
bash ./examples/s2s/scripts/inference/inference_asr_online.sh
```