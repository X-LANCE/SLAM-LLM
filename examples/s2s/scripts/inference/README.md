# Notes for Inference

- Our current inference pipeline supports **single** input only and does not support batch processing.

- We provide two inference modes: `text only` and `text & speech`. You can set the `decode_text_only` parameter in the inference script to choose your preferred mode.

- If using **CosyVoice** for decoding (as employed in **SLAM-Omni**), please take note of the following:
  - Download the corresponding CosyVoice-300M-SFT model from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) and set the `codec_decoder_path` parameter in your script to its location.
  - You can customize the output voice tone by specifying the `audio_prompt_path`. A selection of optional voices is provided in the `prompt` directory. If not specified, the default voice tone will be used.