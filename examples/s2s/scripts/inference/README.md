# NOTE 
- Our current inference only supports **single** input and does not support batch input.
- We provide two inference mode: `text only` and `text & speech`. You can set the `decode_text_only` parameter in the inference script to choose the mode you want to use.
- If you use the CosyVoice codec during inference, you can freely choose the output voice tone by setting the `audio_prompt_path`. We also provide some optional voices in the `prompt` directory. If not specified, the default voice will be used.
