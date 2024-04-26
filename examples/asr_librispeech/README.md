# ASR_Librispeech

## Performance and checkpoints
We only train the linear projector in this recipe.
Encoder | Projector | LLM | test-clean | test-other
|---|---|---|---|---
[WavLM-large](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view) | [Linear]()(~18.88M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 2.28 | 4.78


## Data preparation
You need to prepare the data jsonl in this format.
```
{"key": "1001-134707-0000_ASR", "source": "/data/open_data/librispeech_audio/audio/librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only."}
...
{"key": "1001-134707-0000_ASR", "source": "/data/open_data/librispeech_audio/audio/librispeech_1001-134707-0000.wav", "target": "1 little recks the laborer. How near his work is holding him to God, The loving laborer through space and time, after all, not to create, only or found only."}
```

## Decode with checkpoints
```
bash decode_wavlm_large_linear_vicuna_7b.sh
```
Modify the path including `speech_encoder_path`, `llm_path`, `output_dir`, `ckpt_path`, `val_data_path` and `decode_log` in the script when you run the shell script. 

## Train a new model

### Use whisper as the encoder
```
bash finetune_whisper_large_linear_vicuna_7b.sh
```
Whisper takes mel as input. Pay attention to the key `dataset_config.mel_size` for different version of the whisper model family. 

### Use self-supervised model(such as WavLM) as the encoder
```
bash finetune_wavlm_large_linear_vicuna_7b.sh
```
WavLM takes raw wavform as input. Pay attention to the key `dataset_config.normalize` and `model_config.normalize` for different version of the SSL models for different SSL models are different in these keys. 
