# AAC_Audiocaps

## Performance and checkpoints
We only train the linear projector in this recipe.
Audio Encoder | Projector | LLM | SPIDEr
|---|---|---|---|
[EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | Linear(~18.88M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 46.92


## Data preparation
You need to prepare the `jsonl` data in this format.
```json
{"key": "Santa Motor", "prompt": "<AAC>", "source": "/root/data/Clotho_v2/evaluation/Santa Motor.wav", "target": "A machine whines and squeals while rhythmically punching or stamping.", "target_len": 10, "source_len": 10, "text-type": "Transcribe", "audio_language": "english", "text_language": "english", "task-type": "<AAC>"}
{"key": "Radio Garble", "prompt": "<AAC>", "source": "/root/data/Clotho_v2/evaluation/Radio Garble.wav", "target": "A radio dispatcher and an officer are communicating over the radio.", "target_len": 11, "source_len": 11, "text-type": "Transcribe", "audio_language": "english", "text_language": "english", "task-type": "<AAC>"}
```

## Generate Audio Caption with Checkpoints
```bash
bash inference_eat_audiocaps.sh
```
Modify the path including `speech_encoder_path`, `llm_path`, `output_dir`, `ckpt_path`, `val_data_path` and `decode_log` in the script when you run the shell script. 

## Model Training
```bash
bash finetune_eat_audiocaps.sh
```
EAT takes mel as input. Pay attention to the key `dataset_config.mel_size` for different version of the whisper model family. 
