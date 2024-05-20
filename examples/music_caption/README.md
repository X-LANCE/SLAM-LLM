# Music Caption

## Performance and checkpoints
Here is a recipe for music captioning, using MusicFM as encoder. We only train the linear projector. For more about MusicFM and its checkpoints, please refer to [this repository](https://github.com/minzwon/musicfm).

The following results are obtained by training on the LP-MusicCaps-MC training set and evaluating on the LP-MusicCaps-MC test set.
Encoder | Projector | LLM | BLEU-1 | METEOR | SPICE | SPIDER 
|---|---|---|---|---|---|---
[MusicFM(pretrained with MSD)](https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt) | [Linear](https://drive.google.com/file/d/1-9pob6QvJRoq5Dy-LZbiDfF6Q7QRO8Au/view?usp=sharing)(~18.88M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 25.6 | 10.0 | 8.7 | 6.9


## Data preparation
You need to prepare the data jsonl in this format. Note that you may need to pre-extract the sample rate and duration of audio files for better loading efficiency.
```
{"key": "[-0Gj8-vB1q4]-[30-40]", "source": "path/to/MusicCaps/wav/[-0Gj8-vB1q4]-[30-40].wav", "target": "The low quality recording features a ballad song that contains sustained strings, mellow piano melody and soft female vocal singing over it. It sounds sad and soulful, like something you would hear at Sunday services.", "duration": 10.0, "sample_rate": 48000}
...
{"key": "[-0vPFx-wRRI]-[30-40]", "source": "path/to/MusicCaps/wav/[-0vPFx-wRRI]-[30-40].wav", "target": "a male voice is singing a melody with changing tempos while snipping his fingers rhythmically. The recording sounds like it has been recorded in an empty room. This song may be playing, practicing snipping and singing along.", "duration": 10.0, "sample_rate": 48000}
```

## Decode with checkpoints
```
bash decode_musicfm_linear_vicuna_7b_10s.sh
```
Modify the path including `music_encoder_path`, `music_encoder_stat_path`, `music_encoder_config_path`(if specified), `ckpt_path`, `val_data_path` and `decode_log` in the script when you run the shell script. 

## Train a new model

### Use MusicFM as encoder for music modality.
```
finetune_musicfm_linear_vicuna_7b_10s.sh
```
