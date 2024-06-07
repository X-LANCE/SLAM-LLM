# MALA-ASR_SLIDESPEECH

## Performance and checkpoints
We only train the linear projector in this recipe.
Encoder | Projector | LLM | dev | test
|---|---|---|---|---|
[WavLM-large](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view) | [Linear](https://drive.google.com/file/d/1hYS5UI3W0WVOZRVbqWxDUWIFMO9VgzHk/view?usp=drive_link)(~15.74M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 8.91 | 9.14 


## Data preparation
Refer to official [SLIDESPEECH CORPUS](https://slidespeech.github.io/)

## Decode with checkpoints
```
bash decode_MaLa-ASR_withkeywords_L95.sh
```
Modify the path including `speech_encoder_path`, `llm_path`, `output_dir`, `ckpt_path` and `decode_log` in the script when you run the shell script. 

## Train a new model

### Use self-supervised model(such as WavLM) as the encoder
```
bash finetune_MaLa-ASR_withkeywords_L95.sh
```