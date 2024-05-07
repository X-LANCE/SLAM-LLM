# VSR_LRS3

## Performance and checkpoints
We only train the linear projector in this recipe.
Encoder | Projector | LLM | test 
|---|---|---|---|
[AV-HuBERT Large + Self-Training](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt) | [Linear]()(~18.88M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 29.47 


## Data preparation
Follow the steps in [preparation](https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation) of av_hubert to pre-process LRS3 dataset

## Environment
Use the specific fairseq version of [av_hubert](https://github.com/facebookresearch/av_hubert), which is compatible with hydra-core versions below 1.0.7 and omegaconf versions below 2.0.6.


## Decode with checkpoints
```
bash decode_avhubert_vo_vicuna_7b.sh
```
Modify the path including `speech_encoder_path`, `llm_path`, `output_dir`, `ckpt_path` and `decode_log` in the script when you run the shell script. 

## Train a new model

### Use the visual part of AV-HuBERT Large as the encoder
```
bash finetune_avhubert_vo_vicuna_7b.sh
```


