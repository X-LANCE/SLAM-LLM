# AAC_Audiocaps

## Performance and checkpoints
We only train the linear projector in this recipe. We use [EAT](https://github.com/cwx-worst-one/EAT) and [BEATs](https://github.com/microsoft/unilm/tree/master/beats) as the main audio encoder for SLAM-AAC. Be sure to set up the corresponding environments based on the instructions provided in each repository.
Audio Encoder | Projector | LLM | SPIDEr
|---|---|---|---|
[EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | Linear(~18.88M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | 0.4692


## Data preparation
Prepare your `jsonl` data in the following format:
```json
{"key": "Y7fmOlUlwoNg_1", "prompt": "<AAC>", "source": "/root/data/AudioCaps/waveforms/test/Y7fmOlUlwoNg.wav", "target": "Constant rattling noise and sharp vibrations", "target_len": 6, "source_len": 6, "text-type": "Transcribe", "audio_language": "english", "text_language": "english", "task-type": "<AAC>"}
{"key": "Y6BJ455B1aAs_1", "prompt": "<AAC>", "source": "/root/data/AudioCaps/waveforms/test/Y6BJ455B1aAs.wav", "target": "A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle", "target_len": 18, "source_len": 18, "text-type": "Transcribe", "audio_language": "english", "text_language": "english", "task-type": "<AAC>"}
```
Ensure your data aligns with this structure for consistent results.


## Model Training
To train the model, you could run the following command:
```bash
bash scripts/finetune_eat_audiocaps.sh
```
You could modify the variable including `audio_encoder_path`, `llm_path`, `output_dir`, `train_jsonl_path` and `val_jsonl_path` in the script to fit your setup. 

## Inference
To perform inference with trained models, you could use this command:
```bash
bash scripts/inference_eat_audiocaps.sh
```
Ensure your environment is set up and data paths are correct for accurate results.