# AAC_Audiocaps

## Performance and checkpoints
We use [EAT](https://github.com/cwx-worst-one/EAT) as the audio encoder in this repo. Be sure to set up the corresponding environments based on the instructions provided in each repository. Here are checkpoints and performance for training only the linear layer and training the linear layer with LLM tuning via LoRA. We train and evaluate the performance of SLAM-AAC on the [Audiocaps](https://github.com/cdjkim/audiocaps) dataset.
Audio Encoder | Projector | LLM | PEFT | METEOR | CIDEr | SPICE | SPIDEr
|---|---|---|---|---|---|---|---|
[EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | [Linear](https://drive.google.com/file/d/1xyhgx8cUKSIKpYgPlEWjHL-jLgSnhfGJ/view?usp=sharing)(~16.26M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | x | 0.2508 | 0.7532 | **0.1853** |0.4692
[EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | [Linear](https://drive.google.com/drive/folders/1_Pl3DLSbu6i2KyNCvzf74HAWXLroBgN3?usp=sharing)(~16.26M) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [LoRA](https://drive.google.com/drive/folders/1_Pl3DLSbu6i2KyNCvzf74HAWXLroBgN3?usp=sharing)(~4.19M) | **0.2606** | **0.7922** | 0.1852 | **0.4887**


## Data Preparation
Prepare your `jsonl` data in the following format:
```json
{"key": "Y7fmOlUlwoNg_1", "source": "/root/data/AudioCaps/waveforms/test/Y7fmOlUlwoNg.wav", "target": "Constant rattling noise and sharp vibrations"}
{"key": "Y6BJ455B1aAs_1", "source": "/root/data/AudioCaps/waveforms/test/Y6BJ455B1aAs.wav", "target": "A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle"}
```
Ensure your data aligns with this structure for consistent results.


## Model Training
To train the model, you could run the following command:
```bash
bash scripts/finetune_eat_audiocaps.sh
```
You could modify the variable including `audio_encoder_path`, `llm_path`, `output_dir`, `train_jsonl_path`, and `val_jsonl_path` in the script to fit your setup. For training only the linear layer (without using LoRA or other PEFT methods), you can set the following parameters: `use_peft=false` and `freeze_llm=true`.

## Inference
To perform inference with trained models, you could use this command:
```bash
bash scripts/inference_eat_audiocaps.sh
```
Ensure your environment is set up and data paths are correct to reproduce results. 
