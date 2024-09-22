# SLAM-AAC

SLAM-AAC is a LLM-based model for Automated Audio Captioning (AAC) task. Inspired by techniques in machine translation and ASR, the model enhances audio captioning by incorporating paraphrasing augmentation and a plug-and-play CLAP-Refine strategy. For more details, please refer to the [paper]().

## Model Architecture
SLAM-AAC uses EAT as the audio encoder and Vicuna-7B as the LLM decoder. During training, only the Linear Projector and LoRA modules are trainable. For inference, multiple candidates are generated using different beam sizes, which are then refined using the CLAP-Refine strategy.

![](./docs/model.png)

## Performance and checkpoints
We have released the pre-trained checkpoint of SLAM-AAC, as well as the fine-tuned checkpoints for the Clotho and AudioCaps datasets. The provided checkpoints include the model's Linear Projector and LoRA modules. Please note that when using each component, be sure to set up the corresponding environments according to the instructions provided in the respective repositories (e.g., for [EAT](https://github.com/cwx-worst-one/EAT)).

### Pre-training
SLAM-AAC was pre-trained on a combination of AudioCaps, Clotho, WavCaps, and MACS datasets. For more information on these datasets, you can refer to [this repository](https://github.com/Labbeti/aac-datasets). Additionally, the Clotho dataset was augmented using a back-translation-based paraphrasing technique.
Audio Encoder | LLM | Checkpoint | Pre-training Dataset|
|:---:|:---:|:---:|:---:|
[EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) |[vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |  | AudioCaps, Clotho, WavCaps, MACS |

### Fine-tuning
Dataset | Audio Encoder | LLM | Checkpoint | METEOR | CIDEr | SPICE | SPIDEr | SPIDEr-FL | FENSE
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Clotho | [EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | | 19.7 | 51.5 | 14.8 |33.2 | 33.0 | 54.0 |
| AudioCaps | [EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) |  | 26.8 | 84.1 | 19.4 | 51.8 | 51.5 | 66.8 |


## Data preparation
Prepare your `jsonl` data in the following format:
```json
{"key": "Y7fmOlUlwoNg_1", "source": "/root/data/AudioCaps/waveforms/test/Y7fmOlUlwoNg.wav", "target": "Constant rattling noise and sharp vibrations"}
{"key": "Y6BJ455B1aAs_1", "source": "/root/data/AudioCaps/waveforms/test/Y6BJ455B1aAs.wav", "target": "A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle"}
```
Ensure your data aligns with this structure for consistent results.

## Model Training
To train the model, you could run the following command:
```bash
bash scripts/finetune_audiocaps.sh
```
You could modify the variable including `audio_encoder_path`, `llm_path`, `output_dir`, `train_jsonl_path` and `val_jsonl_path` in the script to fit your setup. For training only the linear layer (without using LoRA or other PEFT methods), you can set the following parameters: `use_peft=false` and `freeze_llm=true`.

## Inference
To perform inference with trained models, you could use this command:
```bash
bash scripts/inference_audiocaps.sh
```
Ensure your environment is set up and data paths are correct to reproduce results. 


<!-- ##  Citation
You can refer to the paper for more results. 
```

``` -->