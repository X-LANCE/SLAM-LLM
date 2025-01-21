# SLAM-AAC

SLAM-AAC is a LLM-based framework for Automated Audio Captioning (AAC) task. Inspired by techniques in machine translation and ASR, the model enhances audio captioning by incorporating **paraphrasing augmentation** and a plug-and-play **CLAP-Refine** strategy. For more details, please refer to the [paper](https://arxiv.org/abs/2410.09503).

## Model Architecture
SLAM-AAC uses **EAT** as the audio encoder and **Vicuna-7B** as the LLM decoder. During training, only the Linear Projector and LoRA modules are trainable. For inference, multiple candidates are generated using different beam sizes, which are then refined using the CLAP-Refine strategy.

![](./docs/model.png)

## Performance and checkpoints
Pre-trained and fine-tuned checkpoints for the **Clotho** and **AudioCaps** datasets are available. These checkpoints include the Linear Projector and LoRA modules. Ensure proper setup of the corresponding environments (e.g., [EAT](https://github.com/cwx-worst-one/EAT)) before use.


### Pre-training
SLAM-AAC was pre-trained on AudioCaps, Clotho, WavCaps, and MACS datasets. For more information on these datasets, you can refer to [this repository](https://github.com/Labbeti/aac-datasets). Additionally, the Clotho dataset was augmented using a back-translation-based paraphrasing technique.  
Audio Encoder | LLM | Checkpoint | Pre-training Dataset|
|:---:|:---:|:---:|:---:|
[EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) |[vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [link](https://drive.google.com/drive/folders/10kOjB112AeGYA_0mIUr8f1-i5rSg08_O?usp=sharing) | AudioCaps, Clotho, WavCaps, MACS |

### Fine-tuning
We fine-tuned the pre-trained model on the Clotho and AudioCaps datasets, respectively. The final evaluation was conducted using audio captions generated with the CLAP-Refine decoding strategy.
Dataset | Audio Encoder | LLM | Checkpoint | METEOR | CIDEr | SPICE | SPIDEr | SPIDEr-FL | FENSE
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Clotho | [EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [link](https://drive.google.com/drive/folders/1QX7CM9YAddPi02_NRChI5mzsNmBBtA63?usp=sharing) | 19.7 | 51.5 | 14.8 |33.2 | 33.0 | 54.0 |
| AudioCaps | [EAT-base (fine-tuned)](https://drive.google.com/file/d/1aCYiQmoZv_Gh1FxnR-CCWpNAp6DIJzn6/view?usp=sharing) | [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [link](https://drive.google.com/drive/folders/1GhFPiSVmBE9BvBhYWCEqkFuH-avKl-4g?usp=sharing) | 26.8 | 84.1 | 19.4 | 51.8 | 51.5 | 66.8 |


## Data preparation
Ensure your `jsonl` data follows this format:
```json
{"key": "Y7fmOlUlwoNg_1", "source": "/root/data/AudioCaps/waveforms/test/Y7fmOlUlwoNg.wav", "target": "Constant rattling noise and sharp vibrations"}
{"key": "Y6BJ455B1aAs_1", "source": "/root/data/AudioCaps/waveforms/test/Y6BJ455B1aAs.wav", "target": "A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle"}
```
In addition, you can refer to the [manifest](https://drive.google.com/drive/folders/1NJinoWg3yXKSPm-pRrhqKLvCD9dtDuDG?usp=sharing) file we've provided, which includes the Clotho dataset enhanced with **paraphrasing augmentation** as bonus.

## Model Training
To pre-train the SLAM-AAC model with pre-training data, you can run the following command:
```bash
# Pre-train the model
bash scripts/pretrain.sh
```

You can fine-tune the model on the AudioCaps or Clotho datasets using the [provided checkpoint](https://drive.google.com/drive/folders/10kOjB112AeGYA_0mIUr8f1-i5rSg08_O?usp=sharing) or your own pre-trained model by running the following commands:

```bash
# Fine-tune on AudioCaps
bash scripts/finetune_audiocaps.sh

# Fine-tune on Clotho
bash scripts/finetune_clotho.sh
```

You can also fine-tune the model without loading any pre-trained weights, though this may result in reduced performance.


### Note
- In the current version of SLAM-LLM, the `peft_ckpt` parameter is no longer required. However, if you are using the checkpoint provided by us, which was trained with an earlier version, please keep the `peft_ckpt` parameter in your configuration to ensure compatibility.
- Due to differences in dependency versions, there may be slight variations in the performance of the SLAM-AAC model.

## Inference
To perform inference with the trained models with beam search:
```bash
# Inference on AudioCaps (Beam Search)
bash scripts/inference_audiocaps_bs.sh

# Inference on Clotho (Beam Search)
bash scripts/inference_clotho_bs.sh
```

To generate better captions, use the CLAP-Refine strategy with multiple beam search decoding. This method leverages our pre-trained [CLAP](https://drive.google.com/drive/folders/1X4NYE08N-kbOy6s_Itb0wBR_3X8oZF56?usp=sharing) model. Though it takes more time, it ensures higher-quality results. Use the following commands to apply it:


```bash
# Inference on AudioCaps (CLAP-Refine)
bash scripts/inference_audiocaps_CLAP_Refine.sh

# Inference on Clotho (CLAP-Refine)
bash scripts/inference_clotho_CLAP_Refine.sh
```

If you already have the generated candidates and want to directly refine them using the CLAP-Refine strategy, you can run the following command:
```bash
bash scripts/clap_refine.sh
```

##  Citation
If you find SLAM-AAC useful, please cite the following paper:
```
@article{chen2024slam,
  title={SLAM-AAC: Enhancing Audio Captioning with Paraphrasing Augmentation and CLAP-Refine through LLMs},
  author={Chen, Wenxi and Ma, Ziyang and Li, Xiquan and Xu, Xuenan and Liang, Yuzhe and Zheng, Zhisheng and Yu, Kai and Chen, Xie},
  journal={arXiv preprint arXiv:2410.09503},
  year={2024}
}
```
