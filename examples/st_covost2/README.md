# ST_covost2


## Model Stracture
<img src="image/framework.jpg" alt="示例图片" style="width:75%;">


## Multitask 
<img src="image/prompt.png" alt="示例图片" style="width:50%;">



## Download Model 
We only train the q-former projector in this recipe.
Encoder | Projector | LLM 
|---|---|---
[whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [q-former](https://huggingface.co/yxdu/cotst) | [Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) 
```
git lfs clone https://huggingface.co/openai/whisper-large-v3
git lfs clone https://huggingface.co/yxdu/cotst
git lfs clone https://huggingface.co/Qwen/Qwen2-7B
```


## Data 
You need to download this dataset.
```
(https://github.com/facebookresearch/covost)
```



## Data preparation
You need to prepare the data jsonl in this format.  
You can find the test jsonl in "test_st.jsonl"
```
{"audio": "/userhome/speech/data/common/4/en/clips/common_voice_en_699711.mp3", "prompt": "<|en|>", "gt": "\"She'll be all right.\"", "source": "covost_en"}
{"audio": "/userhome/speech/data/common/4/en/clips/common_voice_en_699711.mp3", "prompt": "<|de|>", "gt": "\"She'll be all right.\"<|de|>Sie wird schon in Ordnung sein.", "source": "covost_ende"}
{"audio": "/userhome/speech/data/common/4/en/clips/common_voice_en_699711.mp3", "prompt": "<|ja|>", "gt": "\"She'll be all right.\"<|ja|>彼女は大丈夫だろう。", "source": "covost_enja"}
{"audio": "/userhome/speech/data/common/4/en/clips/common_voice_en_699711.mp3", "prompt": "<|zh|>", "gt": "\"She'll be all right.\"<|zh|>她会没事的。", "source": "covost_enzh"}
{"audio": "/userhome/speech/data/common/4/en/clips/common_voice_en_699711.mp3", "prompt": "\"She'll be all right.\"<|de|>", "gt": "\"She'll be all right.\"<|de|>Sie wird schon in Ordnung sein.", "source": "covost_enende"}
{"audio": "/userhome/speech/data/common/4/en/clips/common_voice_en_699711.mp3", "prompt": "\"She'll be all right.\"<|ja|>", "gt": "\"She'll be all right.\"<|ja|>彼女は大丈夫だろう。", "source": "covost_enenja"}
{"audio": "/userhome/speech/data/common/4/en/clips/common_voice_en_699711.mp3", "prompt": "\"She'll be all right.\"<|zh|>", "gt": "\"She'll be all right.\"<|zh|>她会没事的。", "source": "covost_enenzh"}
```
## Train Stage
Here, we have designed a three-step training process, where each training session uses the checkpoint obtained from the previous training session.
```
#In this step, we perform ASR pretraining to acquire speech recognition capabilities.
bash asr_pretrain.sh

#In this phase, we conduct multimodal machine translation training to enhance the final performance.
bash mmt.sh

#monolingual SRT training and multitask training.
bash srt.sh
bash zsrt.sh
```


## Infer Stage
You can try our pre-trained model.

```
bash infer_enzh.sh
```

##  Citation
You can refer to the paper for more results. 
```
@article{du2024cot,
  title={CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Thought},
  author={Yexing Du, Ziyang Ma, Yifan Yang, Keqi Deng, Xie Chen, Bo Yang, Yang Xiang, Ming Liu, Bing Qin},
  journal={arXiv preprint arXiv:2409.19510},
  year={2024}
}
```