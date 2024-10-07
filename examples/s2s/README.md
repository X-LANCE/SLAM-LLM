# s2s-dev

## Data Preparation
You can download the [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) dataset from the Hugging Face Hub using the following command:
```python
from datasets import load_dataset
ds = load_dataset("gpt-omni/VoiceAssistant-400K")
```

You can save the dataset to disk using the following command:
```python
save_path = "/path/to/save/directory"
ds.save_to_disk(save_path)
```

If you save the dataset to disk, you can load it using the following command:
```python
from datasets import load_from_disk
ds = load_from_disk(save_path)
```

## Training
To pre-train the model with TTS task using the VoiceAssistant-400K dataset, you can run the following command:
```bash
bash ./examples/s2s/scripts/pretrain_tts.sh
```

To fine-tune the model with S2S task using the VoiceAssistant-400K dataset, you can run the following command:
```bash
bash ./examples/s2s/scripts/finetune_s2s.sh
```


## Inference
To generate the text and speech using the pre-trained model given the speech input, you can run the following command:
```bash
bash ./examples/s2s/scripts/inference_s2s.sh
```