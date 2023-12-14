import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
import whisper
from llama_recipes.utils.compute_utils import calculate_output_length_1d


class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        # data_parallel_size = dist.get_world_size()
        data_parallel_size = 1
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt_template = "USER: {}\n ASSISTANT:"

        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)

        # # debug
        # if split == "train":
        #     self.data_list = contents[:80]
        # else:
        #     self.data_list = contents[80:100]

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
    
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        speech_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)
        
        speech_raw = whisper.load_audio(speech_path)
        speech_mel = whisper.log_mel_spectrogram(speech_raw).permute(1, 0)

        speech_length = (speech_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
        speech_length = speech_length // 5 # ad-hoc for 5x cov1d downsample
        speech_pseudo = torch.full((speech_length,), -1)

        prompt = """
        <|ASR|>
        """
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)

        example_ids = torch.cat((speech_pseudo, prompt_ids))  # [speech,prompt]
        example_mask = example_ids.ge(-1)  # [True,True]

        return {
            "input_ids": example_ids,
            "attention_mask": example_mask,
            'speech_mel': speech_mel,
            'speech_length': speech_length,
            'key': key,
            'target':target
        }

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                 for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                      for s in samples])
    
        speech_mel_max_length = max([s['speech_mel'].shape[0] for s in samples])
        speech_mel = torch.stack([self.pad(s['speech_mel'], speech_mel_max_length, 0)
                                  for s in samples])
    
        speech_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            speech_mask[line, :sample['speech_length']] = 1
        keys = [s['key'] for s in samples]
        targets = [s['target'] for s in samples]
    
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'speech_mel': speech_mel,
            'speech_mask': speech_mask,
            'keys': keys,
            'targets': targets
        }



def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
