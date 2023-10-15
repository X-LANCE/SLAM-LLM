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


from llama_recipes.datasets import utils

class AudioDataset(Dataset):
    def __init__(
            self, 
            dataset_config,
            tokenizer=None,
            split='train'
        ):
        super().__init__()
        self.dataset_config = dataset_config
        self.max_words = dataset_config.max_words
        self.target_length = dataset_config.target_length # default = 1024
        self.tokenizer = tokenizer
        self.data = torch.randn(100, 160000)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        
        feats = self._wav2feat(data)
        question = "What is the answer to life, the universe and everything?"
        answer = "I don't know."

        format_instruction = question
        input1 = utils.format_prompt(format_instruction, None)
        input2 = input1 + answer + '</s>'

        input1 = self.tokenizer(input1)
        input2 = self.tokenizer(input2)
    
        labels = input2['input_ids'][len(input1['input_ids']):]

        return {
            'input_ids': input2['input_ids'],
            'attention_mask': input2['attention_mask'],
            'labels': labels,
            'inputs_embeds': feats,
        }

    def _wav2feat(self, data):
        wav = data.reshape(1, -1)

        feats = torchaudio.compliance.kaldi.fbank( # 25ms and 10ms
            wav, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
        )
        n_frames = feats.shape[0]

        p = self.target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            feats = m(feats)
        elif p < 0:
            feats = feats[0:self.target_length, :]
        
        return feats.unsqueeze(0) # channels, frames, dim


    def pad(self, sequence, max_length, padding_idx=0):
        if len(sequence) < max_length:
            sequence = sequence + [padding_idx] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence

    def collator(self, samples):
        assert samples is not None
        max_input_length = max([len(s['input_ids']) for s in samples])
        max_input_length = min(max_input_length, self.max_words)
        input_ids = torch.tensor([self.pad(s['input_ids'], max_input_length) for s in samples])

        max_attention_length = max([len(s['attention_mask']) for s in samples])
        max_attention_length = min(max_attention_length, self.max_words)
        attention_mask = torch.tensor([self.pad(s['attention_mask'], max_attention_length) for s in samples])

        max_target_length = max([len(s['labels']) for s in samples])
        max_target_length = min(max_target_length, self.max_words)
        labels = torch.tensor([self.pad(s['labels'], max_target_length) for s in samples])

        inputs_embeds = torch.stack([s['inputs_embeds'] for s in samples])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'inputs_embeds': inputs_embeds,
        }



def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = AudioDataset(dataset_config, tokenizer, split)

    return dataset
