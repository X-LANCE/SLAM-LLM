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


prompt = (
        f"USER: <Speech><SpeechHere></Speech> {prompt}\n ASSISTANT:"
    )

def apply_prompt_template(prompt, answer):
    return prompt.format(prompt=prompt)

class AudioDataset(Dataset):
    def __init__(
            self, 
            dataset_config,
            tokenizer=None,
            split='train'
        ):
        super().__init__()
        self.data = torch.randn(100, 160000)

        self.dataset_config = dataset_config
        self.max_words = dataset_config.max_words
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100 # The default setting in CrossEntropyLoss

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        
        # load audio and pad/trim it to fit 30 seconds
        audio_raw = whisper.load_audio(item)
        audio_raw = whisper.pad_or_trim(audio_raw)
        # make log-Mel spectrogram
        audio_feats = whisper.log_mel_spectrogram(audio_raw)

        prompt="""
        Please provide an emotional response based on the emotional speech you hear.
        Remember to format your answer as follows: <|EMOTION|><|DEGREE|><|REPLY|>.
        <|EMOTION|> is a standalone adjective. 
        <|DEGREE|> is an number ranging from 0 to 2. 
        <|REPLY|> is a reply based on a the speech. 
        """
        answer="""
        <|happy|><2><|The moon looks so beautiful tonight.|>
        """

        prompt = apply_prompt_template(prompt=prompt)
        example = prompt + answer
        prompt_ids = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )

        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = self.IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
            'audio_mel': audio_mel
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
        input_ids = torch.stack([s['input_ids'] for s in samples])
        labels = torch.stack([s['labels'] for s in samples])
        attention_mask = torch.stack([s['attention_mask'] for s in samples])
        
        audio_feats = torch.stack([s['audio_feats'] for s in samples])
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'audio_feats': audio_feats,
        }


def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = AudioDataset(dataset_config, tokenizer, split)

    return dataset
