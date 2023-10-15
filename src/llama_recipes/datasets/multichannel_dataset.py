import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


from llama_recipes.datasets import kaldi, utils

class MultichannelDataset(Dataset):
    _ext_IR = ".npy" 
    _ext_audio = ".flac"
    _sample_rate = 16000

    #TODO: adjust the mean and std
    fbank_norm_mean = -4.2677393 
    fbank_norm_std = 4.5689974
    
    def __init__(
            self, 
            dataset_config,
            tokenizer=None,
            split='train'
        ):
        super().__init__()
        qa_path = osp.join(dataset_config.qa_path, split + '.json')
        self.ir_root = dataset_config.ir_root
        self.audioset_root = dataset_config.audioset_root
        self.channel_type = dataset_config.channel_type
        self.max_words = dataset_config.max_words
        self.target_length = dataset_config.target_length
        self.reverb_dataset = dataset_config.reverb_dataset

        print("load AQA dataset...")
        self.data = json.load(open(qa_path))['data']
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        qa_item = self.data[index]

        audio_path = osp.join(self.audioset_root, qa_item['audio_id'] + self._ext_audio)

        house_id, prefix  = qa_item['reverberation'].split('-', maxsplit=1)
        ir_path = osp.join(self.ir_root, self.reverb_dataset, self.channel_type, house_id, prefix + self._ext_IR)
        
        feats = self._wav2concate((ir_path, audio_path))
            
        question = qa_item['question']
        answer = qa_item['answer']

        format_instruction = question
        input1 = utils.format_prompt(format_instruction, None)
        input2 = input1 + answer + '</s>'

        #! TODO here
        input1 = self.tokenizer(input1)
        input2 = self.tokenizer(input2)
        
        # labels = copy.deepcopy(input2)
        # labels[:len(input1)] = -1
        # input2_mask = input2.ge(0)
        # label_mask = labels.ge(0)
        # input2[~input2_mask] = 0
        # labels[~label_mask] = 0
        # input2_mask = input2_mask.float()
        # label_mask = label_mask.float()
        labels = input2['input_ids'][len(input1['input_ids']):]

        return {
            'input_ids': input2['input_ids'],
            'attention_mask': input2['attention_mask'],
            'labels': labels,
            'inputs_embeds': feats,
        }

    def _wav2concate(self, input1):
        ir_path, audio_path = input1

        # rawwave, sr = torchaudio.load(audio_path) # sr = 48000
        rawwave, sr = sf.read(audio_path)
        
        if len(rawwave.shape) > 1:
            rawwave = rawwave[:, 0]
        
        if sr != 48000:
            rawwave = signal.resample_poly(rawwave, 48000, sr)

        rawwave = rawwave.reshape(1, -1)
        if ir_path is not None:
            ir = np.load(ir_path)
            conv_wave = signal.fftconvolve(rawwave, ir, mode='full')
            waveform = signal.resample_poly(conv_wave.T, 16000, 48000).T
        else:
            waveform = signal.resample_poly(rawwave.T, 16000, 48000).T
        
        waveform = waveform - waveform.mean(axis=1, keepdims=True)

        waveform = torch.from_numpy(waveform).float()

        stacked_feats = []
        for chans in range(waveform.shape[0]):
            wav = waveform[chans, :].unsqueeze(0)

            fbank = kaldi.fbank( # 25ms and 10ms
                wav, htk_compat=True, sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10
            )
            target_length = self.target_length

            # concatened = torch.cat((magnitude, phase), dim=-1) # seq_len * 257(2) ==> frames * 514
            # n_frames = concatened.shape[0]
            n_frames = fbank.shape[0]

            p = target_length - n_frames
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]
            stacked_feats.append(fbank)
        
        stacked_feats = torch.stack(stacked_feats)
        return stacked_feats


    def collator(self, samples):
        assert samples is not None
        max_input_length = max([len(s['input_ids']) for s in samples])
        max_input_length = min(max_input_length, self.max_words)
        input_ids = torch.tensor([
            s['input_ids'] + [0] * (max_input_length - len(s['input_ids']))
            for s in samples
        ])

        max_attention_length = max([len(s['attention_mask']) for s in samples])
        max_attention_length = min(max_attention_length, self.max_words)
        attention_mask = torch.tensor([
            s['attention_mask'] + [0] * (max_attention_length - len(s['attention_mask']))
            for s in samples
        ])

        max_target_length = max([len(s['labels']) for s in samples])
        max_target_length = min(max_target_length, self.max_words)
        labels = torch.tensor([
            s['labels'] + [0] * (max_target_length - len(s['labels'])) 
            for s in samples
        ])

        inputs_embeds = torch.stack([s['inputs_embeds'] for s in samples])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'inputs_embeds': inputs_embeds,
        }



def get_multichannel_dataset(dataset_config, tokenizer, split):
    dataset = MultichannelDataset(dataset_config, tokenizer, split)

    return dataset
