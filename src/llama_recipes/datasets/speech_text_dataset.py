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
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100 # The default setting in CrossEntropyLoss
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "<|{}|><|{}|>"

        with open(dataset_config.data_path, 'r') as file:
            data = file.readlines()
        if split == "train":
            self.data = data[:60]
        else:
            self.data = data[60:]
        
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        dialog_name, dialog = item.split('\t', 1)
        dialog_list = eval(dialog)
        
        while True:
            sentence_id = random.randint(0, len(dialog_list)-2)
            if 'emotion' in dialog_list[sentence_id].keys() and 'emotion' in dialog_list[sentence_id+1].keys():
                if dialog_list[sentence_id]['emotion'] != 'xxx' and dialog_list[sentence_id+1]['emotion'] != 'xxx':
                    break 
        speech_raw = whisper.load_audio(dialog_list[sentence_id]['wav'])
        # speech_raw = whisper.pad_or_trim(speech_raw)
        speech_mel = whisper.log_mel_spectrogram(speech_raw).permute(1,0)

        prompt="""
        Please provide an emotional response based on the emotional speech you hear.
        Remember to format your answer as follows: <|EMOTION|><|REPLY|>.
        <|EMOTION|> is a standalone adjective. 
        <|REPLY|> is a reply based on a the speech. 
        """
        answer="""
        <|happy|><|The moon looks so beautiful tonight.|>
        """

        prompt = self.prompt_template.format(prompt)
        answer = self.answer_template.format(dialog_list[sentence_id+1]['emotion'], dialog_list[sentence_id+1]['trans'])

        prompt_ids = self.tokenizer.encode(prompt)  # FIX(GZF)
        prompt_length = len(prompt_ids)
        speech_length = (speech_mel.shape[0] + 1) // 2 # ad-hoc for whisper for 2x downsample from mel to feats
        speech_pseudo = torch.full((speech_length,),-1)
        
        example_ids = self.tokenizer.encode(answer) # FIX(GZF): [answer]
        example_ids = prompt_ids + example_ids
        example_ids.append(self.tokenizer.eos_token_id) # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64
        )
        example_ids = torch.cat((speech_pseudo, example_ids)) # [speech,prompt,answer,eos]
        
        labels_ids = copy.deepcopy(example_ids) # [speech,prompt,answer,eos]
        labels_ids[:speech_length + prompt_length+1] = -1 #[-1,-1,answer,eos]; FIX(zhifu): speech_length + prompt_length->speech_length + prompt_length+1
        example_mask = example_ids.ge(-1) #FIX(GZF): [True,True,True,True]
        label_mask = labels_ids.ge(0) #[False,False,True,True]
        example_ids[~example_mask] = 0 #[speech,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX #[-100,answer,eos,-100]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            'speech_mel': speech_mel,
            'speech_length': speech_length
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
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat((sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
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
        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX) 
                                for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                for s in samples]) #FIX(GZF): attention_mask
        
        speech_mel_max_length = max([s['speech_mel'].shape[0] for s in samples])
        speech_mel = torch.stack([self.pad(s['speech_mel'], speech_mel_max_length, 0) 
                                for s in samples])
        
        speech_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            speech_mask[line, :sample['speech_length']+1] = 1 #FIX(GZF): sample['speech_length']+1

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'speech_mel': speech_mel,
            'speech_mask': speech_mask
        }


def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = AudioDataset(dataset_config, tokenizer, split)

    return dataset
