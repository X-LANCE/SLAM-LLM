import os
import random
import json, yaml
import copy
import h5py

import numpy as np
import soundfile as sf
from scipy import signal

import torch
from torch.utils.data import Dataset

def format_prompt(instruction, input=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
        ),
        "prompt_no_input": (
            "Based on the audio you've heard, refer to the instruction and provide a response.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    }
    if input is None:
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


class SpatialAudioDatasetJsonl(Dataset):
    def __init__(
            self,
            dataset_config,
            tokenizer,
            split,
        ):
        super().__init__()
        dataset_path = os.path.join(dataset_config['qa_data_root'], dataset_config['stage'], split + '.jsonl')
        with open(dataset_path) as f:
            self.data = [json.loads(line) for line in f.readlines()]

        self.anechoic_data_root = dataset_config['anechoic_data_root'] # which is AudioSet in this case
        self.reverb_data_root = dataset_config['reverb_data_root']
        self.channel_type = dataset_config['channel_type']
        
        self.ext_audio = dataset_config['ext_audio']
        self.max_words = dataset_config['max_words']
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)

        self.tokenizer = tokenizer

        self.normalize = dataset_config['normalize']
        self.inference_mode = dataset_config['inference_mode']
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        audio_path = os.path.join(self.anechoic_data_root, sample['audio_id'] + self.ext_audio)
        reverb_path = os.path.join(self.reverb_data_root, self.channel_type, sample['reverb_id'])

        if sample['audio_id2'] is not None and sample['reverb_id2'] is not None:
            audio_path2 = os.path.join(self.anechoic_data_root, sample['audio_id2'] + self.ext_audio)
            reverb_path2 = os.path.join(self.reverb_data_root, self.channel_type, sample['reverb_id2'])
        else:
            audio_path2 = None
            reverb_path2 = None
            
        waveforms = self.load_waveform(audio_path, reverb_path, audio_path2, reverb_path2)
        
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        prompt = sample['question']
        prompt = format_prompt(prompt, None)
        prmopt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prmopt_ids)

        answer = sample['answer']        
        example = prompt + answer

        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(example_ids, dtype=torch.int64)        
        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length+prompt_length-1] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": waveforms,
            "audio_length": audio_length
        }
    
    @classmethod
    def normalize_audio(cls, audio_data, target_dBFS=-14.0):
        rms = np.sqrt(np.mean(audio_data**2)) # Calculate the RMS of the audio
    
        if rms == 0:  # Avoid division by zero in case of a completely silent audio
            return audio_data
        
        current_dBFS = 20 * np.log10(rms) # Convert RMS to dBFS
        gain_dB = target_dBFS - current_dBFS # Calculate the required gain in dB
        gain_linear = 10 ** (gain_dB / 20) # Convert gain from dB to linear scale
        normalized_audio = audio_data * gain_linear # Apply the gain to the audio data
        return normalized_audio

    @classmethod
    def load_waveform(cls, audio_path, reverb_path=None, audio_path2=None, reverb_path2=None, normalize=True):
        waveform, sr = sf.read(audio_path)
        
        if len(waveform.shape) > 1:
            waveform = waveform[:, 0] 
        if sr != 32000:
            waveform = signal.resample_poly(waveform, 32000, sr)
            sr = 32000
        if normalize:
            waveform = cls.normalize_audio(waveform, -14.0)

        waveform = waveform.reshape(1, -1)
        if reverb_path is not None:
            reverb = np.load(reverb_path)
            waveform = signal.fftconvolve(waveform, reverb, mode='full')

        waveform = torch.from_numpy(waveform).float()
        waveform = cls.padding(waveform, max_length=10*sr)

        if audio_path2 is not None and reverb_path2 is not None:
            waveform2, sr2 = sf.read(audio_path2)
            
            if len(waveform2.shape) > 1:
                waveform2 = waveform2[:, 0]
            if sr2 != 32000:
                waveform2 = signal.resample_poly(waveform2, 32000, sr2)
                sr2 = 32000
            if normalize:
                waveform2 = cls.normalize_audio(waveform2, -14.0)
            
            waveform2 = waveform2.reshape(1, -1)
            reverb2 = np.load(reverb_path2)
            waveform2 = signal.fftconvolve(waveform2, reverb2, mode='full')
            waveform2 = torch.from_numpy(waveform2).float()
            waveform2 = cls.padding(waveform2, max_length=10*sr)

            waveform = (waveform + waveform2) / 2
        return waveform

    @classmethod
    def padding(cls, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if sequence.shape[1] < max_length:
                    sequence = torch.nn.functional.pad(sequence, (0, max_length - sequence.shape[1]))
                else:
                    sequence = sequence[:, :max_length]
            else:
                if len(sequence) < max_length:
                    sequence = torch.cat(
                        (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([
            self.padding(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id) for s in samples])
        attention_mask = torch.stack([
            self.padding(s['attention_mask'], input_ids_max_length, False) for s in samples])

        audio = torch.stack([s['audio'] for s in samples])

        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, :sample['audio_length']] = 1

        # if self.inference_mode:
        #     keys = [s['key'] for s in samples]
        #     targets = [s['target'] for s in samples]

        #     return {
        #         "input_ids": input_ids,
        #         "attention_mask": attention_mask,
        #         "audio": audio,
        #         "modality_mask": modality_mask,
        #         "keys": keys,
        #         "targets": targets
        #     }

        labels = torch.stack([self.padding(s['labels'], input_ids_max_length, self.IGNORE_INDEX)
                              for s in samples])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio,
            "modality_mask": modality_mask
        }

def get_spatial_audio_dataset(dataset_config, tokenizer, split):
    dataset = SpatialAudioDatasetJsonl(dataset_config, tokenizer, split)
    return dataset