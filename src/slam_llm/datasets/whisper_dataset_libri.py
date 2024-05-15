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
from slam_llm.utils.compute_utils import calculate_output_length_1d

import torch
from torch.utils.data import Dataset
import whisper
import kaldiio
import copy
import numpy as np
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

class WhisperDataset(Dataset):
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
        self.prompt = dataset_config.get("prompt", None)
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        # self.prompt_library = [
        #     "Begin by converting the spoken words into written text. ",
        #     "Can you transcribe the speech into a written format? ",
        #     "Focus on translating the audible content into text. ",
        #     "Transcribe the speech by carefully listening to it. ",
        #     "Would you kindly write down the content of the speech? ",
        #     "Analyze the speech and create a written transcription. ",
        #     "Engage with the speech to produce a text-based version. ",
        #     "Can you document the speech in written form? ",
        #     "Transform the spoken words into text accurately. ",
        #     "How about putting the speech's content into writing? "
        # ]
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

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

    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)


        audio_raw = whisper.load_audio(audio_path)
        audio_raw = whisper.pad_or_trim(audio_raw)  #torch.Size([480000])
        audio_mel = whisper.log_mel_spectrogram(audio_raw,128).permute(1, 0)   # 128 torch.Size([3000, 128])

        return {
            'audio_mel': audio_mel,
            'key': key,
            'target': target,
            # 'ocr':ocr,
            # "previous_sentence":previous_sentence
        }             


    def collator(self, samples):
        assert samples is not None

        audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
        audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0) for s in samples])
        
        keys = [s['key'] for s in samples]
        targets = [s['target'] for s in samples]
        # ocrs = [s['ocr'] for s in samples]
        # previous_sentences = [s['previous_sentence'] for s in samples]


        return {
            'audio_mel': audio_mel,
            'keys': keys,
            'targets': targets,
            # 'ocrs': ocrs,
            # 'previous_sentences' : previous_sentences
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


    def __len__(self):
        return len(self.data_list)

def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = WhisperDataset(dataset_config, tokenizer, split)

    return dataset



   