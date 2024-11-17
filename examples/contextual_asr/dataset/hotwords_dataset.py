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


class HotwordsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_config,
            tokenizer=None,
            split='train',
        ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        data_parallel_size = 1
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 
        self.Pkeep = dataset_config.get("Pkeep", 0.5)
        self.Norder = dataset_config.get("Norder", 4)

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

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", "ASR")
        key = data_dict.get("key", None)

        audio_raw = whisper.load_audio(audio_path)
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320 # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        if self.inference_mode:
            return {
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
            }
        else:
            return {
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "target":target,
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

        if self.dataset_config.infer_type=="nobias":
            selected_ngrams=""
        else:
            selected_ngrams_list = []
            for s in samples:
                label = s['target']
                if random.random() < self.Pkeep: 
                    words = label.split()
                    n = min(random.randint(1, self.Norder),len(words))
                    if len(words) >= n:
                        start_index = random.randint(0,len(words)-n)
                    selected_ngrams = words[start_index:start_index + n]
                    selected_ngrams_list.append(" ".join(selected_ngrams))
            selected_ngrams = " ".join(selected_ngrams_list)
            
        prompt = "Transcribe speech to text. Some hotwords might help. The hotwords are \"{}\". "
        prompt = prompt.format(selected_ngrams)
        prompt = self.prompt_template.format(prompt)  #'USER: Transcribe speech to text. Some hotwords might help. The hotwords are "ONLY OR FOUND DEMANDS YOU THREE RESPONSIVE AND COVER\'D BY TO". \n ASSISTANT:'
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        if self.inference_mode:
            for i in range(len(samples)):
                audio_pseudo = torch.full((samples[i]["audio_length"],), -1)
                prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
                example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
                example_mask = example_ids.ge(-1)  # [True,True]
                
                samples[i]["input_ids"] = example_ids
                samples[i]["attention_mask"] = example_mask
        else:
            for i in range(len(samples)):
                audio_length = samples[i]["audio_length"]
                audio_pseudo = torch.full((audio_length,), -1)
                answer = self.answer_template.format(samples[i]["target"])
                example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
                example_ids = self.tokenizer.encode(example)  # [prompt,answer]
                example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
                example_ids = torch.tensor(
                    example_ids, dtype=torch.int64
                )
                example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

                labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
                labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
                example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

                label_mask = labels_ids.ge(0)  # [False,False,True,True]
                example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
                labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

                samples[i]["input_ids"] = example_ids
                samples[i]["labels"] = labels_ids
                samples[i]["attention_mask"] = example_mask

        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id) for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False) for s in samples])

        if self.input_type == "raw":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0) for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0) for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, :sample['audio_length']] = 1

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets
            }
        else:
            labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX) for s in samples])
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask
            }

def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = HotwordsDataset(dataset_config, tokenizer, split)
    return dataset

