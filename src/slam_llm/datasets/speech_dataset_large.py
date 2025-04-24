import torch
from torch.utils.data import Dataset,IterableDataset
import whisper
import kaldiio
import types
from functools import partial
# import pyroomacoustics as pra
import torch.distributed as dist
import string
import copy
import numpy as np
import copy
from tqdm import tqdm
import os
import json
import random
import torchaudio
import random
import logging
import subprocess


class MultiTaskDataset(IterableDataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()
        self.multitask_prompt_list = {}
        self.append_info_tasks = dataset_config.append_info_tasks
        with open(dataset_config.multitask_prompt_path) as f_prompt:
            for line in f_prompt:
                item = json.loads(line.strip())
                if item["task"] in self.multitask_prompt_list:
                    self.multitask_prompt_list[item["task"]].append(item["prompt"])
                else:
                    self.multitask_prompt_list[item["task"]] = [item["prompt"]]
        print(f"[Prompt] {self.multitask_prompt_list}")
        if split == "train":
            self.data_path = dataset_config.train_scp_file_path
        elif split == "val":
            self.data_path = dataset_config.dev_scp_file_path
        elif split == "test":
            self.data_path = dataset_config.test_scp_file_path
        else:
            raise ValueError("split must be train val test")
        
        self.llm_name = dataset_config.get("llm_name", None)
        self.prompt_template1 = dataset_config.get("prompt_style", "{}")
        self.answer_template = "{}"
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.pad_or_trim = dataset_config.get("pad_or_trim", False)
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

    def __iter__(self):
        multitask_task_path = os.path.join(self.data_path,"multitask.jsonl")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Not in the multi-processing environment of DataLoader.
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # Obtain the process information in the distributed environment.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # Calculate the data range that each worker and each process should handle.
        total_num_workers = num_workers * world_size
        worker_rank = rank * num_workers + worker_id 
        data_index = 0
        with open(multitask_task_path) as f_task:
            for line in f_task:
                if (data_index % total_num_workers) == worker_rank:
                    item = json.loads(line.strip())
                    ark_path = item["path"]
                    numpy_array = kaldiio.load_mat(ark_path)
                    audio_raw = numpy_array[1].astype(np.float32) / 32768
                    if len(audio_raw) / 16000 > 30: 
                        continue
                    key = item["key"]
                    target = item["target"]
                    if self.input_type == "raw":
                        audio_raw = torch.from_numpy(audio_raw).float()
                        if self.normalize:
                            audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
                        audio_length = len(audio_raw) // 320 # ad-hoc for fairseq 320x downsample
                        audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
                    elif self.input_type == "mel":
                        if self.pad_or_trim == True:
                            audio_raw = whisper.pad_or_trim(audio_raw)
                        audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
                        audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
                        audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
                    if self.fix_length_audio > 0:
                        audio_length = self.fix_length_audio
                    audio_pseudo = torch.full((audio_length,), -1) # placeholder

                    prompt = random.choice(self.multitask_prompt_list[item["task"]])
                    prompt = self.prompt_template1.format(prompt)
                    if item["task"] in self.append_info_tasks:
                        prompt = prompt.format(item[item["task"]])
                    prompt_ids = self.tokenizer.encode(prompt)
                    prompt_length = len(prompt_ids)
                    
                    if self.inference_mode:
                        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
                        example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
                        example_mask = example_ids.ge(-1)  # [True,True]
                        yield {
                            "input_ids": example_ids,
                            "attention_mask": example_mask,
                            "audio": audio_raw if self.input_type == "raw" else None,
                            "audio_mel": audio_mel if self.input_type == "mel" else None,
                            'audio_length': audio_length,
                            'key': key,
                            'target': target,
                        }
                    else:
                        answer = self.answer_template.format(target)
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
                        yield {
                            "input_ids": example_ids,
                            "labels": labels_ids,
                            "attention_mask": example_mask,
                            "audio": audio_raw if self.input_type == "raw" else None,
                            "audio_mel": audio_mel if self.input_type == "mel" else None,
                            'audio_length': audio_length,
                        }
                data_index += 1      
            
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
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                    for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                        for s in samples])
        if self.input_type == "raw":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                        for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel":
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                    for s in samples])
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

        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX)
                                for s in samples])
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

class MultiTaskDynamicBatchDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, window_class) -> None:
        super().__init__()
        self.dp = dataset
        
        assert window_class is not None
        self.window_class = window_class
        self.collator = self.dp.collator
        self._buffer = []
    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, self._buffer):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._buffer
                del self._buffer
                self._buffer = [elem]
        if len(self._buffer) > 0:
            yield self._buffer
        del self._buffer
        self._buffer = []
         
    
def window_class(elem,buffer,max_frame_length):
    if len(buffer) == 0:
        return True
    max_frame = max(len(elem["input_ids"]),max([ len(_["input_ids"]) for _ in buffer]))
    return (len(buffer) + 1) * max_frame > max_frame_length

def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = MultiTaskDataset(dataset_config, tokenizer, split)
    if split == "train":
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.train_max_frame_length))
    else:
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.eval_max_frame_length))
    return dataset



    
