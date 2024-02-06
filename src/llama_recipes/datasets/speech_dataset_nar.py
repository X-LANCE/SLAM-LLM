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


import re
import string

def remove_punctuation(text):
    # 定义英文标点符号
    en_punct = string.punctuation
    # 定义中文标点符号（部分常用的）
    cn_punct = '。？！，、；：“”‘’（）《》【】…—～·'
    # 合并英文和中文标点符号
    all_punct = en_punct + cn_punct
    # 创建正则表达式模式，匹配任何在all_punct中的字符
    punct_pattern = re.compile('[{}]'.format(re.escape(all_punct)))
    # 使用正则表达式的sub方法替换掉这些字符
    return punct_pattern.sub('', text)

# # 示例文本
# text = 'Hello, world! 你好，世界！这是一个测试。Hello; 『测试』English: "text", 中文‘测试’。'
# clean_text = remove_punctuation(text)
# print(clean_text)

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
        self.prompt = dataset_config.get("prompt", None)
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
        self.prompt = "Transcribe speech to text."
        # self.prompt_template = "USER: {}\n ASSISTANT:"
        # self.prompt_template = "USER: \nINSTRUCTION: {}\nnINPUT: {}\nASSISTANT: ".format(prompt)
        # self.prompt = self.prompt_template
        # self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        # assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"

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
        # with open(dataset_config.train_data_path, encoding='utf-8') as fin:
        #         for line in fin:
        #             data_dict = json.loads(line.strip())
        #             self.data_list.append(data_dict)
        # if split == "train":
        #     self.data_list = self.data_list[:80]
        # else:
        #     self.data_list = self.data_list[80:100]

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
        target = remove_punctuation(target)
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
            # audio_raw = whisper.pad_or_trim(audio_raw)
            # audio_raw = np.concatenate((np.zeros(random.randint(0, 16000)), audio_raw, np.zeros(random.randint(0, 16000)))).astype(audio_raw.dtype)[:16000*30]
            audio_mel = whisper.log_mel_spectrogram(audio_raw).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        elif self.input_type == "paraformer_fbank":
            self.fix_length_audio = -1
            audio_length = None
            audio_raw = torch.from_numpy(audio_raw)
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        if audio_length is not None:
            audio_pseudo = torch.full((audio_length,), -1) # placeholder

        prompt_pre = "USER: \nINSTRUCTION: {}\nINPUT: ".format(self.prompt) # "USER: \nINSTRUCTION: {}\nnINPUT: {}\nASSISTANT: "

        # prompt = self.prompt_template.format(prompt)
        prompt_ids_pre = self.tokenizer.encode(prompt_pre) # [bos,prompt]
        prompt_pre_length = len(prompt_ids_pre)

        # if self.inference_mode:
        #     prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
        #     example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
        #     example_mask = example_ids.ge(-1)  # [True,True]
        #
        #     return {
        #         "input_ids": example_ids,
        #         "attention_mask": example_mask,
        #         "audio": audio_raw if self.input_type == "raw" else None,
        #         "audio_mel": audio_mel if self.input_type == "mel" else None,
        #         "audio_length": audio_length,
        #         "key": key,
        #         "target": target,
        #     }

        
        # answer = self.answer_template.format(target.lower())
        prompt_input = "{}{}".format(prompt_pre, target)
        prompt_input_ids = self.tokenizer.encode(prompt_input)
        audio_length = len(prompt_input_ids) - prompt_pre_length
        # prompt_input_ids = prompt_input_ids[1:] # remove bos
        example_ids = prompt_input_ids + [self.tokenizer.pad_token_id]
        example_ids = torch.tensor(example_ids, dtype=torch.int64) #[bos, prompt, input, pad]
        example_ids[prompt_pre_length:] = -1  # [bos, prompt,-1,-1]
        example_mask = example_ids.ge(-1) # [true, true, true, true], length mask
        
        prompt_answer = "{}{}".format(prompt_pre, target)
        prompt_answer_ids = self.tokenizer.encode(prompt_answer)
        answer_length = len(prompt_answer_ids) - prompt_pre_length
        labels_ids = copy.deepcopy(prompt_input_ids) + [self.tokenizer.eos_token_id]
        labels_ids = torch.tensor(labels_ids, dtype=torch.int64)  # [bos, prompt, input, eos]
        labels_ids[:prompt_pre_length] = -1  # [-1, -1, input, eos]
        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,input,eos]
        
        # # example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
        # example_ids = self.tokenizer.encode(example)  # [bos,prompt,answer]
        # audio_length = len(example_ids) - prompt_length
        # # audio_length = torch.tensor(audio_length, dtype=torch.int64)
        # example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        # example_ids = torch.tensor(example_ids, dtype=torch.int64)
        # # example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        # labels_ids = copy.deepcopy(example_ids)  # [prompt,answer,eos]
        # labels_ids[:prompt_length] = -1  # [-1,answer,eos];
        # example_ids[prompt_length:] = -1 # [prompt,-1,-1]

        # label_mask = labels_ids.ge(0)  # [False,False,True,True]
        # example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        # labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]
        
        
        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" or self.input_type == "paraformer_fbank"  else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_pre_length,
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
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                 for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                      for s in samples])
        audio_raw, audio_mask = None, None
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
                
        elif self.input_type == "paraformer_fbank":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            audio_length = []
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
                audio_length_cur = torch.tensor([sample['audio_length']], dtype=torch.int64)
                audio_length.append(audio_length_cur)
            audio_length = torch.nn.utils.rnn.pad_sequence(audio_length, batch_first=True, padding_value=0)
            
        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, sample['prompt_length']:sample['prompt_length']+sample['audio_length']] = 1

        # if self.inference_mode:
        #     keys = [s['key'] for s in samples]
        #     targets = [s['target'] for s in samples]
        #
        #     return {
        #         "input_ids": input_ids,
        #         "attention_mask": attention_mask,
        #         "audio": audio_raw if self.input_type == "raw" else None,
        #         "audio_mask": audio_mask if self.input_type == "raw" else None,
        #         "audio_mel": audio_mel if self.input_type == "mel" else None,
        #         "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
        #         "modality_mask": modality_mask,
        #         "keys": keys,
        #         "targets": targets
        #     }

        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX)
                              for s in samples])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask,
            "audio_length": audio_length,
        }


class SpeechDatasetNoPrompttJsonl(torch.utils.data.Dataset):
    
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
        prompt = "Transcribe speech to text."
        # self.prompt_template = "USER: {}\n ASSISTANT:"
        self.prompt_template = "USER: \nINSTRUCTION: {}\nASSISTANT: ".format(prompt)
        self.prompt = self.prompt_template
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        # assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]"
        
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
        # with open(dataset_config.train_data_path, encoding='utf-8') as fin:
        #         for line in fin:
        #             data_dict = json.loads(line.strip())
        #             self.data_list.append(data_dict)
        # if split == "train":
        #     self.data_list = self.data_list[:80]
        # else:
        #     self.data_list = self.data_list[80:100]
    
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
            audio_length = len(audio_raw) // 320  # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // 5  # ad-hoc for 5x fc downsample
        elif self.input_type == "mel":
            # audio_raw = whisper.pad_or_trim(audio_raw)
            # audio_raw = np.concatenate((np.zeros(random.randint(0, 16000)), audio_raw, np.zeros(random.randint(0, 16000)))).astype(audio_raw.dtype)[:16000*30]
            audio_mel = whisper.log_mel_spectrogram(audio_raw).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5  # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        elif self.input_type == "paraformer_fbank":
            self.fix_length_audio = -1
            audio_length = None
            audio_raw = torch.from_numpy(audio_raw)
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        if audio_length is not None:
            audio_pseudo = torch.full((audio_length,), -1)  # placeholder
        
        prompt = self.prompt
        if prompt is None:
            # prompt = random.choice(self.prompt_library)
            # prompt = "Transcribe speech to text. "
            prompt = "Transcribe speech to text."
        # prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)  # [bos,prompt]
        prompt_length = len(prompt_ids)
        
        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]
            
            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
            }
        
        answer = self.answer_template.format(target.lower())
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
        example_ids = self.tokenizer.encode(example)  # [bos,prompt,answer]
        audio_length = len(example_ids) - prompt_length
        # audio_length = torch.tensor(audio_length, dtype=torch.int64)
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(example_ids, dtype=torch.int64)
        # example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]
        
        labels_ids = copy.deepcopy(example_ids)  # [prompt,answer,eos]
        labels_ids[:prompt_length] = -1  # [-1,answer,eos];
        example_ids[prompt_length:] = -1  # [prompt,-1,-1]
        
        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        # example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]
        example_mask = example_ids.ge(-1)
        
        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" or self.input_type == "paraformer_fbank" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
            "prompt_length": prompt_length,
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
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                 for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                      for s in samples])
        audio_raw, audio_mask = None, None
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
            audio_mel_post_mask = torch.zeros(len(samples), (
                    audio_mel_max_length + 1) // 2)  # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
        
        elif self.input_type == "paraformer_fbank":
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            audio_length = []
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
                audio_length_cur = torch.tensor([sample['audio_length']], dtype=torch.int64)
                audio_length.append(audio_length_cur)
            audio_length = torch.nn.utils.rnn.pad_sequence(audio_length, batch_first=True, padding_value=0)
        
        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, sample['prompt_length']:sample['prompt_length'] + sample['audio_length']] = 1
        
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
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask,
            "audio_length": audio_length,
        }


def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)

    return dataset
