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


import logging
logger = logging.getLogger(__name__)

import difflib
from functools import lru_cache
from tqdm import tqdm
import Levenshtein


def build_ngram_index(names, n=2):
    """构建N-Gram倒排索引"""
    index = {}
    for name in names:
        for i in range(len(name) - n + 1):
            ngram = name[i:i+n].lower()
            index.setdefault(ngram, set()).add(name)
    return index

def find_candidate_names(sentence, ngram_index, n=2):
    """通过N-Gram倒排索引找到候选人名"""
    candidates = set()
    for i in range(len(sentence) - n + 1):
        ngram = sentence[i:i+n].lower()
        candidates.update(ngram_index.get(ngram, []))       
    return candidates

# @lru_cache(maxsize=None)
@lru_cache(maxsize=100000)
def similarity(name, sentence):
    return Levenshtein.ratio(name, sentence)  #速度主要来源于这个函数的更换

def generate_ngrams(sentence, n):
    """生成长度为n的n-grams"""
    sentence = sentence.split()
    return [' '.join(sentence[i:i+n]) for i in range(len(sentence)-n+1)]

def calculate_similarity_score(name, sentence, length_tolerance=3):
    max_similarity = 0
    name_sentence = name.split()
    name_length = len(name_sentence)
    sentence_ngrams = generate_ngrams(sentence, name_length) #9
    
    for ngram in sentence_ngrams:
        if abs(len(ngram) - len(name)) <= length_tolerance:
            sim = similarity(name.lower(), ngram.lower())
            max_similarity = max(max_similarity, sim)
    return max_similarity

def score_candidates(candidates, sentence):
    """为候选人名计算得分"""
    scores = {}
    for candidate in candidates:
        score = calculate_similarity_score(candidate, sentence)
        scores[candidate] = score
    return scores



class HotwordsInferAddgtDataset(torch.utils.data.Dataset):
    
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

        # 
        self.hotwords_list=[]
        self.biaswords_list=[]
        with open(dataset_config.infer_file,'r') as fref:
            for line in fref:
                line=line.strip().split('\t')
                # id = line[0]
                # label = line[1]
                hotwords = line[2]
                biaswords= line[3]
                self.hotwords_list.append(hotwords)
                self.biaswords_list.append(biaswords)
        
        self.infer_type=dataset_config.infer_type
        if self.infer_type=="filter":
            self.infer_list=[]
            with open(dataset_config.ctc_file,'r') as finfer:
                for line in finfer:
                    self.infer_list.append(line.strip())

        # analyze
        self.hotwords_num=0
        self.miss_words_num=0


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
            # audio_raw = np.concatenate((np.zeros(random.randint(0, 16000)), audio_raw, np.zeros(random.randint(0, 16000)))).astype(audio_raw.dtype)[:16000*30]
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        if self.infer_type=="nobias":
            ocr=""
        elif self.infer_type=="gt":
            ocr=eval(self.hotwords_list[index])
            ocr=" ".join(ocr)
            ocr = ocr.upper()
        elif self.infer_type=="filter":
            gt=eval(self.hotwords_list[index])
            infer_sentence=self.infer_list[index].lower()
            biaswords=eval(self.biaswords_list[index])
            ngram_index=build_ngram_index(biaswords)
            candidates = find_candidate_names(infer_sentence, ngram_index) #第一个len11
            scores = score_candidates(candidates, infer_sentence)
            sorted_dict = sorted(scores.items(), key=lambda item: item[1],  reverse=True)
            high_score_items = [(k, value) for k, value in sorted_dict if value > 0.9] 
            if len(high_score_items) < 15:
                high_score_items = sorted_dict[:15]
            keys_list = [k for k, _ in high_score_items]
           
            if len(high_score_items)>15:
                logger.info("longer than 15 candidates, cand_num: %d", len(high_score_items))

            # ======== count recall
            miss=False
            for name in gt:
                self.hotwords_num+=1
                if name not in keys_list:
                    logger.info("miss name: %s", name)
                    keys_list.insert(0,name)      # !!!!!!
                    self.miss_words_num+=1
                    miss=True
            if miss:
                logger.info("key: %s", key)
                logger.info("infer sentence: %s",infer_sentence)
                logger.info("target sentence: %s", target)
                logger.info("name: %s, gt: %s, keys_list: %s", name, gt, keys_list)
            # ========
            
        ocr = " ".join(keys_list).upper()
        prompt = "Transcribe speech to text. Some hotwords might help. The hotwords are \"{}\". "
        prompt = prompt.format(ocr)
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt) #'USER: Transcribe speech to text. Some hotwords might help. The hotwords are "anon harshly". \n ASSISTANT:'
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

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_length": audio_length,
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



def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = HotwordsInferAddgtDataset(dataset_config, tokenizer, split)

    return dataset
