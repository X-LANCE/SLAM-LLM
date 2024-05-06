import os.path as osp
import random
import json, yaml
import copy
from fairseq.data import Dictionary
import numpy as np
from scipy import signal
import soundfile as sf
import os
import torch
import torchaudio
from torch.utils.data import Dataset
from fairseq.data import (
    data_utils,
    indexed_dataset,
)
from fairseq import utils

def make_mask(prompt_length, token_length):
    future_mask = torch.triu(
        utils.fill_with_neg_inf(
            torch.zeros(prompt_length + token_length, prompt_length + token_length)
        ), 1
    )
    future_mask[:prompt_length, :prompt_length] = 0
    return future_mask

def merge(samples, key, pad_idx, left_pad=False, eos_idx=None, move_eos_to_beginning=False, pad_to_length=None):
    return data_utils.collate_tokens(
        [s[key] for s in samples],
        pad_idx,
        eos_idx,
        left_pad,
        move_eos_to_beginning,
        pad_to_length=pad_to_length,
        pad_to_multiple=1,
    ), [s[key].size(0) for s in samples]

def get_target_prev_out(samples, flag, prev_left_tag, order, first_at=False, pad_idx=None):
    target = None
    target, lengthes = merge(
        samples,
        flag,
        pad_idx=pad_idx,
        pad_to_length=None,
    )
    target = target.index_select(0, order)
    tgt_lengths = torch.LongTensor(lengthes).index_select(0, order)
    ntokens = tgt_lengths.sum().item()

    prev_output_tokens = None
    # we create a shifted version of targets for feeding the
    # previous output token(s) into the next decoder step
    prev_output_tokens, _ = merge(
        samples,
        flag,
        pad_idx=pad_idx,
        move_eos_to_beginning=True,
        eos_idx=prev_left_tag,
        pad_to_length=None,
    )
    prev_output_tokens = prev_output_tokens.index_select(0, order)
    return target, tgt_lengths, ntokens, prev_output_tokens

def sort_by_len(lengthes):
    src_lengths = torch.LongTensor(lengthes)
    src_lengths, sort_order = src_lengths.sort(descending=True)
    return src_lengths, sort_order

class VallexDataset(Dataset):
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split="train",
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split

        self.at_dict = Dictionary.load(os.path.join(self.dataset_config.train_data_path, "dict.at.txt"))
        self.st_dict = Dictionary.load(os.path.join(self.dataset_config.train_data_path, "dict.st.txt"))
        
        self.at_dict.tts_flag = self.at_dict.add_symbol("<TTS>")
        self.st_dict.asr_flag = self.st_dict.add_symbol("<ASR>")
        self.at_dict.mt_flag = self.st_dict.add_symbol("<MT>")
        
        self.zh_at_dataset, self.zh_st_dataset, self.zh_sizes, self.zh_datanum = self.load_data("zh")
        self.en_at_dataset, self.en_st_dataset, self.en_sizes, self.en_datanum = self.load_data("en")
        
        self.lang_id_dict = {"en": 0,"zh": 1}


    def load_data(self, lang):
        temp_data_path = os.path.join(
            self.dataset_config.train_data_path, "{}.at{}.{}".format(self.split, 0, lang)
        )
        print("loadding:"+temp_data_path)
        at_dataset = data_utils.load_indexed_dataset(temp_data_path, self.at_dict, None)
        print(
            "{} {} at2st {} examples".format(temp_data_path, self.split, len(at_dataset))
        )
        
        temp_data_path = os.path.join(
            self.dataset_config.train_data_path, "{}.st.{}".format(self.split, lang)
        )
        assert indexed_dataset.dataset_exists(temp_data_path, impl=None), temp_data_path
        st_dataset = data_utils.load_indexed_dataset(temp_data_path, self.st_dict, None)
        
        sizes = []
        for at_item, st_item in zip(at_dataset, st_dataset):
            sizes.append(int(at_item.size(0) + st_item.size(0)))
        return at_dataset, st_dataset, np.array(sizes), len(np.array(sizes))
    
    def __len__(self):
        return max(self.zh_datanum, self.en_datanum)
    
    def __getitem__(self, index):
        zh_idx = index%self.zh_datanum
        en_idx = index%self.en_datanum
        
        zh_size = self.zh_sizes[zh_idx]
        en_size = self.en_sizes[en_idx]
        
        zh_at = self.zh_at_dataset[zh_idx]
        zh_st = self.zh_st_dataset[zh_idx]
        en_at = self.en_at_dataset[en_idx]
        en_st = self.en_st_dataset[en_idx]
        
        return {
            "zh_at": zh_at,
            "zh_st": zh_st,
            "en_at": en_at,
            "en_st": en_st,
        }
    
    def collator(self, samples):
        assert samples is not None
        # zh
        zh_st, zh_st_len = merge(samples, "zh_st", pad_idx=self.st_dict.pad())
        zh_st_len, zh_sort_order = sort_by_len(zh_st_len)
        zh_st = zh_st.index_select(0, zh_sort_order)
        zh_tgt_at, zh_tgt_len, zh_ntokens, zh_prev_at = get_target_prev_out(
            samples, "zh_at", prev_left_tag=self.at_dict.bos(), order=zh_sort_order, first_at=True, pad_idx=self.at_dict.eos()
        )
        zh_self_atten_mask = make_mask(zh_st.size(1), zh_prev_at.size(1))
        zh_padding_mask = torch.cat([zh_st, zh_prev_at], dim=1).eq(self.st_dict.pad())
        zh_id = torch.ones(size=[zh_st.size(0), 1]) * self.lang_id_dict["zh"]
        
        # en
        en_st, en_st_len = merge(samples, "en_st", pad_idx=self.st_dict.pad())
        en_st_len, en_sort_order = sort_by_len(en_st_len)
        en_st = en_st.index_select(0, en_sort_order)
        en_tgt_at, en_tgt_len, en_ntokens, en_prev_at = get_target_prev_out(
            samples, "en_at", prev_left_tag=self.at_dict.bos(), order=en_sort_order, first_at=True, pad_idx=self.at_dict.eos()
        )
        en_self_atten_mask = make_mask(en_st.size(1), en_prev_at.size(1))
        en_padding_mask = torch.cat([en_st, en_prev_at], dim=1).eq(self.st_dict.pad())
        en_id = torch.ones(size=[en_st.size(0), 1]) * self.lang_id_dict["en"]

        # print(zh_st.size(), zh_prev_at.size(), zh_tgt_at.size(), zh_self_atten_mask.size(), zh_padding_mask.size())
        # print(en_st.size(), en_prev_at.size(), en_tgt_at.size(), en_self_atten_mask.size(), en_padding_mask.size())
        return {
            "zh": {
                "st_tokens": zh_st,
                "at_tokens_wbos": zh_prev_at,
                "at_tokens_tgt": zh_tgt_at,
                "self_atten_mask": zh_self_atten_mask,
                "padding_mask": zh_padding_mask,
                "langid": zh_id.long()
            },
            "en": {
                "st_tokens": en_st,
                "at_tokens_wbos": en_prev_at,
                "at_tokens_tgt": en_tgt_at,
                "self_atten_mask": en_self_atten_mask,
                "padding_mask": en_padding_mask,
                "langid": en_id.long()
            }
        }

def get_vallex_dataset(dataset_config, tokenizer, split):
    dataset = VallexDataset(dataset_config, tokenizer, split)
    return dataset