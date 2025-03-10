import os
import time
import json
import torch
import torch_npu
import sys
sys.path.append('/aistor/aispeech/hpc_stor01/home/pengjing00sx/SLAM-LLM/examples/asr_fireredasr/model')
from fireredasr.models.fireredasr import FireRedAsr
from torch.utils.data import Dataset, DataLoader
from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper

def model_factory(train_config, model_config, **kwargs):
    model_dir = model_config.ckpt_path
    model_path = os.path.join(model_dir, "model.pth.tar")
    encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
    llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
    model, tokenizer = load_firered_llm_model_and_tokenizer(
        model_path, encoder_path, llm_dir, train_config)
    model.eval()
    return model, tokenizer

def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir, train_config):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    if train_config.freeze_encoder:
        package["args"].freeze_encoder = 1
    if train_config.use_peft:
        package["args"].freeze_llm = 1
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer


class FireRedDataset(Dataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()
        cmvn_path = dataset_config.cmvn_file
        self.feature_extractor = ASRFeatExtractor(cmvn_path)
        self.tokenizer = tokenizer
        self.split = split
        self.inference_mode = dataset_config.inference_mode
        self.data_list = {}
        self.multitask_task_list = []
        if split == "train":
            data_path = dataset_config.train_scp_file_path
        elif split == "val":
            data_path = dataset_config.dev_scp_file_path
        elif split == "test":
            data_path = dataset_config.test_scp_file_path
        else:
            raise ValueError("Invalid split")
        data_scp_file_path = os.path.join(data_path,"my_wav.scp")
        with open(data_scp_file_path) as f:
            for line in f:
                key, path = line.split(" ")
                self.data_list[key] = path
        multitask_task_path = os.path.join(data_path,"multitask.jsonl")
        with open(multitask_task_path) as f:
            for line in f:
                item = json.loads(line.strip())
                if item["key"] in self.data_list:
                    self.multitask_task_list.append(item)
                else:
                    print(item)

    def __len__(self):
        return len(self.multitask_task_list)
    
    def __getitem__(self, index):

        # Deal with every wav one by one
        item = self.multitask_task_list[index]        
        key = [item["key"]]
        target = [item["target"]]

        return {
                "key": key,
                "target": target,
            }

    def collator(self, samples):
        assert samples is not None

        # Extract each field from the samples
        keys = [sample["key"][0] for sample in samples]
        targets = [sample["target"][0] for sample in samples]

        # Get padded feats
        batch_wav_path = []
        for key in keys:
            ark_path = self.data_list[key]
            batch_wav_path.append(ark_path)
        feats, lengths, durs = self.feature_extractor(batch_wav_path)

        # Get input_ids and target_ids
        # inference
        if self.inference_mode:
            input_ids, attention_mask, target_ids, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*len(keys), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
        # training
        else:
            input_ids, attention_mask, target_ids, clean_texts = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=targets, tokenizer=self.tokenizer,
                    max_len=128, decode=False)
        
        return {
            "keys": keys,
            "targets": targets,
            "feats": feats,
            "lengths": lengths,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
        }

def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = FireRedDataset(dataset_config, tokenizer, split)
    return dataset
