import os
import time
import json
import torch
import torch_npu
import sys
import copy
sys.path.append('/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/asr_fireredasr/model')
from fireredasr.models.fireredasr import FireRedAsr
from torch.utils.data import Dataset, DataLoader, IterableDataset
from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper
import torch.distributed as dist
import kaldiio

def model_factory(train_config, model_config, **kwargs):
    model_dir = model_config.firered_path
    model_path = os.path.join(model_dir, "model.pth.tar")
    encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
    llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
    model, tokenizer = load_firered_llm_model_and_tokenizer(
        model_path, encoder_path, llm_dir, train_config)
    ckpt_path = kwargs.get("ckpt_path", None) 
    if ckpt_path is not None:
            print("loading other parts from: {}".format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt_dict, strict=False)
    return model, tokenizer

def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir, train_config):
    # model_path = "/aistor/aispeech/hpc_stor01/home/pengjing00sx/SLAM-LLM/examples/asr_fireredasr/exp/aishell-1/20250311/conformer_linear_Qwen2-7B-Instruct_encodertrue_loratrue_padtrue_normal_asr_speedfalse_specaugfalse-1058/fireredasrllm_epoch_1_step_100/model.pth.tar"
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    # print(type(package["args"]))
    # input()
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    # if train_config.freeze_encoder:
    package["args"].freeze_encoder = 0
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
                key, path = line.strip().split(" ")
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

class FireRedDatasetLarge(IterableDataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()
        cmvn_path = dataset_config.cmvn_file
        self.feature_extractor = ASRFeatExtractor(cmvn_path)
        self.tokenizer = tokenizer
        self.split = split
        self.inference_mode = dataset_config.inference_mode
        
        # 根据split选择对应的数据路径
        if split == "train":
            data_path = dataset_config.train_scp_file_path
        elif split == "val":
            data_path = dataset_config.dev_scp_file_path
        elif split == "test":
            data_path = dataset_config.test_scp_file_path
        else:
            raise ValueError("Invalid split")
        
        # 加载多任务数据
        self.multitask_task_path = os.path.join(data_path, "multitask.jsonl")
    
    def get_audio_duration(self, wav_path):
        """计算 WAV 音频的时长（单位：秒）"""
        sample_rate, wav_np = kaldiio.load_mat(wav_path)
        if sample_rate != 16000:
            return None
        dur = wav_np.shape[0] / sample_rate
        return dur

    def __iter__(self):
        multitask_task_path = self.multitask_task_path
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 不在 DataLoader 的多进程环境中
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # 获取分布式环境中的进程信息
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # 计算每个 worker 和每个进程应该处理的数据范围
        total_num_workers = num_workers * world_size
        worker_rank = rank * num_workers + worker_id 
        data_index = 0

        with open(multitask_task_path) as f_task:
            for line in f_task:
                if (data_index % total_num_workers) == worker_rank :
                    item = json.loads(line.strip())

                    # ark_path = item.get("path", None)
                    # if ark_path is None:
                    #     print(f"⚠️ 缺少 ark_path: {item}")
                    #     continue
                    # # **🚀 计算音频时长**
                    # duration = self.get_audio_duration(ark_path)
                    # if duration is None or duration > 30.0:
                    #     continue  # 跳过时长超限的样本

                    yield{
                        "key": item["key"],
                        "target": item["target"],
                        "ark_path": item["path"]
                    }
                data_index += 1

    def collator(self, samples):
        assert samples is not None

        # 提取每个样本的字段
        keys = [sample["key"] for sample in samples]
        targets = [sample["target"] for sample in samples]
        batch_wav_path = [sample["ark_path"] for sample in samples]

        # 获取特征
        feats, lengths, durs = self.feature_extractor(batch_wav_path)

        # 获取 input_ids 和 target_ids
        if self.inference_mode:
            input_ids, attention_mask, target_ids, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""] * len(keys), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
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
    dataset = FireRedDatasetLarge(dataset_config, tokenizer, split)
    return dataset