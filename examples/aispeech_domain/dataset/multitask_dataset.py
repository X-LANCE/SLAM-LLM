import torch
from torch.utils.data import Dataset,IterableDataset
import whisper
import kaldiio
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
import torchaudio.transforms as T
from torchaudio.transforms import SpeedPerturbation
import torchaudio
import torchaudio.functional as F
import random
class MultiTaskDataset(IterableDataset):
    def __init__(self, dataset_config, tokenizer=None, split='train',musan_path=None):
        super().__init__()
        self.multitask_prompt_list = {}
        multitask_prompt_path = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/multiprompt.jsonl"
        with open(multitask_prompt_path) as f_prompt:
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
            assert(0)
        if musan_path is not None:
            self.musan_list = []
            with open(musan_path) as f:
                for line in f:
                    key,path = line.split(" ")
                    self.musan_list.append(path)
        

        self.llm_name = dataset_config.get("llm_name", None)
        self.prompt_style = dataset_config.get("prompt_style", "normal")
        if self.llm_name == "Qwen2.5-7B-Instruct":
            if self.prompt_style == "normal":
                self.prompt_template1 = "{}"
            elif self.prompt_style == "instruct":
                self.prompt_template1 = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        else:
            if self.prompt_style == "normal":
                self.prompt_template1 = "{}"
            elif self.prompt_style == "instruct":
                self.prompt_template1 = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.spec_augmentation =  dataset_config.get("spec_augmentation", False)
        self.speed_perturb = dataset_config.get("speed_perturb", False)
        self.add_noise = dataset_config.get("musan", False)
        self.add_reverb = dataset_config.get("add_reverb", False)
        self.noise_file_path = dataset_config.get("noise_file_path", False)
        if self.add_noise == True:
            self.musan_wav_files = []
            for root, dirs, files in os.walk(self.noise_file_path):
                for file in files:
                    if file.endswith('.wav'):
                        self.musan_wav_files.append(os.path.join(root, file))
        self.pad_or_trim = dataset_config.get("pad_or_trim", False)
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.prompt_mode = dataset_config.get("prompt_mode", None)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 



    def speedPerturb(self, audio_raw):
        orig_freq = 16000
        # 定义速度扰动因子，例如 [0.9, 1.0, 1.1] 表示速度减少10%，保持不变，增加10%Q
        factors = [0.9,1,1.1]
        # 创建速度扰动变换
        speed_perturbation = SpeedPerturbation(orig_freq, factors)
        # 应用速度扰动
        audio_raw = torch.from_numpy(audio_raw)
        # 由于 SpeedPerturbation 返回的是一个函数，我们需要调用它
        # 并且传入原始音频张量
        audio_raw = speed_perturbation(audio_raw)[0]
        return audio_raw
    def specAugment(self, spec):
        spec = spec.permute(1, 0).unsqueeze(0)
        stretch = T.TimeStretch(n_freq=128)
        rate = random.random()*0.2 + 0.9
        Timemasking = T.TimeMasking(time_mask_param=100)
        Frequencymasking = T.FrequencyMasking(freq_mask_param=27)
        spec = stretch(spec, rate).to(torch.float32)
        spec = Timemasking(spec)
        spec = Timemasking(spec)
        spec = Frequencymasking(spec)
        spec = Frequencymasking(spec)  
        spec = spec.squeeze(0).permute(1, 0)
        return spec
    def addNoise(self, audio_raw):
        noise, _ = torchaudio.load(random.choice(self.musan_wav_files))
        noise.unsqueeze_(0)
        # 如果语音比噪声长，随机选择噪声的起始点
        if audio_raw.shape > noise.shape:
            # 随机选择噪声的起始点
            start_idx = random.randint(0, audio_raw.shape - noise.shape)
            # 在语音的随机位置开始添加噪声
            speech_with_noise = torch.zeros_like(audio_raw)
            speech_with_noise[:, start_idx:start_idx + noise.shape] += noise
        else:
            # 如果噪声比语音长，从噪声的随机位置开始截取
            start_idx = random.randint(0, noise.shape - audio_raw.shape)
            noise = noise[:, start_idx:start_idx + audio_raw.shape]
            # 直接将噪声添加到语音中
        snr_dbs = random.randomint(1, 30)
        noisy_speeches = F.add_noise(audio_raw, noise, snr_dbs)
        return noisy_speeches
    # def simulate_room_reverb(self, audio_raw,fs):
    #     room_dim = [random.uniform(3, 10), random.uniform(3, 10), random.uniform(2, 5)]
    # # 随机生成目标混响时间（RT60，单位：秒）
    #     rt60_tgt = random.uniform(0.3, 1.0)
    #     # 生成随机房间参数
    #     # 使用 Sabine 公式计算吸声系数和反射阶数
    #     e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    #     # 创建房间
    #     room = pra.ShoeBox(
    #         room_dim,
    #         fs=fs,
    #         materials=pra.Material(e_absorption),
    #         max_order=int(max_order),
    #         use_rand_ism=True,  # 使用随机化图像方法减少回声
    #         max_rand_disp=0.05,  # 最大随机位移（单位：米）
    #     )
    #     # 随机生成声源位置
    #     source_position = [random.uniform(0.5, room_dim[0] - 0.5),
    #                     random.uniform(0.5, room_dim[1] - 0.5),
    #                     random.uniform(0.5, room_dim[2] - 0.5)]
    #     room.add_source(source_position, signal=audio_raw)
    #     # 随机生成麦克风位置
    #     mic_locs = np.c_[
    #         [random.uniform(0.5, room_dim[0] - 0.5), random.uniform(0.5, room_dim[1] - 0.5), random.uniform(0.5, room_dim[2] - 0.5)],
    #     ]
    #     room.add_microphone_array(mic_locs)
    #     # 运行模拟
    #     room.simulate()
    #     # 返回麦克风阵列的信号
    #     return room.mic_array.signals[0, :]
    def __iter__(self):
        multitask_task_path = os.path.join(self.data_path,"multitask.jsonl")
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
                    key = item["key"]
                    target = item["target"]
                    yield {
                        "target"
                    }             
                data_index += 1        
    
    

    def collator(self, samples):
        assert samples is not None
        target = [ _["target"] for _ in samples]
        processed_data = self.tokenizer(text=target, return_tensors="pt")
        # 处理labels的生成
        labels = copy.deepcopy(processed_data["input_ids"])
        processed_data["labels"] = labels
        return processed_data


def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = MultiTaskDataset(dataset_config, tokenizer, split)
    return dataset



    