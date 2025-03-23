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
                    try:
                        item = json.loads(line.strip())
                        ark_path = item["path"]
                        numpy_array = kaldiio.load_mat(ark_path)
                        audio_raw = numpy_array[1].astype(np.float32) / 32768
                        if len(audio_raw) / 16000 > 30: 
                            continue
                        key = item["key"]
                        target = item["target"].upper()
                        ## data augmentation
                        if self.split == "train" and self.speed_perturb == True:
                            audio_raw = self.speedPerturb(audio_raw)
                        if self.split == "train" and self.add_noise == True:
                            audio_raw = self.addNoise(audio_raw, self.musan_list)
                        # if self.split == "train" and self.add_reverb == True:
                        #     audio_raw = self.simulate_room_reverb(audio_raw, 16000).astype(np.float32)
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
                            if self.split == "train" and self.spec_augmentation == True:
                                audio_mel = self.specAugment(audio_mel)
                            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
                            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
                            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
                        if self.fix_length_audio > 0:
                            audio_length = self.fix_length_audio
                        audio_pseudo = torch.full((audio_length,), -1) # placeholder

                        prompt = random.choice(self.multitask_prompt_list[item["task"]])
                        prompt = self.prompt_template1.format(prompt)
                        if item["task"] in ["prevtext","hotword","domain"]:
                            prompt = prompt.format(item[item["task"]].upper())
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
                    except:
                        print("[Item Error]",key,target)
                        exit(1)
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


def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = MultiTaskDataset(dataset_config, tokenizer, split)
    return dataset



    