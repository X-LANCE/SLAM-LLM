import json
import copy

import numpy as np

import torch
import whisper
from slam_llm.utils.compute_utils import calculate_output_length_1d
from utils.snac_utils import layershift, get_snac_answer_token
import librosa


class SpeechDatasetJsonl(torch.utils.data.Dataset):
    
    def __init__(self,
                 dataset_config,
                 tokenizer=None,
                 split='train',
                 ):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        
        # self.data_list = contents
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.prompt_template = "USER: {}\n ASSISTANT: "
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        self.manifest_format = dataset_config.get("manifest_format", "datasets")
        self.seed = dataset_config.get("seed", 42)
        self.split_size = dataset_config.get("split_size", 0.1)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 
        assert self.manifest_format in ["datasets", "jsonl"], "manifest_format must be one of [datasets, jsonl]"

        # vocab config
        self.vocab_config = dataset_config.get("vocab_config", None)
        self.text_vocabsize = self.vocab_config.text_vocabsize
        self.text_specialtokens = self.vocab_config.text_specialtokens
        self.audio_vocabsize = self.vocab_config.audio_vocabsize
        self.audio_specialtokens = self.vocab_config.audio_specialtokens
        self.padded_text_vocabsize = self.vocab_config.padded_text_vocabsize
        self.padded_audio_vocabsize = self.vocab_config.padded_audio_vocabsize
        self.total_vocabsize = self.vocab_config.total_vocabsize
        self._eot = self.vocab_config.eot
        self._pad_t = self.vocab_config.pad_t
        self._input_t = self.vocab_config.input_t
        self._answer_t = self.vocab_config.answer_t
        self._asr = self.vocab_config.asr
        self._eoa = self.vocab_config.eoa
        self._pad_a = self.vocab_config.pad_a
        self._input_a = self.vocab_config.input_a
        self._answer_a = self.vocab_config.answer_a
        self._split = self.vocab_config.split
        self.code_layer = self.vocab_config.code_layer

        self.special_token_a = self._answer_a
        self.special_token_t = self._answer_t

        # task config 
        self.task_type = dataset_config.get("task_type", "s2s")

        self.data_list = []

        # TODO: design a better way to load data
        if self.manifest_format == "datasets":
            from datasets import load_dataset, load_from_disk
            if dataset_config.load_from_cache_file:       
                ds = load_dataset(dataset_config.train_data_path)       # load_from huggingface datasets
            else:
                ds = load_from_disk(dataset_config.train_data_path)   # load_from local disk
            train_val_split = ds['train'].train_test_split(test_size=self.split_size, seed=self.seed)
            if split == "train":
                self.data_list = train_val_split['train']
            else:
                self.data_list = train_val_split['test']
        else:
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

    # extract audio feature (raw waveform or mel spectrogram) from audio file
    # NOTE: here datasets format is just for VoiceAssistant-400K dataset
    def extract_audio_feature(self, audio_path):
        # audio path is a dictionary, resample the audio to 16kHz
        if self.manifest_format == "datasets" and isinstance(audio_path, dict):
            audio_raw = audio_path['array']
            audio_raw_sr = audio_path['sampling_rate']
            audio_raw = librosa.resample(audio_raw, orig_sr=audio_raw_sr, target_sr=16000).astype(np.float32)
        elif self.manifest_format == "datasets" and isinstance(audio_path, str):
            audio_res, audio_length = get_snac_answer_token(audio_path)
            return audio_res, audio_length
        else:
            audio_raw = whisper.load_audio(audio_path)
            
        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw)
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320
            audio_length = audio_length // 5
            audio_res = audio_raw
        elif self.input_type == "mel":
            audio_raw = whisper.pad_or_trim(audio_raw)
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2
            audio_length = audio_length // 5
            audio_res = audio_mel

        return audio_res, audio_length

    def get_input_ids(self, length, special_token_a, special_token_t):
        input_ids = []
        for i in range(self.code_layer):
            input_ids_item = []
            input_ids_item.append(layershift(self._input_a, i))
            input_ids_item += [layershift(self._pad_a, i)] * length
            input_ids_item += [(layershift(self._eoa, i)), layershift(special_token_a, i)]
            input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
        input_id_T = torch.tensor([self._input_t] + [self._pad_t] * length + [self._eot, special_token_t])
        input_ids.append(input_id_T.unsqueeze(0))
        return input_ids

    def get_answer_ids(self, length):
        answer_ids = []
        for i in range(self.code_layer):
            answer_ids_item = []
            answer_ids_item += [layershift(self._pad_a, i)] * length
            answer_ids.append(torch.tensor(answer_ids_item).unsqueeze(0))
        answer_id_T = torch.tensor([self._pad_t] * length)
        answer_ids.append(answer_id_T.unsqueeze(0))
        return answer_ids
    
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        task_type = self.task_type  # TODO: this type could be determined by sampling strategy instead of hard-coded
        audio_mel = None
        example_ids = None
        key = None
        audio_length = 0
        target_audio_length = 0

        if self.manifest_format == "datasets":
            source_audio = data_dict.get("question_audio", None)
            target_audio = data_dict.get("answer_snac", None)
            source_text = data_dict.get("question", None)
            target_text = data_dict.get("answer", None)
            if source_audio is not None:
                key = source_audio['path']
        elif self.manifest_format == "jsonl":
            source_audio = data_dict.get("source_wav", None)
            target_audio = data_dict.get("target_wav", None)
            source_text = data_dict.get("source_text", None)
            target_text = data_dict.get("target_text", None)
            key = data_dict.get("key", None)
        else:
            raise ValueError("manifest_format must be one of [datasets, jsonl]")

        if task_type == "s2s" or task_type == "asr":
            audio_mel, audio_length = self.extract_audio_feature(source_audio)
        
        if task_type == "s2s" or task_type == "tts":
            target_audio, target_audio_length = self.extract_audio_feature(target_audio)

        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio

        prompt = self.prompt
        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)

        if task_type == "s2s" or task_type == "asr":
            example_ids = self.get_input_ids(audio_length + prompt_length, self.special_token_a, self.special_token_t)
            text_layer = example_ids[self.code_layer]
            text_layer = torch.cat((text_layer[:,:audio_length + 1], prompt_ids.unsqueeze(0), text_layer[:,-2:]), dim=1)     # <bos> <audio> <prompt> <eos> <task>
            example_ids[self.code_layer] = text_layer
        elif task_type == "tts":
            target_text_ids = self.tokenizer.encode(target_text)
            target_text_length = len(target_text_ids)
            target_text_ids = torch.tensor(target_text_ids, dtype=torch.int64)
            example_ids = self.get_input_ids(target_text_length + prompt_length, self.special_token_a, self.special_token_t) # <bos> <text> <prompt> <eos> <task>
            text_layer = example_ids[self.code_layer]
            text_layer = torch.cat((text_layer[:,:1], target_text_ids.unsqueeze(0), prompt_ids.unsqueeze(0), text_layer[:,-2:]), dim=1)
            example_ids[self.code_layer] = text_layer
        else:
            raise ValueError(f"task_type {task_type} is not supported")

        input_length = audio_length
        if task_type == "tts":
            input_length = target_text_length   # NOTE: when task_type is tts, input_length is target_text_length

        if self.inference_mode:
            example_mask = example_ids[0][0].ge(-1)  # [True,True]
            example_ids = torch.stack(example_ids).squeeze()

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio_mel": audio_mel,
                "input_length": input_length,
                "audio_length": audio_length,
                "target_audio": target_audio,
                "target_audio_length": target_audio_length,
                "key": key,
                "source_text": source_text,
                "target_text": target_text,
                "prompt_length": prompt_length,
                "task_type": task_type,
            }

        answer_text = self.answer_template.format(target_text)
        answer_text_ids = self.tokenizer.encode(answer_text)  # [answer]
        answer_text_ids.append(self._eot) # [answer,eos]
        answer_text_ids = torch.tensor(answer_text_ids, dtype=torch.int64)

        answer_length = max(len(answer_text_ids), target_audio_length)
        answer_ids = self.get_answer_ids(answer_length)                 # NOTE: somtimes answer_text_ids is longer than target_audio_length 
        answer_ids[self.code_layer] = torch.cat((answer_text_ids.unsqueeze(0), answer_ids[self.code_layer][:,len(answer_text_ids):]),dim=1)     # [answer_text,eos]
        text_padding_length = target_audio_length - len(answer_text_ids)

        labels_ids = copy.deepcopy(answer_ids)
        ori_example_ids = copy.deepcopy(example_ids)
        
        if target_audio is not None:    
            for i in range(self.code_layer):
                labels_ids[i] = torch.cat((target_audio[i].unsqueeze(0), answer_ids[i][:,target_audio_length:]), dim=1)
                answer_ids[i] = torch.cat((layershift(target_audio[i], i).unsqueeze(0), labels_ids[i][:,target_audio_length:]), dim=1)

        for i in range(self.code_layer + 1):
            example_ids[i] = torch.cat((ori_example_ids[i], answer_ids[i]), dim=1)  # [audio,prompt,answer,eos]
            labels_ids[i] = torch.cat((ori_example_ids[i], labels_ids[i]), dim=1)

        example_ids = torch.stack(example_ids).squeeze()
        labels_ids = torch.stack(labels_ids).squeeze()
        labels_ids[:,:input_length + prompt_length + 3] = -1  # [-1,-1,answer,eos]; NOTE: here 3 include <bos> <eos> <ans_t>

        if text_padding_length > 0:
            labels_ids[self.code_layer,-text_padding_length:] = -1   # [-1,-1,answer_text,eos,-1]
        else:
            audio_padding_length = -text_padding_length
            labels_ids[:self.code_layer,-audio_padding_length:] = -1  # [-1,-1,answer_text,eos,-1]
        
        example_mask = example_ids[0].ge(-1)  # [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            "audio_mel": audio_mel,
            "input_length": input_length,
            "audio_length": audio_length,
            "target_audio": target_audio,
            "target_audio_length": target_audio_length,
            "key": key,
            "source_text": source_text,
            "target_text": target_text,
            "prompt_length": prompt_length,
            "task_type": task_type,
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
        
    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    # sequence = torch.nn.functional.pad(sequence, (0, padding_length)) FIXME: this is wrong before in SLAM-LLM
                    padding_tensor = torch.full((sequence.size(0), padding_length), padding_idx, dtype=sequence.dtype)
                    if padding_side == "left":
                        sequence = torch.cat((padding_tensor, sequence), dim=1)
                    else:
                        sequence = torch.cat((sequence, padding_tensor), dim=1)
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        assert samples is not None 
        input_prompt_lengths = [s["input_length"] + s['prompt_length'] + 3 for s in samples] #[319, 319, 319, 319]
        input_answer_lengths = [len(s["input_ids"][0]) - s["input_length"] - s['prompt_length'] - 3 for s in samples]  #[264, 99, 206, 141]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        input_length = torch.tensor([s["input_length"] for s in samples])
        audio_length = torch.tensor([s["audio_length"] for s in samples])
        audio_raw = None
        audio_mask = None
        audio_mel = None
        audio_mel_post_mask = None

        if self.input_type == "raw" and self.task_type in ["s2s", "asr"]:
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                     for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
        elif self.input_type == "mel" and self.task_type in ["s2s", "asr"]:
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                  for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), (audio_mel_max_length + 1) // 2) # ad-hoc for whisper for 2x downsample from mel to feats
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :(sample['audio_mel'].shape[0] + 1) // 2] = 1
    
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index] + 1 # +1 for <bos>
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        task_type = [s['task_type'] for s in samples]

        if self.inference_mode:
            keys = [s['key'] for s in samples]
            target_text = [s['target_text'] for s in samples]
            source_text = [s['source_text'] for s in samples]
            target_audio = [s['target_audio'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw,
                "audio_mask": audio_mask,
                "input_length": input_length,
                "audio_length": audio_length,
                "audio_mel": audio_mel,
                "audio_mel_post_mask": audio_mel_post_mask,
                "modality_mask": modality_mask,
                "keys": keys,
                "target_texts": target_text,
                "source_texts": source_text,
                "target_audio": target_audio,
                "task_types": task_type
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "input_length": input_length,
            "audio_length": audio_length,
            "audio_mel": audio_mel,
            "audio_mel_post_mask": audio_mel_post_mask,
            "modality_mask": modality_mask,
            "task_types": task_type
        }


def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)
    return dataset
