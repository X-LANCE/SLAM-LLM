import torch
from torch.utils.data import Dataset
import whisper
import kaldiio
import copy
import numpy as np
from tqdm import tqdm


class SlidespeechDataset(Dataset):
    def __init__(self, dataset_config, tokenizer=None, split='train',):
        super().__init__()
        self.data_list = []
        self.num_samples_list = []
        self.label_list = []
        self.ocr_list = []
        self.key_list=[] # for debug
        self.asr_list=[] # not gt

        if split == "train":
            with open(dataset_config.train_scp_file_path + "my_wav.scp",'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.data_list.append(line[1])
                    self.key_list.append(line[0])

            with open(dataset_config.train_scp_file_path + "utt2num_samples",'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.num_samples_list.append(int(line[1]))

            with open(dataset_config.train_scp_file_path + "text",'r') as f:
                for line in f:
                    line = line.strip().split(' ',1)
                    if len(line) == 1:
                        self.label_list.append(None)
                    else:
                        if dataset_config.lower:
                            self.label_list.append(line[1].lower())
                        else:
                            self.label_list.append(line[1])

            with open(dataset_config.train_scp_file_path + "hot_related/ocr_1gram_top50_mmr070_hotwords_list",'r') as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) == 1:
                        self.ocr_list.append(None)
                    else:
                        line = line[1]
                        line = line.split('$')
                        line = " ".join(line)

                        if dataset_config.lower:
                            self.ocr_list.append(line.lower())
                        else:
                            self.ocr_list.append(line)


        elif split == "val": 
            with open(dataset_config.dev_scp_file_path + "my_wav.scp",'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.data_list.append(line[1])
                    self.key_list.append(line[0])
            
            with open(dataset_config.dev_scp_file_path + "utt2num_samples",'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.num_samples_list.append(int(line[1]))

            with open(dataset_config.dev_scp_file_path + "text",'r') as f:
                for line in f:
                    line = line.strip().split(' ',1)
                    if len(line) == 1:
                        self.label_list.append(None)
                    else:
                        if dataset_config.lower:
                            self.label_list.append(line[1].lower())
                        else:
                            self.label_list.append(line[1])

            with open(dataset_config.dev_scp_file_path + "hot_related/ocr_1gram_top50_mmr070_hotwords_list",'r') as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) == 1:
                        self.ocr_list.append(None)
                    else:
                        line = line[1]
                        line = line.split('$')
                        line = " ".join(line)

                        if dataset_config.lower:
                            self.ocr_list.append(line.lower())
                        else:
                            self.ocr_list.append(line)

        elif split == "test":
            with open(dataset_config.test_scp_file_path + "my_wav.scp",'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.data_list.append(line[1])
                    self.key_list.append(line[0])

            with open(dataset_config.test_scp_file_path + "utt2num_samples",'r') as f:
                for line in f:
                    line = line.strip().split()
                    self.num_samples_list.append(int(line[1]))

            with open(dataset_config.test_scp_file_path + "text",'r') as f:
                for line in f:
                    line = line.strip().split(' ',1)
                    if len(line) == 1:
                        self.label_list.append(None)
                    else:
                        if dataset_config.lower:
                            self.label_list.append(line[1].lower())
                        else:
                            self.label_list.append(line[1])

            with open(dataset_config.test_scp_file_path + "hot_related/ocr_1gram_top50_mmr070_hotwords_list",'r') as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) == 1:
                        self.ocr_list.append(None)
                    else:
                        line = line[1]
                        line = line.split('$')
                        line = " ".join(line)

                        if dataset_config.lower:
                            self.ocr_list.append(line.lower())
                        else:
                            self.ocr_list.append(line)



        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.mel_size = dataset_config.get("mel_size", 80) # 80 for whisper large v1 and v2, 128 for large v3
        self.prompt = dataset_config.get("prompt", None)
        self.prompt_template1 = "USER: {}\n ASSISTANT:"
        self.prompt_template2 = "USER: Transcribe speech to text. Use hotwords in ppt to improve speech recognition accuracy. But if the hotwords are irrelevant, just ignore them. The hotwords are \"{}\". \n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.input_type = dataset_config.get("input_type", None)
        assert self.input_type in ["raw", "mel"], "input_type must be one of [raw, mel]" 

    def get_source_len(self, data_dict):
        return data_dict["source_len"]

    def get_target_len(self, data_dict):
    
        return data_dict["target_len"] if "target_len" in data_dict else 0
    
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        ark_path = self.data_list[index]        
        numpy_array = kaldiio.load_mat(ark_path)
        audio_raw = numpy_array[1].astype(np.float32)
        num_samples = self.num_samples_list[index]
        assert(audio_raw.shape[0] == num_samples)
        ocr = self.ocr_list[index]
        target = self.label_list[index]
        key = self.key_list[index]


        if self.input_type == "raw":
            audio_raw = torch.from_numpy(audio_raw).float()
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            audio_length = len(audio_raw) // 320 # ad-hoc for fairseq 320x downsample
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
        elif self.input_type == "mel":
            audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=self.mel_size).permute(1, 0)
            audio_length = (audio_mel.shape[0] + 1) // 2  # ad-hoc for whisper for 2x downsample from mel to feats
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
            # audio_length = calculate_output_length_1d(audio_length, 5, 5, 0) # ad-hoc for 5x cov1d downsample
        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1) # placeholder


        if self.dataset_config.use_ocr == True and ocr != None:
            prompt = self.prompt_template2.format(ocr)
        else:
            prompt = self.prompt_template1.format(self.prompt)
        # if self.dataset_config.task=="keyword_yizhi":
        #     if self.dataset_config.use_ocr == False or ocr == None:
        #         ocr=""
        #     prompt = self.prompt_template2.format(ocr)
        prompt_ids = self.tokenizer.encode(prompt)
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
                'audio_length': audio_length,
                'key': key,
                'target': target,
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
            'audio_length': audio_length,
        }             

    def pad(self, sequence, max_length, padding_idx=0):#
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
    dataset = SlidespeechDataset(dataset_config, tokenizer, split)
    return dataset



   