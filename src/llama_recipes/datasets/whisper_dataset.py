# import torch
# from torch.utils.data import Dataset
# import whisper
# import kaldiio
# import copy
# import numpy as np
# from tqdm import tqdm

# import logging
# logger = logging.getLogger(__name__)

# class WhisperDataset(Dataset):
#     def __init__(self, dataset_config, model_config, tokenizer=None, split='train',):
#         super().__init__()
#         self.data_list = []
#         self.label_list = []
#         self.key_list = []
#         self.infer_list=[]
#         self.line_name_list =[]
#         self.name_list=[]

#         if split == "train":
#             pass

#         elif split == "val":
#             with open(dataset_config.dev_scp_file_path + "2/giga_ner_wsplit.txt",'r') as f:
#                 for line in f:
#                     line = line.strip().split('\t')

#                     self.key_list.append(line[0])
#                     self.data_list.append(line[1])
#                     self.label_list.append(line[2]) 
#                     self.line_name_list.append(line[3]) 
                          
#         elif split == "test":  # 3188  只有prev用这个 不用ground truth 用解码
#             pass

#         self.model_config = model_config
#         self.dataset_config = dataset_config
#         self.tokenizer = tokenizer
#         self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
#         self.prompt_template = self.dataset_config.prompt
#         self.answer_template = "{}"
#         self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
#         self.inference_mode = dataset_config.get("inference_mode", False)
#         self.split = split

#     def __getitem__(self, index):
#         wav_path = self.data_list[index]
#         audio_raw = whisper.load_audio(wav_path) #(35600,)

#         target = self.label_list[index] #'KIM WAS NOT DOWN WITH THE CRITIQUE'
#         key = self.key_list[index] #'1012-133424-0005'

#         audio_raw = whisper.pad_or_trim(audio_raw)  #torch.Size([480000])
#         audio_mel = whisper.log_mel_spectrogram(audio_raw,128).permute(1, 0)   # 128 torch.Size([3000, 128])

#         return {
#             'audio_mel': audio_mel,
#             'key': key,
#             'target': target,
#             # 'ocr':ocr,
#             # "previous_sentence":previous_sentence
#         }             


#     def collator(self, samples):
#         assert samples is not None

#         audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
#         audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0) for s in samples])
        
#         keys = [s['key'] for s in samples]
#         targets = [s['target'] for s in samples]
#         # ocrs = [s['ocr'] for s in samples]
#         # previous_sentences = [s['previous_sentence'] for s in samples]


#         return {
#             'audio_mel': audio_mel,
#             'keys': keys,
#             'targets': targets,
#             # 'ocrs': ocrs,
#             # 'previous_sentences' : previous_sentences
#         }


#     def pad(self, sequence, max_length, padding_idx=0):
#         if isinstance(sequence, (int, list, tuple)):
#             if len(sequence) < max_length:
#                 sequence = sequence + [padding_idx] * (max_length - len(sequence))
#             else:
#                 sequence = sequence[:max_length]
#         elif isinstance(sequence, torch.Tensor):
#             if len(sequence) < max_length:
#                 sequence = torch.cat(
#                     (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
#             else:
#                 sequence = sequence[:max_length]
#         else:
#             raise Exception("Type mismatch during padding!")
#         return sequence


#     def __len__(self):
#         return len(self.data_list)

# def get_audio_dataset(dataset_config, model_config, tokenizer, split):
#     dataset = WhisperDataset(dataset_config, model_config, tokenizer, split)
#     return dataset



   