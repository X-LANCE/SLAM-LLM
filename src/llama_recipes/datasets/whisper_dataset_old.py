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
#         self.num_samples_list = []
#         self.label_list = []
#         self.ocr_list = []
#         self.key_list=[] # for debug

#         if split == "train":
#             # 一次性load 全部数据
#             # logger.info("loading training audio data!")
#             # scp_file = dataset_config.train_scp_file_path + "my_wav.scp" 
#             # with kaldiio.ReadHelper('scp:'+ scp_file) as reader:
#             #     for key, numpy_array in tqdm(reader):
#             #         self.data_list.append( numpy_array[1].astype(np.float32) )
#             # logger.info("Finish loading!")

#             with open(dataset_config.train_scp_file_path + "my_wav.scp",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     self.data_list.append(line[1])
#                     self.key_list.append(line[0])

#             with open(dataset_config.train_scp_file_path + "utt2num_samples",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     self.num_samples_list.append(int(line[1]))

#             with open(dataset_config.train_scp_file_path + "text",'r') as f:
#                 for line in f:
#                     line = line.strip().split(' ',1)
#                     if len(line) == 1:
#                         self.label_list.append(None)
#                     else:
#                         if dataset_config.lower:
#                             self.label_list.append(line[1].lower())
#                         else:
#                             self.label_list.append(line[1])

#             # with open(dataset_config.train_scp_file_path + "my_ocr_text_type2",'r') as f:
#             #     for line in f:
#             #         line = line.strip().split(' ',1)
#             #         if len(line) == 1:
#             #             self.ocr_list.append(None)
#             #         else:
#             #             self.ocr_list.append(line[1])
      
#             with open(dataset_config.train_scp_file_path + "hot_related/ocr_1gram_top50_mmr070_hotwords_list",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     if len(line) == 1:
#                         self.ocr_list.append(None)
#                     else:
#                         line = line[1]
#                         line = line.split('$')
#                         line = " ".join(line)

#                         if dataset_config.lower:
#                             self.ocr_list.append(line.lower())
#                         else:
#                             self.ocr_list.append(line)


#         elif split == "val":
#             # 一次性load 全部数据
#             # logger.info("loading validation audio data!")
#             # scp_file = dataset_config.dev_scp_file_path + "my_wav.scp" 
#             # with kaldiio.ReadHelper('scp:'+ scp_file) as reader:
#             #     for key, numpy_array in tqdm(reader):
#             #         self.data_list.append( numpy_array[1].astype(np.float32) )
#             # logger.info("Finish loading!")

#             with open(dataset_config.dev_scp_file_path + "my_wav.scp",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     self.data_list.append(line[1])
#                     self.key_list.append(line[0])
            

#             with open(dataset_config.dev_scp_file_path + "utt2num_samples",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     self.num_samples_list.append(int(line[1]))

#             with open(dataset_config.dev_scp_file_path + "text",'r') as f:
#                 for line in f:
#                     line = line.strip().split(' ',1)
#                     if len(line) == 1:
#                         self.label_list.append(None)
#                     else:
#                         if dataset_config.lower:
#                             self.label_list.append(line[1].lower())
#                         else:
#                             self.label_list.append(line[1])

#             # with open(dataset_config.dev_scp_file_path + "ocr_text_type2",'r') as f:
#             #     for line in f:
#             #         line = line.strip().split(' ',1)
#             #         if len(line) == 1:
#             #             self.ocr_list.append(None)
#             #         else:
#             #             self.ocr_list.append(line[1])

#             with open(dataset_config.dev_scp_file_path + "hot_related/ocr_1gram_top50_mmr070_hotwords_list",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     if len(line) == 1:
#                         self.ocr_list.append(None)
#                     else:
#                         line = line[1]
#                         line = line.split('$')
#                         line = " ".join(line)

#                         if dataset_config.lower:
#                             self.ocr_list.append(line.lower())
#                         else:
#                             self.ocr_list.append(line)

#         elif split == "test":  # 3188  只有prev用这个 不用ground truth 用解码
#             with open(dataset_config.test_scp_file_path + "my_wav.scp",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     self.data_list.append(line[1])      

#             with open(dataset_config.test_scp_file_path + "my_wav.scp",'r') as f:
#                 for line in f:
#                     line = line.strip().split()
#                     self.num_samples_list.append(int(line[1]))

#             with open(dataset_config.test_scp_file_path + "text",'r') as f:
#                 for line in f:
#                     line = line.strip().split(' ',1)
#                     if len(line) == 1:
#                         self.label_list.append(None)
#                     else:
#                         self.label_list.append(line[1])

#             with open(dataset_config.test_scp_file_path + "ocr_text_type2",'r') as f:
#                 for line in f:
#                     line = line.strip().split(' ',1)
#                     if len(line) == 1:
#                         self.ocr_list.append(None)
#                     else:
#                         self.ocr_list.append(line[1])

#         self.model_config = model_config
#         self.dataset_config = dataset_config
#         self.tokenizer = tokenizer
#         self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
#         self.prompt_template1 = "USER: {}\n ASSISTANT:"
#         #self.prompt_template2 = "USER: Transcribe speech to text. The speech is related to a slide, which contains key information. The text from the slide is \"{}\". Please use the text to enhance the accuracy of the ASR task.\n ASSISTANT:"
#         self.prompt_template2 = "USER: Transcribe speech to text. Some hotwords on the slide might help. The hotwords are \"{}\". \n ASSISTANT:"
#         self.prompt_template2= self.dataset_config.prompt
#         self.answer_template = "{}"
#         self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
#         self.inference_mode = dataset_config.get("inference_mode", False)

#         self.prev_prompt_template = "USER: Transcribe speech to text. Its previous sentence is \"{}\". \n ASSISTANT:"
#         self.prev_keywords_prompt_template="USER: Transcribe speech to text. Its previous sentence is \"{}\". Use hotwords in ppt to improve speech recognition accuracy. But if the hotwords are irrelevant, just ignore them. The hotwords are \"{}\". \n ASSISTANT:"

#     def __getitem__(self, index):
#         ark_path = self.data_list[index]
#         numpy_array = kaldiio.load_mat(ark_path)  #???
#         audio_raw = numpy_array[1].astype(np.float32) #(12320,)  (103680,)
#         #audio_raw = self.data_list[index]
#         # audio_raw = torch.from_numpy(audio_raw).float()

#         num_samples = self.num_samples_list[index]  #12320
#         assert(audio_raw.shape[0] == num_samples)
        
#         ocr = self.ocr_list[index]
#         target = self.label_list[index]
#         key = self.key_list[index]


#         audio_raw = whisper.pad_or_trim(audio_raw)  #torch.Size([480000])
#         audio_mel = whisper.log_mel_spectrogram(audio_raw,128).permute(1, 0)   # 128


#         previous_sentence = None
#         if index!=0:
#             prev_key = self.key_list[index-1]
#             prev_prefix = prev_key.rsplit('+',1)[0]
#             prefix = key.rsplit('+',1)[0]
#             if prev_prefix == prefix:
#                 prev_number = int(prev_key.rsplit('+',1)[1])
#                 number = int(key.rsplit('+',1)[1])
#                 #if number-prev_number<=6:
#                 if number-prev_number <= self.dataset_config.prev_bar:
#                     previous_sentence = self.label_list[index-1]
    

#         return {
#             'audio_mel': audio_mel,
#             'key': key,
#             'target': target,
#             'ocr':ocr,
#             "previous_sentence":previous_sentence
#         }             


#     def collator(self, samples):
#         assert samples is not None

#         audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
#         audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0) for s in samples])
        
#         keys = [s['key'] for s in samples]
#         targets = [s['target'] for s in samples]
#         ocrs = [s['ocr'] for s in samples]
#         previous_sentences = [s['previous_sentence'] for s in samples]


#         return {
#             'audio_mel': audio_mel,
#             'keys': keys,
#             'targets': targets,
#             'ocrs': ocrs,
#             'previous_sentences' : previous_sentences
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



   