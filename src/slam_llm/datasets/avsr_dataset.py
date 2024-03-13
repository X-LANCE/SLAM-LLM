import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import random
import torch
import math
import copy

import cv2 as cv
from torch.nn.utils.rnn import pad_sequence

import logging
logger = logging.getLogger(__name__)

from slam_llm.utils.compute_utils import calculate_output_length_1d
import torch.nn as nn

class AVSRDataset(Dataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()

        self.tokenizer = tokenizer
        self.modal = dataset_config.modal
        self.dataset = split                        #train|val|test
        self.data_path = dataset_config.data_path
        self.h5file = dataset_config.h5file   
        self.noiseFile = dataset_config.noiseFile
        self.noiseSNR =  dataset_config.noiseSNR
        self.noiseProb = dataset_config.noiseProb
        self.stepSize = dataset_config.stepSize  #16384
        self.charToIx = dataset_config.charToIx
        self.pretrain_subset = dataset_config.pretrain_subset
        self.train_subset = dataset_config.train_subset
        self.valid_subset = dataset_config.valid_subset
        self.test_subset= dataset_config.test_subset

        if self.dataset == "train": 
            pretrain_dir = self.data_path + self.pretrain_subset  # "LRS3/pretrain.txt"
            train_dir = self.data_path + self.train_subset        # "LRS3/train.txt"

            with open(pretrain_dir, "r") as f:
                lines = f.readlines()
                pretrain_datalist = [self.data_path + line.strip()[3:] for line in lines] #长度：118516

            with open(train_dir, "r") as f:
                lines = f.readlines()
                train_datalist = [self.data_path + line.strip()[3:] for line in lines] #长度:31662

            self.datalist = pretrain_datalist+ train_datalist
            lrs3Aug=True

        elif self.dataset == "val":
            val_dir = self.data_path +  self.valid_subset  # "LRS3/val.txt"
            with open(val_dir, "r") as f:
                lines = f.readlines()
                val_datalist = [self.data_path + line.strip()[3:] for line in lines]
            self.datalist = val_datalist
            lrs3Aug=False

        else:
            test_dir = self.data_path +  self.test_subset # "LRS3/test.txt"
            with open(test_dir, "r") as f:
                lines = f.readlines()
                test_datalist = [self.data_path + line.strip()[3:] for line in lines]
            self.datalist = test_datalist
            lrs3Aug=False

        with h5py.File(self.noiseFile, "r") as f:  #{'noiseFile': '/home/xcpan/LRS2/mvlrs_v1/Noise.h5', 'noiseProb': 0.25, 'noiseSNR': 5}
            self.noise = f["noise"][0]  #ndarray:57600000

        if lrs3Aug:
            self.transform = transforms.Compose([
                ToTensor(),
                RandomCrop(112),
                RandomHorizontalFlip(0.5),
                Normalize(mean=[0.4161], std=[0.1688])
            ])
        else:
            self.transform = transforms.Compose([
                ToTensor(),
                CenterCrop(112),
                Normalize(mean=[0.4161], std=[0.1688])
            ])

        # LLM new
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.reqInpLen = dataset_config.reqInpLen

    def open_h5(self):
        self.h5 = h5py.File(self.h5file, "r")

    def __getitem__(self, index):  #avsr 是shuffle的dataloader echat好像默认false 没shu  index从0开始
        """
            LRS3 : pretrain 118516  train 31662  val 320   test 1321
            LRS2 : pretrain 96318   train 45839  val 1082  test 1243    142157 = 96318 + 45839 = pretrain + train  143239 = 96318+45839+1082=pretrain+train+val

            index goes from 0 to stepSize-1
            dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            fetch the sample at position 'index' in this randomly selected partition
        """ 

        if not hasattr(self, 'h5'):
            self.open_h5()

        if self.dataset == "train":   #index=610
            base = self.stepSize * np.arange(int(len(self.datalist) / self.stepSize) + 1)   # datalist, 118516 应该全是pretrain的 从pretrain.txt 搞出来的 # stepsize 16384
            ixs = base + index                        # [  0  16384  32768  49152  65536  81920  98304 114688 131072 147456]
            ixs = ixs[ixs < len(self.datalist)]       # [  610  16994  33378  49762  66146  82530  98914 115298]
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)  #以某种方式随机采样  #33378

        if index==99639 or index== 71740 or index==19753 or index==14116 or index==49729 or index==26726:  #dirty data
            index+=1

        # passing the sample files and the target file paths to the prepare function to obtain the input tensors
        targetFile = self.datalist[index] + ".txt"  
        if self.dataset == "val":
            index += 150178             # 原本 142157 
        elif self.dataset == "test":
            index += 150498             # 原本 143239

        if np.random.choice([True, False], p=[self.noiseProb, 1 - self.noiseProb]):
            noise = self.noise
        else:
            noise = None

        if index < 118516:     #原本是96318   查过了 这个数确实是lrs2的那个行数 也就是文件数  原本应该是pretrain处理的 有一部分搞到main处理了 所以没有crop 导致超过500
            #inp, trgtin, trgtout, trgtLen, trgttext  = self.prepare_pretrain_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)
            inp, trgtin, trgtout, trgtLen, target = self.prepare_pretrain_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)
            if inp==0 and trgtin ==0 and  trgtout ==0 and trgtLen==0:
                index+=1
                targetFile = self.datalist[index] + ".txt"
                #inp, trgtin, trgtout, trgtLen, trgttext = self.prepare_pretrain_input(index, self.modal, self.h5, targetFile,self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)  #就只是往后挪了一格 很弱
                inp, trgtin, trgtout, trgtLen,target = self.prepare_pretrain_input(index, self.modal, self.h5, targetFile,self.charToIx, self.transform, noise, self.noiseSNR, (3, 21), 160)  #就只是往后挪了一格 很弱
        else:
            #inp, trgtin, trgtout, trgtLen, trgttext = self.prepare_main_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR)
            inp, trgtin, trgtout, trgtLen,target = self.prepare_main_input(index, self.modal, self.h5, targetFile, self.charToIx, self.transform, noise, self.noiseSNR)   


        # new!
        audio_raw = inp[0]  #cpu torch.Size([48800])
        visual_raw = inp[1]  #cpu torch.Size([77, 1, 112, 112])

        prompt = "Transcribe video to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "

        prompt = self.prompt_template.format(prompt)
        answer = self.answer_template.format(target)

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        #audio_length, visual_length,inputLen = self.calculate_output_length(audio_raw,visual_raw)
        audio_length_pre = self.calculate_output_length(audio_raw,visual_raw)  #video  #tensor(80)
        audio_length = audio_length_pre // 5 # ad-hoc for 5x fc downsample  #tensor(16)
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

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
            # 'audio_mel': audio_mel,   
            'audio_length': audio_length,

            'inp':inp,
            'trgtin': trgtin,
            'trgtout': trgtout,
            'trgtLen':trgtLen,

            'audio_length_pre':audio_length_pre,
        }
                                                  

        #return inp, trgtin, trgtout, trgtLen  #, trgttext   #VO (none,(72,1,112,112) )

    def calculate_output_length(self,audio_raw,visual_raw):
        # 过wav2vec2
        audio_len = audio_raw.shape[0]
        audio_len = math.floor(audio_len/320)  #152

        # visual 没有变
        visual_len= visual_raw.shape[0] #77

        audLen = torch.tensor(audio_len)
        vidLen = torch.tensor(visual_len)

        dismatch = audLen - 2 * vidLen #tensor([0, 1, 0, 2], device='cuda:0')
        vidPadding = torch.ceil(torch.div(dismatch, 2)).int()   #tensor([0.0000, 0.5000, 0.0000, 1.0000], device='cuda:0')  tensor([0, 1, 0, 1], device='cuda:0', dtype=torch.int32)
        vidPadding = vidPadding * (vidPadding > 0)  #tensor([0, 1, 0, 1], device='cuda:0', dtype=torch.int32)
        audPadding = 2 * vidPadding - dismatch  #tensor([0, 1, 0, 0], device='cuda:0')

        mask = (vidPadding + vidLen) > self.reqInpLen   #80  tensor([False,  True,  True,  True], device='cuda:0')
        vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen) #tensor([21,  1,  0,  1], device='cuda:0', dtype=torch.int32)
        mask = (audPadding + audLen) > 2 * self.reqInpLen  #tensor([False,  True,  True,  True], device='cuda:0')
        audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)  #tensor([42,  1,  0,  0], device='cuda:0')

        vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
        vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()
        audLeftPadding = torch.floor(torch.div(audPadding, 2)).int()
        audRightPadding = torch.ceil(torch.div(audPadding, 2)).int()

        audioBatch = torch.randn(audLen,1024) #.to('cuda') # pseudo audio Batch
        videoBatch = torch.randn(vidLen,2048) #.to('cuda') # pseudo audio Batch 
   
        pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding, audRightPadding))
        audioBatch = pad(audioBatch.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding, vidRightPadding))
        videoBatch = pad(videoBatch.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        audio_length, visual_length = audioBatch.shape[0], videoBatch.shape[0]
        inputLen = (vidLen + vidPadding).long()

        #  过卷积层  其实没有变 就用inputLen
        return inputLen
    

    def __len__(self):
        """
            each iteration covers only a random subset of all the training samples whose size is given by the step size   step size的作用在这里 感觉也没什么大用
            this is done only for the pretrain set, while the whole val/test set is considered
        """

        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)

    def collator(self, samples):
        assert samples is not None
        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id)
                                 for s in samples])
        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX)
                              for s in samples])
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False)
                                      for s in samples])

        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, :sample['audio_length']] = 1   #downsample 再/5

        # audio & mask
        if not self.modal == "VO":
            #aud_seq_list = [data[0][0] for data in dataBatch]
            aud_seq_list = [data['inp'][0] for data in samples]
            aud_padding_mask = torch.zeros((len(aud_seq_list), len(max(aud_seq_list, key=len))), dtype=torch.bool)
            for i, seq in enumerate(aud_seq_list):
                aud_padding_mask[i, len(seq):] = True
            aud_seq_list = pad_sequence(aud_seq_list, batch_first=True)  #可以通过设置 batch_first=True 参数来指定输出的tensor中是否将batch维度放在第一维度
        else:
            aud_seq_list = None
            aud_padding_mask = None
        # visual & len
        if not self.modal == "AO":
            #vis_seq_list = pad_sequence([data[0][1] for data in dataBatch], batch_first=True)  #(4,147,1,112,112)   #pad_sequence((none,62,1,112,112))
            vis_seq_list = pad_sequence([data['inp'][1] for data in samples], batch_first=True)  #(4,147,1,112,112)   #pad_sequence((none,62,1,112,112))
            #vis_len = torch.tensor([len(data[0][1]) for data in dataBatch]) #就是这四个句子每一个的长度 tensor([ 62,  62,  97, 147])   #时间帧上pad
            vis_len = torch.tensor([len(data['inp'][1]) for data in samples]) #就是这四个句子每一个的长度 tensor([ 62,  62,  97, 147])   #时间帧上pad

        else:
            vis_seq_list = None
            vis_len = None

        inputBatch = (aud_seq_list, aud_padding_mask, vis_seq_list, vis_len)  #!!!
 
        if self.modal == "AO":
            inputBatch = (inputBatch[0].float(), inputBatch[1], None, None)
        elif self.modal == "VO":
            inputBatch = (None, None, inputBatch[2].float(), inputBatch[3].int())
        else:
            inputBatch = (inputBatch[0].float(), inputBatch[1], inputBatch[2].float(), inputBatch[3].int())
 
        return {
            'input_ids': input_ids,  #torch.Size([4, 114])
            'labels': labels, #torch.Size([4, 114])
            'attention_mask': attention_mask,  #torch.Size([4, 114])
            # 'audio_mel': audio_mel,
            # 'audio_mel_post_mask': audio_mel_post_mask,
            'modality_mask': modality_mask,
    
            "audio": inputBatch[0],  #torch.Size([4, 92800])
            "audio_mask": inputBatch[1],  #torch.Size([4, 92800])
            "visual": inputBatch[2],  #torch.Size([4, 146, 1, 112, 112])
            "vis_len": inputBatch[3],  #torch.Size([4])
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
        else:
            raise Exception("Type mismatch during padding!")
        return sequence


    def prepare_pretrain_input(self,index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR, numWordsRange, maxLength):  #(3,21)  160
        """
        Function to convert the data sample in the pretrain dataset into appropriate tensors.
        """

        try:
            with open(targetFile, "r") as f:
                lines = f.readlines()
        except:
            logger.info("error")
            logger.info(targetFile)
            logger.info(index)
            return 0, 0, 0, 0

        lines = [line.strip() for line in lines]

        trgt = lines[0][7:]

        coun = trgt.count("{")
        for i in range(coun):
            left = trgt.find("{")
            if left != -1:
                right = trgt.find("}")
                trgt = trgt.replace(trgt[left:right + 2], "")

        trgt=trgt.strip()
        words = trgt.split(" ")

        numWords = len(words) // 3
        if numWords < numWordsRange[0]:   #3   #（numwordsRange 是个tuple（3，21）
            numWords = numWordsRange[0]
        elif numWords > numWordsRange[1]:  #21
            numWords = numWordsRange[1]

        while True:
            # if number of words in target is less than the required number of words, consider the whole target
            if len(words) <= numWords:
                trgtNWord = trgt

                # audio file
                if not modal == "VO":
                    audInp = np.array(h5["flac"][index])
                    audInp = (audInp - audInp.mean()) / audInp.std()
                    if noise is not None:
                        pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                        noise = noise[pos:pos + len(audInp)]
                        noise = noise / np.max(np.abs(noise))
                        gain = 10 ** (noiseSNR / 10)
                        noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                        audInp = audInp + noise
                    audInp = torch.from_numpy(audInp)
                else:
                    audInp = None

                # visual file
                if not modal == "AO":
                    try:
                        vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]
                        vidInp = torch.tensor(vidInp).unsqueeze(1)
                        vidInp = transform(vidInp)
                    except:
                        logger.info("error")
                        logger.info(targetFile)
                        logger.info(index)
                        return 0,0,0,0
                else:
                    vidInp = None
            else:
                # make a list of all possible sub-sequences with required number of words in the target
                nWords = [" ".join(words[i:i + numWords])
                        for i in range(len(words) - numWords + 1)]
                nWordLens = np.array(
                    [len(nWord) + 1 for nWord in nWords]).astype(float)

                # choose the sub-sequence for target according to a softmax distribution of the lengths
                # this way longer sub-sequences (which are more diverse) are selected more often while
                # the shorter sub-sequences (which appear more frequently) are not entirely missed out
                ix = np.random.choice(np.arange(len(nWordLens)), p=nWordLens / nWordLens.sum())
                trgtNWord = nWords[ix]

                # reading the start and end times in the video corresponding to the selected sub-sequence
                startTime = float(lines[4 + ix].split(" ")[1])
                endTime = float(lines[4 + ix + numWords - 1].split(" ")[2])

                # audio file
                if not modal == "VO":
                    samplerate = 16000
                    audInp = np.array(h5["flac"][index])  #（81920，）
                    audInp = (audInp - audInp.mean()) / audInp.std()
                    if noise is not None:
                        pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                        noise = noise[pos:pos + len(audInp)]
                        noise = noise / np.max(np.abs(noise))
                        gain = 10 ** (noiseSNR / 10)
                        noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                        audInp = audInp + noise
                    audInp = torch.from_numpy(audInp)
                    audInp = audInp[int(samplerate * startTime):int(samplerate * endTime)]  #！！！！！！！
                else:
                    audInp = None

                # visual file
                if not modal == "AO":
                    videoFPS = 25
                    try:
                        vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)
                        vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]  ##这一句报错x
                        vidInp = torch.tensor(vidInp).unsqueeze(1)
                        vidInp = transform(vidInp)
                        vidInp = vidInp[int(np.floor(videoFPS * startTime)): int(np.ceil(videoFPS * endTime))]
                    except:
                        logger.info("error")
                        logger.info(targetFile)
                        logger.info(index)
                        return 0, 0, 0, 0

                else:
                    vidInp = None

            """
            trgtin = [charToIx[item] for item in trgtNWord]  #trgtNWord: 'POPULATION BY PROVIDING THEM A SAFE SPACE WHERE THESE GIRLS COULD COME AND MEET OTHER GIRLS READ SOME BOOKS PLAY SOME'
            trgtout = [charToIx[item] for item in trgtNWord]
            trgtin.insert(0, charToIx["<EOS>"])
            trgtout.append(charToIx["<EOS>"])
            """

            # 替换成
            trgtin = self.tokenizer.encode(trgtNWord)   #[1, 349, 4590, 13309, 8098, 6770, 13756, 13044, 4214, 6093, 29924, 319, 317, 5098, ...]
            trgtout = self.tokenizer.encode(trgtNWord) 
            trgtin.insert(0, self.tokenizer.eos_token_id ) #[2,xxx]
            trgtout.append(self.tokenizer.eos_token_id ) #[]

            trgtin = np.array(trgtin)
            trgtout = np.array(trgtout)
            trgtLen = len(trgtout)

            inp = (audInp, vidInp)
            trgtin = torch.from_numpy(trgtin)
            trgtout = torch.from_numpy(trgtout)
            trgtLen = torch.tensor(trgtLen)
            inpLen = len(vidInp) if not self.modal == "AO" else len(audInp) / 640
            if inpLen <= maxLength:   #maxlength:160
                break
            elif inpLen > maxLength + 80:
                numWords -= 2
            else:
                numWords -= 1

        return inp, trgtin, trgtout, trgtLen  , trgtNWord

    def prepare_main_input(self, index, modal, h5, targetFile, charToIx, transform, noise, noiseSNR):
        """
        Function to convert the data sample in the main dataset into appropriate tensors.
        """
        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:]  #'SO WE NEED YOU TO HELP US IN OUR REVIVAL CAMPAIGN'  'YOU ARE A HEALER IN A STONE AGE VILLAGE'

            coun = trgt.count("{")
            for i in range(coun):
                left = trgt.find("{")
                if left != -1:
                    right = trgt.find("}")
                    trgt  = trgt .replace(trgt [left:right + 2], "")

        """
        trgtin = [charToIx[item] for item in trgt] #[8, 4, 1, 15, 2, 1, 7, 2, 2, 12, 1, 14, 4, 13, 1, 3, 4, 1, 9, 2, 11,
        trgtin.insert(0, charToIx["<EOS>"])  #[39,8,4,...]
        trgtout = [charToIx[item] for item in trgt]
        trgtout.append(charToIx["<EOS>"])   #[..,39] 在最后面加39
        """

        trgtin = self.tokenizer.encode(trgt) 
        trgtout = self.tokenizer.encode(trgt) 
        trgtin.insert(0, self.tokenizer.eos_token_id )
        trgtout.append(self.tokenizer.eos_token_id )

        trgtin = np.array(trgtin)
        trgtout = np.array(trgtout)
        trgtLen = len(trgtout)  #50

        # audio file
        if not modal == "VO":
            audInp = np.array(h5["flac"][index])  # ndarray(22528,)
            audInp = (audInp - audInp.mean()) / audInp.std()
            if noise is not None:
                pos = np.random.randint(0, len(noise) - len(audInp) + 1)
                noise = noise[pos:pos + len(audInp)]
                noise = noise / np.max(np.abs(noise))
                gain = 10 ** (noiseSNR / 10)
                noise = noise * np.sqrt(np.sum(audInp ** 2) / (gain * np.sum(noise ** 2)))
                audInp = audInp + noise
            audInp = torch.from_numpy(audInp)
        else:
            audInp = None

        # visual file
        if not modal == "AO":
            vidInp = cv.imdecode(h5["png"][index], cv.IMREAD_COLOR)  #(120,2040,3)
            vidInp = np.array(np.split(vidInp, range(120, len(vidInp[0]), 120), axis=1))[:, :, :, 0]  #(17,120,120)
            vidInp = torch.tensor(vidInp).unsqueeze(1)  #(17,1,120,120)
            vidInp = transform(vidInp) #(17,1,112,112)
        else:
            vidInp = None

        inp = (audInp, vidInp)
        trgtin = torch.from_numpy(trgtin)
        trgtout = torch.from_numpy(trgtout)
        trgtLen = torch.tensor(trgtLen)

        return inp, trgtin, trgtout, trgtLen,trgt   #'THE FIRST TIME WHEN IT TOOK ME FIVE MONTHS FROM THE DECISION OF'


def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = AVSRDataset(dataset_config, tokenizer, split)
    return dataset

class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, device='cpu'):
        self.size = size
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        margin = tensor.shape[-1] - self.size
        hcrop = random.randint(0, margin - 1)
        wcrop = random.randint(0, margin - 1)
        tensor = tensor[:, :, hcrop:-(margin - hcrop), wcrop:-(margin - wcrop)]
        return tensor


class CenterCrop:

    def __init__(self, size, device='cpu'):
        self.size = size
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        crop = (tensor.shape[-1] - self.size) // 2
        tensor = tensor[:, :, crop:-crop, crop:-crop]
        return tensor


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        if random.random() < self.p:
            tensor = torch.flip(tensor, dims=(3,))
        return tensor