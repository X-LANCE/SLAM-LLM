from fairseq.checkpoint_utils import load_model_ensemble_and_task  
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .moco_visual_frontend import MoCoVisualFrontend
from .utils import PositionalEncoding, conv1dLayers, outputConv, MaskedLayerNorm, generate_square_subsequent_mask

from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel


class AVNet(nn.Module):  

    def __init__(self, model_config):
        super(AVNet, self).__init__()
        
        self.modal =  model_config.modal
        self.numClasses = model_config.CHAR_NUM_CLASSES
        self.reqInpLen = model_config.MAIN_REQ_INPUT_LENGTH
        self.dModel= model_config.DMODEL  #!!!
        self.nHeads = model_config.TX_ATTENTION_HEADS
        self.numLayers = model_config.TX_NUM_LAYERS
        self.peMaxLen= model_config.PE_MAX_LENGTH
        self.audinSize = model_config.AUDIO_FEATURE_SIZE
        self.vidinSize = model_config.VIDEO_FEATURE_SIZE
        self.fcHiddenSize = model_config.TX_FEEDFORWARD_DIM
        self.dropout = model_config.TX_DROPOUT
        self.MoCofile = model_config.MOCO_FRONTEND_FILE
        self.W2Vfile =  model_config.WAV2VEC_FILE

        # A & V Modal
        tx_norm = nn.LayerNorm(self.dModel)
        self.maskedLayerNorm = MaskedLayerNorm()
        if self.modal == "AV":
            self.ModalityNormalization = nn.LayerNorm(self.dModel)
        self.EncoderPositionalEncoding = PositionalEncoding(dModel=self.dModel, maxLen=self.peMaxLen)  #512,500

        # audio
        if not self.modal == "VO":
            # front-end
            wav2vecModel, cfg, task = load_model_ensemble_and_task([self.W2Vfile], arg_overrides={
                "apply_mask": True,
                "mask_prob": 0.5,
                "mask_channel_prob": 0.25,
                "mask_channel_length": 64,
                "layerdrop": 0.1,
                "activation_dropout": 0.1,
                "feature_grad_mult": 0.0,
            })
            wav2vecModel = wav2vecModel[0]
            wav2vecModel.remove_pretraining_modules()
            self.wav2vecModel = wav2vecModel
            # back-end
            self.audioConv = conv1dLayers(self.maskedLayerNorm, self.audinSize, self.dModel, self.dModel, downsample=True)
            audioEncoderLayer = nn.TransformerEncoderLayer(d_model=self.dModel, nhead=self.nHeads, dim_feedforward=self.fcHiddenSize, dropout=self.dropout)
            self.audioEncoder = nn.TransformerEncoder(audioEncoderLayer, num_layers=self.numLayers, norm=tx_norm)
        else:
            self.wav2vecModel = None   #主要是这三个
            self.audioConv = None
            self.audioEncoder = None

        # visual
        if not self.modal == "AO":
            # front-end
            visualModel = MoCoVisualFrontend(model_config)
            if self.MoCofile is not None:
                visualModel.load_state_dict(torch.load(self.MoCofile, map_location="cpu"), strict=False)
            self.visualModel = visualModel
            # back-end
            self.videoConv = conv1dLayers(self.maskedLayerNorm, self.vidinSize, self.dModel, self.dModel)
            videoEncoderLayer = nn.TransformerEncoderLayer(d_model=self.dModel, nhead=self.nHeads, dim_feedforward=self.fcHiddenSize, dropout=self.dropout)
            self.videoEncoder = nn.TransformerEncoder(videoEncoderLayer, num_layers=self.numLayers, norm=tx_norm)
        else:
            self.visualModel = None  #主要是这三个
            self.videoConv = None
            self.videoEncoder = None

        # JointConv for fusion
        if self.modal == "AV":
            self.jointConv = conv1dLayers(self.maskedLayerNorm, 2 * self.dModel, self.dModel, self.dModel)
            jointEncoderLayer = nn.TransformerEncoderLayer(d_model=self.dModel, nhead=self.nHeads, dim_feedforward=self.fcHiddenSize, dropout=self.dropout)
            self.jointEncoder = nn.TransformerEncoder(jointEncoderLayer, num_layers=self.numLayers, norm=tx_norm)

        # self.jointOutputConv = outputConv(self.maskedLayerNorm, self.dModel, self.numClasses)
        # self.decoderPositionalEncoding = PositionalEncoding(dModel=self.dModel, maxLen=self.peMaxLen)
        # self.embed = torch.nn.Sequential(
        #     nn.Embedding(self.numClasses, self.dModel),
        #     self.decoderPositionalEncoding
        # )
        # jointDecoderLayer = nn.TransformerDecoderLayer(d_model=self.dModel, nhead=self.nHeads, dim_feedforward=self.fcHiddenSize, dropout=self.dropout)
        # self.jointAttentionDecoder = nn.TransformerDecoder(jointDecoderLayer, num_layers=self.numLayers, norm=tx_norm)
        # self.jointAttentionOutputConv = outputConv("LN", self.dModel, self.numClasses)

    def forward(self, inputBatch, maskw2v):
        audioBatch, audMask, videoBatch, vidLen = inputBatch  #torch.Size([2, 32480]),torch.Size([2, 32480]),torch.Size([2, 52, 1, 112, 112]),[52,47]  # audMask尾部有一堆true表示mask，其余都是false
        if not self.modal == "VO":
            try:
                result = self.wav2vecModel.extract_features(audioBatch, padding_mask=audMask, mask=maskw2v)  #new_version  这一步/320 并向下取整 
                audioBatch,audMask =result["x"],result["padding_mask"]  #torch.Size([2, 101, 1024]), torch.Size([2, 101])   #形状变了 所以还得跟形状保持一致
                if audMask==None:
                    audMask= torch.full( (audioBatch.shape[0], audioBatch.shape[1]), False, device=audioBatch.device ) #TODO

                audLen = torch.sum(~audMask, dim=1)  #tensor([101,  90], device='cuda:0')
            except Exception as e:
                print(e)
                print(audioBatch.shape)
                print(audMask)

        else:
            audLen = None

        if not self.modal == "AO":
            videoBatch = videoBatch.transpose(1, 2)
            videoBatch = self.visualModel(videoBatch, vidLen.long())  #torch.Size([99, 2048])
            videoBatch = list(torch.split(videoBatch, vidLen.tolist(), dim=0))  #拆成一个list [(52,2048), (47, 2048)]

        #print(audioBatch.shape,audLen,videoBatch[0].shape,videoBatch[1].shape, videoBatch[2].shape,videoBatch[3].shape,vidLen)
        audioBatch, videoBatch, inputLenBatch, mask = self.makePadding(audioBatch, audLen, videoBatch, vidLen)  #[2, 160, 1024], torch.Size([2, 80, 2048]), tensor([80, 80],  (2,80) #这一步比较关键
        #print( max(max(vidLen).item()*2, max(audLen).item()), audioBatch.shape, videoBatch.shape, inputLenBatch, mask.shape)
        if isinstance(self.maskedLayerNorm, MaskedLayerNorm):
            self.maskedLayerNorm.SetMaskandLength(mask, inputLenBatch)

        if not self.modal == "VO":
            audioBatch = audioBatch.transpose(1, 2)  #? 
            audioBatch = self.audioConv(audioBatch) #[2, 1024, 80]
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            audioBatch = self.EncoderPositionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask)  #[80,2,1024]

        if not self.modal == "AO":
            videoBatch = videoBatch.transpose(1, 2)
            videoBatch = self.videoConv(videoBatch)  #[2, 1024, 80]
            videoBatch = videoBatch.transpose(1, 2).transpose(0, 1)
            videoBatch = self.EncoderPositionalEncoding(videoBatch)
            videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask)  #[80, 2, 1024]

        if self.modal == "AO":
            jointBatch = audioBatch
        elif self.modal == "VO":
            jointBatch = videoBatch
        else:
            jointBatch = torch.cat([self.ModalityNormalization(audioBatch), self.ModalityNormalization(videoBatch)], dim=2)  #torch.Size([80, 2, 2048])
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2) #(2,2048,80)
            jointBatch = self.jointConv(jointBatch) #(2,1024,80)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
            jointBatch = self.EncoderPositionalEncoding(jointBatch)
            jointBatch = self.jointEncoder(jointBatch, src_key_padding_mask=mask) #[80, 2, 1024]

        jointBatch = jointBatch.transpose(0, 1)  #(2,129,1024)  #new
        return jointBatch, inputLenBatch, mask  #[80, 2, 1024], [80,80], [2,80] mask全是false


    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask

    def makePadding(self, audioBatch, audLen, videoBatch, vidLen):
        if self.modal == "AO":
            audPadding = audLen % 2
            mask = (audPadding + audLen) > 2 * self.reqInpLen
            audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)
            audLeftPadding = torch.floor(torch.div(audPadding, 2)).int()
            audRightPadding = torch.ceil(torch.div(audPadding, 2)).int()

            audioBatch = audioBatch.unsqueeze(1).unsqueeze(1)
            audioBatch = list(audioBatch)
            for i, _ in enumerate(audioBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding[i], audRightPadding[i]))
                audioBatch[i] = pad(audioBatch[i][:, :, :audLen[i]]).squeeze(0).squeeze(0)

            audioBatch = pad_sequence(audioBatch, batch_first=True)
            inputLenBatch = ((audLen + audPadding) // 2).long()
            mask = self.makeMaskfromLength([audioBatch.shape[0]] + [audioBatch.shape[1] // 2], inputLenBatch, audioBatch.device)

        elif self.modal == "VO":
            vidPadding = torch.zeros(len(videoBatch)).long().to(vidLen.device)

            mask = (vidPadding + vidLen) > self.reqInpLen
            vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen)

            vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int()
            vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int()

            for i, _ in enumerate(videoBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
                videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

            videoBatch = pad_sequence(videoBatch, batch_first=True)
            inputLenBatch = (vidLen + vidPadding).long()
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, videoBatch.device)

        else:
            dismatch = audLen - 2 * vidLen #tensor([0, 1, 0, 2], device='cuda:0')
            vidPadding = torch.ceil(torch.div(dismatch, 2)).int()   #tensor([0.0000, 0.5000, 0.0000, 1.0000], device='cuda:0')  tensor([0, 1, 0, 1], device='cuda:0', dtype=torch.int32)
            vidPadding = vidPadding * (vidPadding > 0)  #tensor([0, 1, 0, 1], device='cuda:0', dtype=torch.int32)
            audPadding = 2 * vidPadding - dismatch  #tensor([0, 1, 0, 0], device='cuda:0')

            mask = (vidPadding + vidLen) > self.reqInpLen   #80  tensor([False,  True,  True,  True], device='cuda:0')
            vidPadding = mask * vidPadding + (~mask) * (self.reqInpLen - vidLen) #tensor([21,  1,  0,  1], device='cuda:0', dtype=torch.int32)
            mask = (audPadding + audLen) > 2 * self.reqInpLen  #tensor([False,  True,  True,  True], device='cuda:0')
            audPadding = mask * audPadding + (~mask) * (2 * self.reqInpLen - audLen)  #tensor([42,  1,  0,  0], device='cuda:0')

            vidLeftPadding = torch.floor(torch.div(vidPadding, 2)).int() #tensor([10,  0,  0,  0], device='cuda:0', dtype=torch.int32)
            vidRightPadding = torch.ceil(torch.div(vidPadding, 2)).int() #tensor([11,  1,  0,  1], device='cuda:0', dtype=torch.int32)
            audLeftPadding = torch.floor(torch.div(audPadding, 2)).int() #tensor([21,  0,  0,  0], device='cuda:0', dtype=torch.int32)
            audRightPadding = torch.ceil(torch.div(audPadding, 2)).int() #tensor([21,  1,  0,  0], device='cuda:0', dtype=torch.int32)
            # input audioBatch, torch.Size([4, 284, 1024])
            audioBatch = audioBatch.unsqueeze(1).unsqueeze(1) #torch.Size([4, 1, 1, 284, 1024])
            audioBatch = list(audioBatch)  #torch.Size([1, 1, 284, 1024]) 一个list
            for i, _ in enumerate(audioBatch):
                pad = nn.ReplicationPad2d(padding=(0, 0, audLeftPadding[i], audRightPadding[i]))
                audioBatch[i] = pad(audioBatch[i][:, :, :audLen[i]]).squeeze(0).squeeze(0)  #audioBatch[i].shape, torch.Size([1, 1, 284, 1024])
                # print(i,audioBatch[i].shape)
                pad = nn.ReplicationPad2d(padding=(0, 0, vidLeftPadding[i], vidRightPadding[i]))
                videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                # print(i,videoBatch[i].shape)

            audioBatch = pad_sequence(audioBatch, batch_first=True)
            videoBatch = pad_sequence(videoBatch, batch_first=True)
            inputLenBatch = (vidLen + vidPadding).long()
            mask = self.makeMaskfromLength(videoBatch.shape[:-1], inputLenBatch, videoBatch.device)

        return audioBatch, videoBatch, inputLenBatch, mask
