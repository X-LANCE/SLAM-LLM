import torch
import torch.nn as nn
import torchvision.models as models
from config import args


class VisualEncoder(nn.Module):
    # def __init__(self, dModel=args["FRONTEND_DMODEL"], nClasses=args["WORD_NUM_CLASSES"], frameLen=args["FRAME_LENGTH"],
    #              vidfeaturedim=args["VIDEO_FEATURE_SIZE"]):
    def __init__(self, model_config):

        super(VisualEncoder, self).__init__()
        self.dModel = model_config.FRONTEND_DMODEL
        self.nClasses = model_config.WORD_NUM_CLASSES
        self.frameLen = model_config.FRAME_LENGTH
        self.vidfeaturedim = model_config.VIDEO_FEATURE_SIZE


        # Conv3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # moco
        MoCoModel = models.__dict__['resnet50']()  #就当搞了个ResNet
        MoCoModel.fc = nn.Identity()
        MoCoModel.conv1 = nn.Identity()
        MoCoModel.bn1 = nn.Identity()
        MoCoModel.relu = nn.Identity()
        MoCoModel.maxpool = nn.Identity()  #有点意思
        self.MoCoModel = MoCoModel

        self.MoCoModel.load_state_dict(torch.load(MoCofile, map_location="cpu"), strict=False)
        
        
        # AV
        self.peMaxLen = model_config.PE_MAX_LENGTH
        tx_norm = nn.LayerNorm(dModel)
        self.maskedLayerNorm = MaskedLayerNorm()
        self.EncoderPositionalEncoding = PositionalEncoding(dModel=self.dModel, maxLen= self.peMaxLen)  #512,500

        # visual backend
        self.nHeads = model_config.X_ATTENTION_HEADS
        self.fcHiddenSize = model_config.TX_FEEDFORWARD_DIM
        self.dropout = model_config.TX_DROPOUT
        self.num_layers = model_config.TX_NUM_LAYERS

        self.videoConv = conv1dLayers(self.maskedLayerNorm, self.vidfeaturedim, self.dModel, self.dModel)
        videoEncoderLayer = nn.TransformerEncoderLayer(d_model=self.dModel, nhead=self.nHeads, dim_feedforward=self.fcHiddenSize, dropout=self.dropout)
        self.videoEncoder = nn.TransformerEncoder(videoEncoderLayer, num_layers=self.num_layers, norm=tx_norm)

    def forward(self, x, x_len):  # x: 8,1,149,112,112
        x = self.frontend3D(x)  #(8,64,149,28,28)
        x = x.transpose(1, 2) #(8,149,64,28,28)
        mask = torch.zeros(x.shape[:2], device=x.device)  #(8,149)
        mask[(torch.arange(mask.shape[0], device=x.device), x_len - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()  #一堆true false
        x = x[~mask]  #（739,64,28,28）
        x = self.MoCoModel(x)  #（739,2048)
        return x


class MaskedLayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(MaskedLayerNorm, self).__init__()
        self.register_buffer('mask', None, persistent=False)
        self.register_buffer('inputLenBatch', None, persistent=False)
        self.eps = eps

    def SetMaskandLength(self, mask, inputLenBatch):
        self.mask = mask
        self.inputLenBatch = inputLenBatch

    def expand2shape(self, inputBatch, expandedShape):
        return inputBatch.unsqueeze(-1).unsqueeze(-1).expand(expandedShape)

    def forward(self, inputBatch):
        dModel = inputBatch.shape[-1]
        maskBatch = ~self.mask.unsqueeze(-1).expand(inputBatch.shape)

        meanBatch = (inputBatch * maskBatch).sum((1, 2)) / (self.inputLenBatch * dModel)
        stdBatch = ((inputBatch - self.expand2shape(meanBatch, inputBatch.shape)) ** 2 * maskBatch).sum((1, 2))
        stdBatch = stdBatch / (self.inputLenBatch * dModel)

        # Norm the input
        normed = (inputBatch - self.expand2shape(meanBatch, inputBatch.shape)) / \
                 (torch.sqrt(self.expand2shape(stdBatch + self.eps, inputBatch.shape)))
        return normed
