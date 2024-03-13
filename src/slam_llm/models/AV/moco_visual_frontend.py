import torch
import torch.nn as nn
import torchvision.models as models


class MoCoVisualFrontend(nn.Module):
    # def __init__(self, dModel=args["FRONTEND_DMODEL"], nClasses=args["WORD_NUM_CLASSES"], frameLen=args["FRAME_LENGTH"],
    #              vidfeaturedim=args["VIDEO_FEATURE_SIZE"]):
    def __init__(self, model_config):
        
        super(MoCoVisualFrontend, self).__init__()
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
        MoCoModel.maxpool = nn.Identity()  
        self.MoCoModel = MoCoModel

    def forward(self, x, x_len):  # x: 8,1,149,112,112
        x = self.frontend3D(x)  #[2, 64, 52, 28, 28]
        x = x.transpose(1, 2) 
        mask = torch.zeros(x.shape[:2], device=x.device)  #(8,149)
        mask[(torch.arange(mask.shape[0], device=x.device), x_len - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()  #一堆true false
        x = x[~mask]  
        x = self.MoCoModel(x)  # torch.Size([99, 2048])
        return x
