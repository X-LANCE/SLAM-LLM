#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn as nn
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.backbones.modules.resnet import BasicBlock, ResNet
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.convolution import Swish


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Conv3dResNet(torch.nn.Module):
    """Conv3dResNet module"""

    def __init__(self, backbone_type="resnet", relu_type="swish"):
        """__init__.

        :param backbone_type: str, the type of a visual front-end.
        :param relu_type: str, activation function used in an audio front-end.
        """
        super(Conv3dResNet, self).__init__()
        self.frontend_nout = 64
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1, self.frontend_nout, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False
            ),
            nn.BatchNorm3d(self.frontend_nout),
            Swish(),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)),
        )

    def forward(self, xs_pad):
        xs_pad = xs_pad.transpose(1, 2)  # [B, T, C, H, W] -> [B, C, T, H, W]

        B, C, T, H, W = xs_pad.size()
        xs_pad = self.frontend3D(xs_pad)  #torch.Size([4, 64, 78, 28, 28])
        Tnew = xs_pad.shape[2] #78
        xs_pad = threeD_to_2D_tensor(xs_pad) #torch.Size([312, 64, 28, 28])
        xs_pad = self.trunk(xs_pad) #torch.Size([312, 512])
        return xs_pad.view(B, Tnew, xs_pad.size(1))
