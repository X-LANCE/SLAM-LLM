#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import torch
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.backbones.modules.resnet1d import (
    BasicBlock1D,
    ResNet1D,
)
from torch.nn.functional import pad
import math

class Conv1dResNet(torch.nn.Module):
    def __init__(self, relu_type="swish", a_upsample_ratio=1):
        super().__init__()
        self.a_upsample_ratio = a_upsample_ratio
        self.trunk = ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio,
        )

    def forward(self, xs_pad): #torch.Size([4, 50176, 1])
        """forward.

        :param xs_pad: torch.Tensor, batch of padded input sequences (B, Tmax, idim)
        """
        B, T, C = xs_pad.size()
        #xs_pad = xs_pad[:, : T // 640 * 640, :] #torch.Size([4, 49920, 1]) 修改到向上取整640

        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = self.trunk(xs_pad) #torch.Size([4, 512, 78])
        return xs_pad.transpose(1, 2)
