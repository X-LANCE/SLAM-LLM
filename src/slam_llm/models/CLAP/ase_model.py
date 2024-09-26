#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
from .audio_encoder import AudioEncoder
from .text_encoder import TextEncoder
import torch.nn.functional as F
import copy
from .losses import AudioTextContrastiveLoss, NTXent
from .utils import text_preprocess


class ASE(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)

        # settings for projection layers
        embed_size = config["embed_size"] 
        audio_width = self.audio_encoder.audio_width
        text_width = self.text_encoder.text_width

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_width, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        self.embed_reg = config["embed_regularization"]

        self.atc_loss = AudioTextContrastiveLoss()
        self.arch_version = 0
        self.use_pd = config.get('pd_text_support', None)
        if self.use_pd:  # projection-based decoding for zero-shot aac
            self.text_embeds_support = torch.load(config['pd_text_support']).cuda()  # [N, dim]

        
    def encode_audio(self, audio):
        audio_feats = torch.mean(self.audio_encoder(audio), dim=1)  
        audio_embeds = F.normalize(self.audio_proj(audio_feats), dim=-1)
        if self.use_pd: 
            text_embeds_support = self.text_embeds_support  
            sim = audio_embeds.squeeze(1) @ text_embeds_support.transpose(-1, -2)  # [btz, N]
            sim = sim / self.temp
            sim = F.softmax(sim, dim=1)
            audio_embeds = (sim @ text_embeds_support).unsqueeze(1)  # [btz, 1, d]
        return audio_embeds

    def encode_text(self, text):
        text_feats, _ = self.text_encoder(text)  # drop attention mask
        text_embeds = F.normalize(self.text_proj(text_feats[:, 0, :]), dim=-1)
        return text_embeds

    def forward(self, audio, text, idx):

        audio_embeds = self.encode_audio(audio)
        text_embeds = self.encode_text(text)

        idx = idx.view(-1, 1)
        pos_idx = torch.eq(idx, idx.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        sim_a2t = audio_embeds @ text_embeds.t() / self.temp
        sim_t2a = text_embeds @ audio_embeds.t() / self.temp
        loss = self.atc_loss(sim_a2t, sim_t2a, sim_targets)
        if self.embed_reg:
            loss = loss + torch.mean(torch.abs(audio_embeds)) / torch.sqrt(torch.sum(audio_embeds**2)) + \
                   torch.mean(torch.abs(text_embeds)) / torch.sqrt(torch.sum(text_embeds**2))

        return loss
