#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
from torchlibrosa import LogmelFilterBank, Spectrogram


class AudioFeature(nn.Module):

    def __init__(self, audio_config):
        super().__init__()
        self.mel_trans = Spectrogram(n_fft=audio_config["n_fft"],
                                     hop_length=audio_config["hop_length"],
                                     win_length=audio_config["n_fft"],
                                     window='hann',
                                     center=True,
                                     pad_mode='reflect',
                                     freeze_parameters=True)

        self.log_trans = LogmelFilterBank(sr=audio_config["sr"],
                                          n_fft=audio_config["n_fft"],
                                          n_mels=audio_config["n_mels"],
                                          fmin=audio_config["f_min"],
                                          fmax=audio_config["f_max"],
                                          ref=1.0, 
                                          amin=1e-6,    
                                          top_db=None,
                                          freeze_parameters=True)

    def forward(self, input):
        # input: waveform [bs, wav_length]
        mel_feats = self.mel_trans(input)
        log_mel = self.log_trans(mel_feats)
        return log_mel
