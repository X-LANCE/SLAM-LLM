# MIT License
#
# Copyright 2023 ByteDance Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import json
import random
import torch
from torch import nn
from einops import rearrange

from ..modules.random_quantizer import RandomProjectionQuantizer
from ..modules.features import MelSTFT
from ..modules.conv import Conv2dSubsampling


class MusicFM25Hz(nn.Module):
    """
    MusicFM

    Input: 128-band mel spectrogram
    Frontend: 2-layer Residual convolution
    Backend: 12-layer Conformer
    Quantizer: a codebook for mel spectrogram
    """

    def __init__(
        self,
        num_codebooks=1,
        codebook_dim=16,
        codebook_size=4096,
        features=["melspec_2048"],
        hop_length=240,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=12,
        mask_hop=0.4,
        mask_prob=0.6,
        is_flash=False,
        stat_path="./data/fma_stats.json",
        model_path="./data/pretrained_fma.pt",
        w2v2_config_path="facebook/wav2vec2-conformer-rope-large-960h-ft",
    ):
        super(MusicFM25Hz, self).__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.features = features

        # load feature mean / std stats
        with open(stat_path, "r") as f:
            self.stat = json.load(f)

        # feature extractor
        self.preprocessor_melspec_2048 = MelSTFT(
            n_fft=2048, hop_length=hop_length, is_db=True
        )

        # random quantizer
        seed = 142
        for feature in self.features:
            for i in range(num_codebooks):
                setattr(
                    self,
                    f"quantizer_{feature}_{i}",
                    RandomProjectionQuantizer(
                        n_mels * 4, codebook_dim, codebook_size, seed=seed + i
                    ),
                )

        # two residual convolution layers + one projection layer
        self.conv = Conv2dSubsampling(
            1, conv_dim, encoder_dim, strides=[2, 2], n_bands=n_mels
        )

        # Conformer
        if is_flash:
            from modules.flash_conformer import (
                Wav2Vec2ConformerEncoder,
                Wav2Vec2ConformerConfig,
            )
        else:
            from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
                Wav2Vec2ConformerEncoder,
                Wav2Vec2ConformerConfig,
            )
        config = Wav2Vec2ConformerConfig.from_pretrained(
            w2v2_config_path
        )
        config.num_hidden_layers = encoder_depth
        config.hidden_size = encoder_dim

        self.conformer = Wav2Vec2ConformerEncoder(config)

        # projection
        self.linear = nn.Linear(encoder_dim, codebook_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # cls token (used for sequence classification)
        random.seed(seed)
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))

        # load model
        if model_path:
            S = torch.load(model_path)["state_dict"]
            SS = {k[6:]: v for k, v in S.items()}
            self.load_state_dict(SS, strict=True)

    def masking(self, x):
        """random masking of 400ms with given probability"""
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop)
        len_masking_token = int(24000 / self.hop_length / 2 / 2 * self.mask_hop)

        # get random mask indices
        start_indices = torch.rand(b, t // len_masking_raw) < self.mask_prob
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)

        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    def encoder(self, x):
        """2-layer conv + w2v-conformer"""
        x = self.conv(x)
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits = {
            key: logits[:, :, i * self.codebook_size : (i + 1) * self.codebook_size]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key]
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """rearrange the batch to flatten every 4 steps"""
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=4)
        return x

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        for key in x.keys():
            layer = getattr(self, "quantizer_%s" % key)
            out[key] = layer(x[key])
        return out

    def get_targets(self, x):
        x = self.preprocessing(x, features=self.features)
        x = self.normalize(x)
        x = self.rearrange(x)
        target_tokens = self.tokenize(x)
        return target_tokens

    def get_predictions(self, x):
        # preprocessing
        x = self.preprocessing(x, features=["melspec_2048"])
        x = self.normalize(x)

        # encoding
        logits, hidden_emb = self.encoder(x["melspec_2048"])

        return logits, hidden_emb

    def get_latent(self, x, layer_ix=12):
        _, hidden_states = self.get_predictions(x)
        emb = hidden_states[layer_ix]
        return emb

    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            masked_logits = logits[key][tuple(masked_indices.t())]
            masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def forward(self, x):
        # get target feature tokens
        target_tokens = self.get_targets(x)

        # masking
        x, masked_indices = self.masking(x)

        # forward
        logits, hidden_emb = self.get_predictions(x)

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)

        return logits, hidden_emb, losses, accuracies
