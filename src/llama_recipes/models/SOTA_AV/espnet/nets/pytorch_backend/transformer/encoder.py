#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet

from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.nets_utils import rename_state_dict

from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
)
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward

from llama_recipes.models.SOTA_AV.espnet.nets.pytorch_backend.transformer.repeat import repeat


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param str encoder_attn_layer_type: encoder attention layer type
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param bool zero_triu: whether to zero the upper triangular part of attention matrix
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        attention_dim=768,
        attention_heads=12,
        linear_units=3072,
        num_blocks=12,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        encoder_attn_layer_type="mha",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        relu_type="prelu",
        a_upsample_ratio=1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if encoder_attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding

        # -- frontend module.
        if input_layer == "conv1d":
            self.frontend = Conv1dResNet(relu_type=relu_type, a_upsample_ratio=a_upsample_ratio)
        elif input_layer == "conv3d":
            self.frontend = Conv3dResNet(relu_type=relu_type)
        else:
            self.frontend = None
        # -- backend module.
        if input_layer in ["conv1d", "conv3d"]:
            self.embed = torch.nn.Sequential(torch.nn.Linear(512, attention_dim), pos_enc_class(attention_dim, positional_dropout_rate))
        else:
            raise NotImplementedError("Support only conv1d and conv3d")

        self.normalize_before = normalize_before
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)

        if encoder_attn_layer_type == "mha":
            encoder_attn_layer = MultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):  ## [B, T, C, H, W] -> [B, C, T, H, W]  xs:torch.Size([4, 78, 1, 112, 112]), masks:None
            xs = self.frontend(xs)  #torch.Size([4, 78, 512])

        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple): #
            xs = xs[0]  #torch.Size([4, 78, 768])

        if self.normalize_before: #
            xs = self.after_norm(xs) #torch.Size([4, 78, 768])

        return xs, masks  # AO 和 VO都是这个尺寸

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):
            xs = self.frontend(xs)

        xs = self.embed(xs)

        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
