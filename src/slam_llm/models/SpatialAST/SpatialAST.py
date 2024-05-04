import torch
import torch.nn as nn

from torchlibrosa.stft import STFT, LogmelFilterBank
from timm.models.layers import to_2tuple

from .vision_transformer import VisionTransformer as _VisionTransformer

def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x) # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2) # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2) # 32, 768, 1212 -> 32, 1212, 768
        return x

class BinauralEncoder(_VisionTransformer):
    """ Spatial Audio Spectrogram Transformer designed for Sound Event Localization and Detection
        --------------------------------------------------------
        References:
        Spatial-AST from BAT: https://github.com/zszheng147/Spatial-AST and https://arxiv.org/abs/2402.01591
        --------------------------------------------------------
    """
    def __init__(self, num_cls_tokens=3, **kwargs):
        super(BinauralEncoder, self).__init__(**kwargs)
        img_size = (1024, 128) # 1024, 128
        in_chans = 1
        emb_dim = 768

        del self.cls_token
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))

        self.patch_embed = PatchEmbed_new(
            img_size=img_size, patch_size=(16, 16), 
            in_chans=in_chans, embed_dim=emb_dim, stride=16
        ) # no overlap. stride=img_size=16

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, win_length=1024, window='hann', 
            center=True, pad_mode='reflect', freeze_parameters=True
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, 
            fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True
        )
        
        self.conv_downsample = nn.Sequential(
            conv3x3(4, 1), 
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        self.bn = nn.BatchNorm2d(2, affine=False)
        del self.norm  # remove the original norm

        self.target_frame = 1024

    def forward_features_mask(self, x):
        B = x.shape[0] #bsz, 512, 768 (unmasked)

        x = x + self.pos_embed[:, 1:, :]
        
        cls_tokens = self.cls_tokens
        cls_tokens = cls_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)   # bsz, 512 + 2 + 10, 768 
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)

        return x

    @torch.no_grad()
    def forward(self, waveforms):
        B, C, T = waveforms.shape

        waveforms = waveforms.reshape(B * C, T)
        real, imag = self.spectrogram_extractor(waveforms) 

        log_mel = self.logmel_extractor(torch.sqrt(real**2 + imag**2)).reshape(B, C, -1, 128)
        log_mel = self.bn(log_mel)
        
        IPD = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
        x = torch.cat([log_mel, torch.matmul(torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1), self.logmel_extractor.melW)], dim=1)

        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(x, (self.target_frame, x.shape[3]), mode="bicubic", align_corners=True)
    
        x = self.conv_downsample(x)
        x = self.patch_embed(x)
        x = self.forward_features_mask(x)

        return x