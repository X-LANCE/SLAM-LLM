import torch
import torch.nn as nn
import torchaudio

from .datamodule.transforms import TextTransform
from .espnet.nets.pytorch_backend.e2e_asr_conformer import E2E


class ModelModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.modality == "visual":  #FIX bug
            self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # -- initialise
        if self.cfg.pretrained_model_path:
            ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
            # if self.cfg.transfer_frontend:
            #     tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
            #     self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            # elif self.cfg.transfer_encoder:
            #     tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
            #     self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            # else:
            self.model.load_state_dict(ckpt)

    def forward(self, sample):
        if self.cfg.modality == "audio": 
            audio_batch = sample[0]  #TODO  #torch.Size([2, 101760])

            enc_feat, _ = self.model.encoder(audio_batch.unsqueeze(-1), None)  #torch.Size([2, 159, 768])

        if self.cfg.modality == "visual":  #改名字
            visual_batch = sample[2]  #torch.Size([4, 78, 1, 112, 112]) 
  
            #enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None) sample已经在cuda上了   ; 不需要squeeze 一进去就是 [B, T, C, H, W] -> [B, C, T, H, W] 
            enc_feat, _ = self.model.encoder(visual_batch, None)   #final: torch.Size([4, 78, 768])  长度未变

            # 配合avsr的代码
        inputLenBatch = sample[3]
        mask = self.makeMaskfromLength(enc_feat.shape[:-1], inputLenBatch, enc_feat.device)

        
        return enc_feat, inputLenBatch, mask




    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask


