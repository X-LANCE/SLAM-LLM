import torch
import torch.nn as nn
from .espnet.nets.pytorch_backend.e2e_asr_conformer_av import E2E
from .datamodule.transforms import TextTransform

class ModelModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.backbone_args = self.cfg.model.audiovisual_backbone

        self.text_transform = TextTransform()  #SentencePiece
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
            # else:  #
            self.model.load_state_dict(ckpt)

    #def forward(self, video, audio):
    def forward(self, sample):
        visual_batch = sample[2]  #torch.Size([4, 78, 1, 112, 112])  torch.Size([2, 106, 1, 112, 112])
        video_feat, _ = self.model.encoder(visual_batch, None)   #torch.Size([2, 106, 768])

        audio_batch = sample[0]  #torch.Size([2, 67360])
        audio_feat, _ = self.model.aux_encoder(audio_batch.unsqueeze(-1), None)  #torch.Size([2, 105, 768]) 修改了变成106

        audiovisual_feat = self.model.fusion(torch.cat((video_feat, audio_feat), dim=-1))  #torch.Size([4, 78, 1536]) ->  torch.Size([4, 78, 768])

        # 配合avsr的代码
        inputLenBatch = sample[3]
        mask = self.makeMaskfromLength(audiovisual_feat.shape[:-1], inputLenBatch, audiovisual_feat.device)

        return audiovisual_feat, inputLenBatch, mask


    def makeMaskfromLength(self, maskShape, maskLength, maskDevice):
        mask = torch.zeros(maskShape, device=maskDevice)
        mask[(torch.arange(mask.shape[0]), maskLength - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return mask