import types
import torch
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        import whisper
        encoder = whisper.load_model(name=model_config.encoder_path, device='cpu').encoder
        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        return encoder


class BEATsEncoder:

    @classmethod
    def load(cls, model_config):
        from .BEATs.BEATs import BEATs, BEATsConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])

        return BEATs_model


class AVEncoder:

    @classmethod
    def load(cls, model_config):
        from .AV.av_net import AVNet
        avnet = AVNet(model_config)
        checkpoint = torch.load(model_config.TRAIN_LRS3_MODEL_FILE)
        avnet.load_state_dict(checkpoint['state_dict'],strict=False)
 
        return avnet

class SOTAAVEncoder:

    @classmethod
    def load(cls, cfg):
        if cfg.modality in ["audio", "visual"]:
            from .SOTA_AV.lightning import ModelModule
        elif cfg.modality == "audiovisual":
            from .SOTA_AV.lightning_av import ModelModule
        modelmodule = ModelModule(cfg)
        msg = modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
        logger.info(msg)
        # modelmodule.to(device)
 
        return modelmodule