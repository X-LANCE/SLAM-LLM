import types
import torch
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):  #torch.Size([2, 80, 3000])
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))  #torch.Size([2, 1280, 3000])
            x = F.gelu(self.conv2(x)) #torch.Size([2, 1280, 1500])
            x = x.permute(0, 2, 1) #torch.Size([2, 1500, 1280])

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
        if model_config.modal == "AV":
            checkpoint = torch.load(model_config.TRAIN_LRS3_MODEL_FILE)
        elif model_config.modal == "AO":
            checkpoint = torch.load(model_config.TRAINED_AO_FILE)  #check 了这么load没问题
        elif model_config.modal == "VO":
            checkpoint = torch.load(model_config.TRAINED_VO_FILE)
        msg = avnet.load_state_dict(checkpoint['state_dict'],strict=False)
        logger.info(msg)

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

class HubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]

        if model_config.encoder_type == "pretrain":
            pass
        
        else:
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        return model

class AVHubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        from .avhubert import hubert_pretraining, hubert, hubert_asr
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]

        return model

class WavLMEncoder(torch.nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .wavlm.WavLM import WavLM, WavLMConfig
        checkpoint = torch.load(model_config.encoder_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        WavLM_model = WavLM(cfg)
        WavLM_model.load_state_dict(checkpoint['model'])
        # assert model_config.normalize == cfg.normalize, "normalize flag in config and model checkpoint do not match"  cfg.normalize:True

        return cls(cfg, WavLM_model)

    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask)[0]