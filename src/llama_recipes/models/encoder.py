import types
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class WavLMEncoder(nn.Module):
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
        assert model_config.normalize == cfg.normalize, "normalize flag in config and model checkpoint do not match"
 
        return cls(cfg, WavLM_model)

    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask)[0]

class AVEncoder:

    @classmethod
    def load(cls, model_config):
        from .AV.av_net import AVNet
        avnet = AVNet(model_config)
        checkpoint = torch.load(model_config.TRAIN_LRS3_MODEL_FILE)
        avnet.load_state_dict(checkpoint['state_dict'],strict=False)
 
        return avnet
    
    
class ParaformerWrappedEncoder(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        from funasr import AutoModel
        from funasr.models.scama.utils import sequence_mask
        model_name = kwargs.get("encoder_name", "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        model = AutoModel(model=model_name, model_revision="v2.0.4")
        
        frontend = model.kwargs.get("frontend")
        model.model.decoder = None
        self.model = model.model
        self.frontend = frontend
        self.mask_fn = sequence_mask


    def forward(self, audio_samples, audio_samples_mask, audio_token_lengths, **kwargs):
        
        device = audio_samples.device
        audio_samples_lengths = audio_samples_mask.sum(-1)
        fbanks, fbanks_lens = self.frontend(audio_samples, audio_samples_lengths)
        fbanks, fbanks_lens = fbanks.to(device), fbanks_lens.to(device)
        
        batch = {"speech": fbanks, "speech_lengths": fbanks_lens}
        enc, enc_lens = self.model.encode(**batch)
        enc_mask = self.mask_fn(enc_lens, enc.size(1), device=device)[:, None, :]
        pre_acoustic_embeds, pre_token_length, _, _ = self.model.predictor(enc,
                                                                           mask=enc_mask,
                                                                           target_label_length=audio_token_lengths,
                                                                           )
        return pre_acoustic_embeds, pre_token_length
