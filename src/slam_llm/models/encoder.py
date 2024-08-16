import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

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


@dataclass
class UserDirModule:
    user_dir: str
    
class EATEncoder:
    
    @classmethod
    def load(cls, model_config):
        import fairseq
        model_path = UserDirModule(model_config.encoder_fairseq_dir)
        fairseq.utils.import_user_module(model_path)
        EATEncoder, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        EATEncoder = EATEncoder[0]

        return EATEncoder
    
    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask = padding_mask, mask=False, remove_extra_tokens = False)['x']

class SpatialASTEncoder:
    @classmethod
    def load(cls, model_config):
        from functools import partial
        from .SpatialAST import SpatialAST 
        binaural_encoder = SpatialAST.BinauralEncoder(
            num_classes=355, drop_path_rate=0.1, num_cls_tokens=3,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        checkpoint = torch.load(model_config.encoder_ckpt, map_location='cpu')
        binaural_encoder.load_state_dict(checkpoint['model'], strict=False) 
        return binaural_encoder

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
        features = self.model.extract_features(source, padding_mask)[0]
        print(f"Shape of extracted features: {features.shape}")
        return features

class AVHubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        from .avhubert import hubert_pretraining, hubert, hubert_asr
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        return model

class HubertEncoder:

    @classmethod
    def load(cls, model_config):
        import fairseq
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_config.encoder_path])
        model = models[0]
        if model_config.encoder_type == "pretrain":
            pass
        elif model_config.encoder_type == "finetune":
            model.w2v_encoder.proj = None
            model.w2v_encoder.apply_mask = False
        else:
            assert model_config.encoder_type in ["pretrain", "finetune"], "input_type must be one of [pretrain, finetune]" 
        return model


class HfTextEncoder:

    @classmethod
    def load(cls, model_config):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_config.encoder_path)
        return model

class MusicFMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_config):
        from .musicfm.model.musicfm_25hz import MusicFM25Hz
        model = MusicFM25Hz(
            stat_path = model_config.encoder_stat_path,
            model_path = model_config.encoder_path,
            w2v2_config_path = model_config.get('encoder_config_path', "facebook/wav2vec2-conformer-rope-large-960h-ft")
        )
        return cls(model_config, model)

    def extract_features(self, source, padding_mask=None):
        _, hidden_states = self.model.get_predictions(source)
        out = hidden_states[self.config.encoder_layer_idx]
        return out


# j: add a new encoder
class Wav2PhonemeEncoder(nn.Module):
    def __init__(self, config, model, feature_extractor):
        super().__init__()
        self.config = config
        self.model = model
        self.feature_extractor = feature_extractor

    @classmethod
    def load(cls, model_config):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        # Load the feature extractor and model
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
        model = Wav2Vec2Model.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
        return cls(model_config, model, feature_extractor)

    def extract_features(self, audio_input):
        # Pass the processed inputs through the Wav2Vec2 model
        outputs = self.model(audio_input)
        # Return the last hidden state as the extracted features
        return outputs.last_hidden_state
