import torch
import os
import logging
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_encoder,
    setup_encoder_projector,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from torchaudio.transforms import Resample
from slam_llm.models.BEATs.BEATs import BEATs
from slam_llm.models.EAT.EAT import EAT_preprocess
import torchaudio

logger = logging.getLogger(__name__)

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    model = slam_model_aac(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get(
        "ckpt_path", None
    )  # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class slam_model_aac(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )


    @torch.no_grad()
    def inference(
        self,
        wav_path = None,
        prompt = None,
        dataset_config = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus = None,
        assistant_model = None,
        streamer = None,
        negative_prompt_ids = None,
        negative_prompt_attention_mask = None,
        **kwargs,
    ):
        device = kwargs.get("device", "cuda")
        if os.path.exists(wav_path):
            try:
                audio_raw, sample_rate = torchaudio.load(wav_path)
                if audio_raw.shape[1] == 0:
                    raise ValueError("Empty audio file")
                resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                audio_raw = resampler(audio_raw)

            except (FileNotFoundError, ValueError, RuntimeError):
                audio_raw = torch.zeros(1, 16000)

            if self.model_config.encoder_name == "beats":
                audio_mel = BEATs.preprocess(audio_raw[0], fbank_mean=dataset_config.fbank_mean, fbank_std=dataset_config.fbank_std)
            elif self.model_config.encoder_name == "eat":
                audio_mel = EAT_preprocess(source=audio_raw[0],norm_mean=dataset_config.fbank_mean,norm_std=dataset_config.fbank_std,
                                        target_length=dataset_config.target_length,fixed_length=dataset_config.fixed_length,random_crop=dataset_config.random_crop)
            else:
                pass
                
            audio_mel = audio_mel.unsqueeze(dim=0)
            audio_mel_mask = torch.ones_like(audio_mel)
            audio_mel = audio_mel.to(device)
            audio_mel_mask = audio_mel_mask.to(device)
            
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(audio_mel, padding_mask = audio_mel_mask, feature_only = True)
            if self.model_config.encoder_name == "eat":
                encoder_outs = self.encoder.model.extract_features(audio_mel.unsqueeze(dim=1), padding_mask = None, mask=False, remove_extra_tokens = False)['x']
            
            if self.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(encoder_outs.size()[:-1], dtype=torch.long).to(encoder_outs.device)
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
        else: # Text QA
            encoder_outs = torch.empty(1, 0, self.llm.model.embed_tokens.embedding_dim).to(device)

        prompt = "USER: {} \n ASSISTANT:".format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)
        
        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)
        
        inputs_embeds = torch.cat((encoder_outs, inputs_embeds[None, :, :]), dim=1)  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(inputs_embeds.device)

        # generate
        model_outputs = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

        return model_outputs
