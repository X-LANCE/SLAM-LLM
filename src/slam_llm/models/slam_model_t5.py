import os
import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size
from peft import PeftModel, PeftConfig
from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy

import logging

logger = logging.getLogger(__name__)


def setup_model(tokenizer, train_config, model_config, **kwargs):
    return slam_model(tokenizer, train_config, model_config, **kwargs)


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if "mupt" in model_config.llm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.llm_path, trust_remote_code=True, use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_list = (
        model_config.encoder_name.split(",") if model_config.encoder_name else []
    )
    if len(encoder_list) == 0:
        return None
    if len(encoder_list) == 1:
        encoder_name = encoder_list[0]
        if encoder_name == "whisper" or encoder_name == "qwen-audio":
            from slam_llm.models.encoder import WhisperWrappedEncoder

            encoder = WhisperWrappedEncoder.load(model_config)
        if encoder_name == "beats":
            from slam_llm.models.encoder import BEATsEncoder

            encoder = BEATsEncoder.load(model_config)
        if encoder_name == "wavlm":
            from slam_llm.models.encoder import WavLMEncoder

            encoder = WavLMEncoder.load(model_config)
        if encoder_name == "moco_wav2vec2":
            from slam_llm.models.encoder import AVEncoder

            encoder = AVEncoder.load(model_config)
        if "llama" in encoder_name.lower():
            from slam_llm.models.encoder import HfTextEncoder

            encoder = HfTextEncoder.load(model_config)
    print_module_size(
        encoder,
        encoder_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
    print_module_size(
        encoder,
        encoder_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )

    return encoder


def setup_llm(train_config, model_config, **kwargs):
    from pkg_resources import packaging

    use_cache = False
    if (
        train_config.enable_fsdp or train_config.enable_ddp
    ) and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        # v = packaging.version.parse(torch.__version__)
        # verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        # if not verify_latest_nightly:
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
        #                     "please install latest nightly.")
        rank = int(os.environ["RANK"])
        if rank == 0:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = AutoConfig.from_pretrained(model_config.llm_path)
            llama_config.use_cache = use_cache
            # with torch.device("meta"):
            model = AutoModelForSeq2SeqLM(llama_config)

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_config.llm_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    if (
        train_config.enable_fsdp or train_config.enable_ddp
    ) and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)
        except ImportError:
            logger.warning(
                "Module 'optimum' not found. Please install 'optimum' it before proceeding."
            )

    print_module_size(
        model,
        model_config.llm_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if (
        train_config.freeze_llm
    ):  # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()

    if kwargs.get(
        "peft_ckpt", None
    ):  # (FIX:MZY):reload will get wrong results when decoding
        logger.info("loading peft_ckpt from: {}".format(kwargs.get("peft_ckpt")))
        model = PeftModel.from_pretrained(
            model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True
        )
        model.print_trainable_parameters()
    elif train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print_module_size(
        model,
        model_config.llm_name,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model


def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from slam_llm.models.projector import EncoderProjectorConcat

        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from slam_llm.models.projector import EncoderProjectorCov1d

        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from slam_llm.models.projector import EncoderProjectorQFormer

        encoder_projector = EncoderProjectorQFormer(model_config)
    print_module_size(
        encoder_projector,
        model_config.encoder_projector,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return encoder_projector


class slam_model(nn.Module):
    def __init__(self, tokenizer, train_config, model_config, **kwargs):
        super().__init__()
        # modality encoder
        self.encoder = setup_encoder(train_config, model_config, **kwargs)

        # llm
        self.llm = setup_llm(train_config, model_config, **kwargs)

        # projector
        self.encoder_projector = setup_encoder_projector(
            train_config, model_config, **kwargs
        )

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_lengths: Optional[torch.LongTensor] = None,
        prompt_lengths: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get(
            "audio_mel_post_mask", None
        )  # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        vis_len = kwargs.get("vis_len", None)
        maskw2v = kwargs.get(
            "maskw2v", False
        )  # (FIX:MZY) False for supervised learning and inference

        # for text encoder
        instruct_ids = kwargs.get("instruct_ids", None)
        instruct_mask = kwargs.get("instruct_mask", None)


        encoder_outs = None
        if audio_mel is not None or audio is not None:
            if self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(
                    audio_mel.permute(0, 2, 1)
                )  # bs*seq*dim
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(
                    audio_mel, audio_mel_mask
                )  # bs*seq*dim
            if self.model_config.encoder_name == "wavlm":
                encoder_outs = self.encoder.extract_features(
                    audio, 1 - audio_mask
                )  # (FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
            if self.model_config.encoder_name == "moco_wav2vec2":
                encoder_outs, inputLenBatch, audio_mel_post_mask = self.encoder(
                    (audio, audio_mask, visual, vis_len), maskw2v
                )  # bs*seq*dim
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        if instruct_ids is not None:
            if self.encoder is not None:
                encoder_outs = self.encoder(
                    input_ids=instruct_ids, attention_mask=instruct_mask
                ).last_hidden_state

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, instruct_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            inputs_embeds = self.llm.shared(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape
        new_inputs_embeds = torch.zeros((batch_size, token_num+encoder_outs.shape[1], dims), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        for i, item in enumerate(encoder_outs):
            new_inputs_embeds[i, :prompt_lengths[i]] = inputs_embeds[i]
            new_inputs_embeds[i, prompt_lengths[i]:prompt_lengths[i]+audio_lengths[i]] = item
        inputs_embeds = new_inputs_embeds

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        # decoder will shift right the labels as decoder input when no decoder_input is given
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(
                    preds.detach(), labels.detach(), ignore_label=-100
                )

        return model_outputs, acc

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return model_outputs

    @torch.no_grad()
    def inference(
        self,
        wav_path=None,
        prompt=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):

        device = kwargs.get("device", "cuda")
        if os.path.exists(wav_path):  # Audio-Text QA
            import whisper

            audio_raw = whisper.load_audio(wav_path)
            audio_raw = whisper.pad_or_trim(audio_raw)

            mel_size = getattr(
                self.dataset_config, "mel_size", 80
            )  # 80 for large v1 and v2, 128 for large v3
            audio_mel = (
                whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
                .permute(1, 0)[None, :, :]
                .to(device)
            )

            encoder_outs = self.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1)
            )

            if self.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        prompt = "USER: {}\n ASSISTANT:".format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds[None, :, :]), dim=1
        )  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )

        # generate
        model_outputs = self.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        return model_outputs
