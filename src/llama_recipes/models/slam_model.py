import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
)
import whisper

from llama_recipes.utils.config_utils import generate_peft_config
from llama_recipes.utils.train_utils import print_model_size


def setup_model(tokenizer, train_config, model_config, **kwargs):
    return slam_model(tokenizer, train_config, model_config, **kwargs)


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if model_config.llm_name=="llama-2-7b-hf":
        tokenizer = LlamaTokenizer.from_pretrained(model_config.llm_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer


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

def setup_llm(train_config, model_config, **kwargs):
    from pkg_resources import packaging
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(model_config.llm_path)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            model_config.llm_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


class slam_model(nn.Module):
    def __init__(
        self,
        tokenizer, 
        train_config, 
        model_config, 
        **kwargs
    ):
        super().__init__()
        # whisper 
        self.speech_encoder = whisper.load_model(model_config.encoder_path).encoder
        self.speech_encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, self.speech_encoder)
        for name, param in self.speech_encoder.named_parameters(): 
            param.requires_grad = False       
        self.speech_encoder.eval()

        # llama
        self.llm = setup_llm(train_config, model_config, **kwargs)

        # projector
        self.speech_encoder_projector = nn.Linear(self.speech_encoder.ln_post.normalized_shape[0] ,self.llm.config.hidden_size)

        # tokenizer
        self.tokenizer = tokenizer

    def forward(self,
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
        speech_mel = kwargs.get("speech_mel", None)
        speech_mask = kwargs.get("speech_mask", None)

        speech_encoder_outs = None
        if speech_mel is not None:
            speech_encoder_outs = self.speech_encoder.extract_variable_length_features(speech_mel.permute(0, 2, 1))
            speech_encoder_outs = self.speech_encoder_projector(speech_encoder_outs)

        input_ids[input_ids == -1] = 0
        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(input_ids)
        else:
            inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
        batch_size, token_num, dims = inputs_embeds.shape
        _, l, _ = speech_encoder_outs.shape
        speech_encoder_outs_pad = F.pad(speech_encoder_outs, (0, 0, 0, token_num-l, 0, 0), value=0.0)
        inputs_embeds = speech_encoder_outs_pad * speech_mask[:, :, None] + inputs_embeds * (~speech_mask[:, :, None])

        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        return model_outputs