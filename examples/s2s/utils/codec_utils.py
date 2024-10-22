from snac import SNAC
from slam_llm.utils.train_utils import print_module_size
import torch
import os

def setup_codec(train_config, model_config, **kwargs):
    if model_config.codec_decoder_type == "SNAC":
        codec_decoder = SNAC.from_pretrained(model_config.codec_decoder_path).eval()
    else:
        raise NotImplementedError
    print_module_size(codec_decoder, model_config.codec_decoder_type, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return codec_decoder

def get_single_layer_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio):
    audio_length = len(audio_tokens) + num_latency_tokens + 1   # 1 is due to end of audio token
    result = []
    result.extend([padding_token] * num_latency_tokens)
    result.extend([audio_tokens[i] for i in range(len(audio_tokens))])
    result.append(end_of_audio)
    result_tensor = torch.tensor([int(token) for token in result])
    result_tensor = result_tensor.unsqueeze(0)
    return result_tensor, audio_length

def get_group_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio, num_layers):
    audio_length = len(audio_tokens) // num_layers + num_latency_tokens + 1   # 1 is due to end of audio token
    result = []
    for layer in range(1, num_layers + 1):
        layer_tokens = []
        layer_tokens.extend([padding_token] * num_latency_tokens)
        layer_tokens.extend([audio_tokens[i] for i in range(len(audio_tokens)) if i % num_layers == layer - 1])
        layer_tokens.append(end_of_audio)
        result.append(torch.tensor([int(token) for token in layer_tokens]))
    result_tensor = torch.stack(result)
    return result_tensor, audio_length