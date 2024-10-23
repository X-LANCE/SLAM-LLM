from slam_llm.utils.train_utils import print_module_size
import torch
import os
import torch.nn as nn

def setup_codec(train_config, model_config, **kwargs):
    if model_config.codec_decoder_type == "SNAC":
        from snac import SNAC
        codec_decoder = SNAC.from_pretrained(model_config.codec_decoder_path).eval()
        codec_decoder_module = codec_decoder
    elif model_config.codec_decoder_type == "CosyVoice":
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import CosyVoice
        codec_decoder = CosyVoice(model_config.codec_decoder_path, load_jit=True, load_onnx=False, fp16=True).model
        codec_decoder_module = nn.ModuleList((codec_decoder.flow,codec_decoder.hift))
    else:
        raise NotImplementedError
    print_module_size(codec_decoder_module, model_config.codec_decoder_type, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return codec_decoder

def get_single_layer_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio):
    audio_length = len(audio_tokens) + num_latency_tokens + 1  # 1 is due to end of audio token
    result = [padding_token] * num_latency_tokens + list(audio_tokens) + [end_of_audio]
    result_tensor = torch.tensor(result).unsqueeze(0)
    return result_tensor, audio_length

def get_group_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio, num_layers):
    padded_audio_tokens = audio_tokens + [end_of_audio]
    padding_needed = (num_layers - len(padded_audio_tokens) % num_layers ) % num_layers
    
    # Add padding to ensure even distribution across layers
    padded_audio_tokens = padded_audio_tokens + [padding_token] * padding_needed
    total_length = len(padded_audio_tokens)
    audio_length = total_length // num_layers + num_latency_tokens

    # Create the result for each layer
    result = []
    for layer in range(num_layers):
        layer_tokens = [padding_token] * num_latency_tokens
        layer_tokens.extend(padded_audio_tokens[layer::num_layers])
        result.append(torch.tensor(layer_tokens))
    
    result_tensor = torch.stack(result)
    return result_tensor, audio_length
