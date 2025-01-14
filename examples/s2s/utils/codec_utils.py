from slam_llm.utils.train_utils import print_module_size
import torch
import torchaudio
import os
import torch.nn as nn
import uuid
import logging
logger = logging.getLogger(__name__)
import pdb
def setup_codec(train_config, model_config, **kwargs):
    if model_config.codec_decoder_type == "SNAC":
        from snac import SNAC
        codec_decoder = SNAC.from_pretrained(model_config.codec_decoder_path).eval()
        codec_decoder_module = codec_decoder
    elif model_config.codec_decoder_type == "CosyVoice":
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
        if model_config.cosyvoice_version==1:
            #codec_decoder = CosyVoice(model_config.codec_decoder_path, load_jit=True, load_onnx=False, fp16=True)
            codec_decoder = CosyVoice(model_config.codec_decoder_path, load_jit=False, load_trt=False, fp16=False)
        elif model_config.cosyvoice_version==2:
            codec_decoder = CosyVoice2(model_config.codec_decoder_path, load_jit=False, load_trt=False, fp16=False) #?
        else:
            raise NotImplementedError
        codec_decoder_module = nn.ModuleList((codec_decoder.model.flow,codec_decoder.model.hift))
    else:
        raise NotImplementedError
    print_module_size(codec_decoder_module, model_config.codec_decoder_type + " Codec", int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return codec_decoder

def get_single_layer_answer_token(audio_tokens, num_latency_tokens, padding_token, end_of_audio):
    audio_length = len(audio_tokens) + num_latency_tokens + 1  # 1 is due to end of audio token
    result = [padding_token] * num_latency_tokens + list(audio_tokens) + [end_of_audio] #明白 第一行
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

def audio_decode_cosyvoice(audio_tokens, model_config, codec_decoder, tone_dir, audio_prompt_path=None, code_layer=1, num_latency_tokens=1, speed=1.0):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        tone_dir (str): The tone directory or setting.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.
    
    Returns:
        torch.Tensor: Generated audio waveform.
    """

    # Reshape audio tokens based on code_layer
    if code_layer > 1:
        audio_tokens_tensor = torch.stack(audio_tokens, dim=0)
        audio_tokens_permuted = audio_tokens_tensor.permute(1, 0)
        audio_tokens = audio_tokens_permuted.reshape(-1).unsqueeze(0)
        audio_tokens = audio_tokens[..., num_latency_tokens * code_layer:]
    else:
        audio_tokens = torch.cat(audio_tokens, dim=-1).unsqueeze(0)
        audio_tokens = audio_tokens[..., num_latency_tokens:]

    # Get vocabulary configuration for end of audio (EOA) and padding token
    eoa = model_config.vocab_config.eoa
    pad_a = model_config.vocab_config.pad_a

    # Truncate audio tokens at the EOA token  前面已经处理过了
    try:
        end_index = torch.nonzero(audio_tokens[0] == eoa)[0]
    except:
        print(eoa)
        print(audio_tokens[0])
    audio_tokens = audio_tokens[..., :end_index]

    # Handle padding tokens if present, # FIXME: this is a temporary fix for the padding issue, where the padding token may be included in the audio tokens
    if pad_a in audio_tokens:
        end_index = torch.nonzero(audio_tokens[0] == pad_a)[0]
        audio_tokens = audio_tokens[..., :end_index]

    # Generate a unique ID for this audio generation
    this_uuid = str(uuid.uuid1()) #uuid.uuid1() 是一个函数，用于生成基于时间的 UUID（Universally Unique Identifier，通用唯一标识符）。uuid.uuid1() 生成的 UUID 包含时间戳和计算机的网络地址（通常是 MAC 地址），从而保证在空间和时间上的唯一性。

    # Set up the prompt speech features and speaker embedding
    if tone_dir == "default_tone":
        # flow_embedding = codec_decoder.frontend.spk2info['英文女']['embedding']
        flow_embedding = codec_decoder.frontend.spk2info['中文女']['embedding']
        # spk_list = list(codec_decoder.frontend.spk2info.keys()) # logger.info(spk_list)  #['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        prompt_speech_feat = torch.zeros(1, 0, 80)
    else:
        from utils.cosyvoice.utils.file_utils import load_wav
        prompt_speech_16k = load_wav(audio_prompt_path, 16000)
        flow_prompt_speech_token, flow_prompt_speech_token_len = codec_decoder.frontend._extract_speech_token(prompt_speech_16k)
        if model_config.cosyvoice_version==1:
            prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
            prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_22050)
        elif model_config.cosyvoice_version==2:
            prompt_speech_24000 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(prompt_speech_16k)
            prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_24000)
        flow_embedding = codec_decoder.frontend._extract_spk_embedding(prompt_speech_16k)

    # Convert tokens to audio waveform
    if model_config.cosyvoice_version==1:
        audio_hat = codec_decoder.model.token2wav(
            token=audio_tokens,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            finalize=True,
            speed=speed
        )
    elif model_config.cosyvoice_version==2:
        audio_hat = codec_decoder.model.token2wav(
            token=audio_tokens,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            token_offset=0,
            finalize=True,
            speed=speed
        )
    else:
        raise NotImplementedError
    return audio_hat
