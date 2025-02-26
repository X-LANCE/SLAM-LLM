import sys
import os
import soundfile as sf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
import torchaudio
import torch
import uuid

codec_decoder_path = "/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice2-0.5B"
codec_decoder = CosyVoice2(codec_decoder_path, load_jit=False, load_trt=False, fp16=False)

# audio_prompt_path="/nfs/yangguanrou.ygr/codes/CosyVoice/zero_shot_prompt.wav" #希望你以后能做的比我还好哟
audio_prompt_path="/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/prompt/prompt_1.wav"


from cosyvoice.utils.file_utils import load_wav
prompt_speech_16k = load_wav(audio_prompt_path, 16000)
flow_prompt_speech_token, flow_prompt_speech_token_len = codec_decoder.frontend._extract_speech_token(prompt_speech_16k)
prompt_speech_24000 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(prompt_speech_16k)
prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_24000)
flow_embedding = codec_decoder.frontend._extract_spk_embedding(prompt_speech_16k)

this_uuid = str(uuid.uuid1())
speed=1.0
audio_tokens=[1718, 3480, 3883, 2897, 2906, 4367, 6059, 2635, 249, 2298, 5401, 5645, 918, 1221, 2769, 4939, 5100, 6285, 5130, 4752, 4915, 80, 305, 2249, 4440, 4915, 1514, 1951, 5675, 4948, 3593, 4115, 2381, 2887, 1230, 2982, 5409, 5652, 3954, 5582, 5093, 5830, 2897, 4725, 4833, 4753, 1353, 300, 4523, 116, 431, 2050, 753, 736, 1788, 4218, 4218, 4218, 4218, 4218, 4137, 2031, 3648, 5832, 5832, 1467, 1975, 5833, 2200, 3750, 63, 2837, 4133, 2384, 2001, 1323, 4131, 1944, 3888, 2441, 4808, 4754, 2567, 2540, 4564, 231, 2490, 791, 1951, 1951, 2744, 5092, 133, 3750, 3751, 3751, 6018, 750, 4567, 4862, 491, 488, 6426, 6429, 6426, 6453, 3996, 2025, 4218, 4218]

audio_tokens = torch.tensor(audio_tokens)
audio_tokens = audio_tokens.unsqueeze(0)
print(audio_tokens.shape)
# audio_tokens=[ audio_tokens]
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
speech_sample_rate=24000

sf.write("/nfs/yangguanrou.ygr/data/gpt4o_third/test_cosyvoice2/gpt4o_675_angry_ballad.wav", audio_hat.squeeze().cpu().numpy(), speech_sample_rate)


# 24000!!!!