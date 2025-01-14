import sys
import os
import soundfile as sf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
import torchaudio
import torch
import uuid

codec_decoder_path="/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice-300M-SFT"
codec_decoder = CosyVoice(codec_decoder_path, load_jit=False, load_trt=False, fp16=False)

audio_prompt_path="/nfs/yangguanrou.ygr/codes/CosyVoice/zero_shot_prompt.wav"


from cosyvoice.utils.file_utils import load_wav
prompt_speech_16k = load_wav(audio_prompt_path, 16000)
flow_prompt_speech_token, flow_prompt_speech_token_len = codec_decoder.frontend._extract_speech_token(prompt_speech_16k)
prompt_speech_22050 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(prompt_speech_16k)
prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_22050)
flow_embedding = codec_decoder.frontend._extract_spk_embedding(prompt_speech_16k)

this_uuid = str(uuid.uuid1())
speed=1.0
audio_tokens=[436, 632, 561, 561, 561, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 66, 658, 658, 658, 208, 185, 600, 412, 1529, 514, 645, 676, 1923, 309, 725, 318, 33, 247, 475, 475, 388, 388, 388, 376, 376, 376, 437, 254, 664, 477, 84, 576, 337, 10, 3836, 704, 656, 1460, 333, 1104, 1104, 406, 406, 149, 149, 2691, 2691, 2691, 1035, 82, 82, 15, 580, 297, 79, 95, 733, 3766, 3766, 391, 101, 289, 64, 390, 310, 41, 41, 41, 252, 236, 236, 642, 8, 3190, 249, 2444, 2444, 2444, 249, 249, 249, 508, 508, 508, 685, 685, 710, 710, 149, 2691, 2691, 2691, 471, 471, 150, 22, 252, 87, 3406, 420, 681, 903, 183, 2514, 506, 3052, 690, 2579, 20, 3480, 24, 470, 460, 460, 298, 395, 124, 425, 570, 385, 385, 3480, 507, 647, 310, 27, 348, 138, 57, 302, 446, 342, 209, 341, 297, 302, 250, 1942, 56, 445, 378, 289, 21, 21, 1381, 655, 337, 10, 10, 10, 3836, 389, 64, 372, 372, 579, 41, 41, 270, 252, 236, 395, 8, 249, 670, 140, 68, 68, 230, 230, 442, 230, 230, 230, 68, 435, 473, 465, 515, 515, 726, 726, 629, 281, 281, 557, 557, 557, 242, 2514, 293, 720, 558, 36, 466, 457, 457, 387, 387, 387, 421, 1382, 424, 424, 263, 609, 707, 3821, 211, 558, 558, 316, 316, 316, 433, 595, 387, 1930, 629, 1307, 1354, 172, 56, 492, 212, 212, 3766, 3766, 3766, 504, 143, 359, 74, 1381, 325, 63, 156, 170, 512, 58, 553, 2644, 2644, 553, 2644, 2644, 58, 333, 550, 649, 649, 609, 481, 189, 374, 355, 15, 2828, 59, 209, 79, 430, 189, 2280, 357, 290, 448, 718, 468, 269, 269, 700, 3158, 366, 515, 515, 212, 49, 466, 466, 483, 151, 88, 1942, 2280, 374, 1, 177, 2330, 445, 445, 56, 79, 733, 297, 297, 297, 673, 102, 232, 1037, 505, 53, 459, 249, 331, 331, 331, 331, 331, 331, 730, 730, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 2189]

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
    finalize=True,
    speed=speed
)        
speech_sample_rate=22050

sf.write("/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/1_debug_s3tokenizer.wav", audio_hat.squeeze().cpu().numpy(), speech_sample_rate)

# 1 没有问题！