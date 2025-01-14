from typing import Any
import torch
import torchaudio
import soundfile as sf
from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
class AudioTokenizer:
    def __init__(
        self,
        device: Any = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

    @property
    def device(self):
        return self._device

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def decode(self, token: torch.Tensor) -> torch.Tensor:
        raise NotImplemented
        
class CosyVoiceTokenizer(AudioTokenizer):
    """CosyVoice."""

    def __init__(
        self,
        model_path: str = "download/CosyVoice-300M",
        device: Any = None,
    ) -> None:
        # Instantiate CosyVoice model
        self.tokenizer = CosyVoice(model_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        token, token_len = self.tokenizer.frontend._extract_speech_token(audio)
        return token.unsqueeze(-1), token_len

    def decode(
        self,
        token: torch.Tensor,
        prompt_audio: torch.Tensor,
    ) -> torch.Tensor:
        token_len = torch.tensor([token.size(1)], dtype=torch.int32)
        prompt_token, prompt_token_len = self.tokenizer.frontend._extract_speech_token(
            prompt_audio
        )
        speaker_embedding = self.tokenizer.frontend._extract_spk_embedding(prompt_audio)
        prompt_audio = torchaudio.transforms.Resample(orig_freq=16000, new_freq=22050)(
            prompt_audio
        )
        prompt_feat, prompt_feat_len = self.tokenizer.frontend._extract_speech_feat(
            prompt_audio
        )

        flow_cache = torch.zeros(1, 80, 0, 2)
        mel, _ = self.tokenizer.model.flow.inference(
            token=token.to(self.device),
            token_len=token_len.to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=prompt_token_len.to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=prompt_feat_len.to(self.device),
            embedding=speaker_embedding.to(self.device),
            flow_cache=flow_cache.to(self.device),
        )

        hift_cache = torch.zeros(1, 1, 0)
        audio, _ = self.tokenizer.model.hift.inference(
            speech_feat=mel, cache_source=hift_cache.to(self.device)
        )

        return audio

    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        speaker_embedding = self.tokenizer.frontend._extract_spk_embedding(audio)
        return speaker_embedding


class CosyVoice2Tokenizer(AudioTokenizer):
    """CosyVoice2."""

    def __init__(
        self,
        model_path: str = "download/CosyVoice2-0.5B",
        device: Any = None,
    ) -> None:
        # Instantiate CosyVoice2 model
        self.tokenizer = CosyVoice2(model_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        token, token_len = self.tokenizer.frontend._extract_speech_token(audio)
        return token.unsqueeze(-1), token_len

    def decode(
        self,
        token: torch.Tensor,
        prompt_audio: torch.Tensor,
    ) -> torch.Tensor:
        token_len = torch.tensor([token.size(1)], dtype=torch.int32)
        prompt_token, prompt_token_len = self.tokenizer.frontend._extract_speech_token(
            prompt_audio
        )
        speaker_embedding = self.tokenizer.frontend._extract_spk_embedding(prompt_audio)
        prompt_audio = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(
            prompt_audio
        )
        prompt_feat, prompt_feat_len = self.tokenizer.frontend._extract_speech_feat(
            prompt_audio
        )

        mel, _ = self.tokenizer.model.flow.inference(
            token=token.to(self.device),
            token_len=token_len.to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=prompt_token_len.to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=prompt_feat_len.to(self.device),
            embedding=speaker_embedding.to(self.device),
            finalize=True,
        )

        hift_cache = torch.zeros(1, 1, 0)
        audio, _ = self.tokenizer.model.hift.inference(
            speech_feat=mel, cache_source=hift_cache.to(self.device)
        )

        return audio

    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        speaker_embedding = self.tokenizer.frontend._extract_spk_embedding(audio)
        return speaker_embedding

prompt_audio_path="/nfs/yangguanrou.ygr/codes/CosyVoice/zero_shot_prompt.wav"
from cosyvoice.utils.file_utils import load_wav
prompt_audio = load_wav(prompt_audio_path, 16000)

my_tok=CosyVoice2Tokenizer(model_path="/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice2-0.5B")
#audio_tokens = [536, 693, 561, 561, 561, 561, 561, 561, 561, 66, 66, 66, 66, 66, 66, 66, 249, 685, 685, 3190, 710, 710, 293, 310, 468, 1700, 1700, 468, 472, 472, 468, 269, 2723, 2344, 79, 311, 563, 425, 77, 550, 312, 2644, 312, 613, 253, 613, 2297, 12, 2891, 448, 448, 4, 386, 290, 308, 2752, 20, 69, 460, 460, 460, 466, 465, 212, 50, 387, 590, 311, 51, 51, 51, 51, 51, 384, 164, 415, 139, 139, 121, 2444, 2444, 249, 249, 262, 249, 287, 3821, 124, 211, 733, 515, 227, 227, 465, 49, 343, 1354, 1354, 47, 104, 33, 33, 27, 2302, 681, 33, 179, 39, 2031, 90, 297, 263, 982, 1091, 1091, 132, 61, 550, 1006, 1006, 550, 535, 535, 390, 406, 457, 434, 298, 54, 1073, 355, 15, 553, 426, 426, 3290, 7, 215, 57, 345, 1423, 249, 19, 271, 149, 262, 680, 657, 327, 1849, 249, 670, 230, 230, 230, 3238, 3238, 3238, 3238, 3238, 53, 678, 1089, 124, 226, 726, 726, 212, 335, 217, 11, 20, 534, 534, 534, 534, 3052, 700, 47, 47, 1930, 595, 59, 2330, 473, 2592, 212, 515, 212, 212, 212, 212, 212, 351, 594, 594, 173, 173, 460, 701, 144, 249, 249, 2444, 249, 249, 249, 2444, 436, 436, 121, 149, 377, 3132, 376, 2454, 415, 415, 655, 389, 165, 483, 183, 183, 515, 565, 506, 47, 1307, 403, 3254, 502, 1935, 386, 290, 247, 403, 160, 137, 3254, 502, 903, 386, 210, 210, 335, 223, 1006, 1006, 550, 550, 550, 550, 372, 649, 360, 121, 121, 249, 249, 249, 249, 670, 670, 670, 442, 442, 442, 442, 442, 442, 216, 442, 140, 345, 631, 710, 124, 1261, 1935, 3889, 4, 648, 2723, 39, 2031, 1, 184, 184, 446, 122, 192, 361, 1, 1, 107, 130, 187, 9, 115, 284, 28, 706, 77, 3575, 3575, 570, 595, 253, 420, 298, 476, 3836, 618, 540, 540, 621, 519, 9, 1037, 1037, 1037, 2297, 1357, 19, 598, 216, 216, 216, 216, 442, 53, 8, 1089, 596, 67, 256, 286, 300, 609, 260, 289, 250, 570, 15, 15, 713, 177, 90, 90, 39, 157, 127, 3254, 3821, 28, 502, 2188, 4, 4, 386, 386, 448, 269, 269, 199, 199, 199, 1700, 687, 687, 298, 47, 47, 309, 726, 383, 465, 465, 465, 466, 47, 323, 101, 399, 137, 187, 292, 297, 424, 3, 358, 533, 213, 728, 585, 3350, 686, 686, 83, 399, 456, 28, 374, 3290, 2644, 1460, 82, 82, 1035, 1035, 1035, 1035, 262, 262, 262, 505, 291, 331, 331, 331, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66]
audio_tokens = [3707, 6486, 6486, 6486, 6486, 6486, 6405, 6405, 6405, 6405, 6405, 6405, 4218, 2148, 6376, 2261, 5273, 2326, 60, 5019, 4849, 2861, 593, 1970, 1805, 1481, 2050, 6453, 6537, 688, 5594, 4879, 4966, 2841, 4947, 502, 1946, 1460, 224, 3705, 5319, 883, 2909, 3590, 6425, 6421, 4233, 4299, 4218, 3648, 3975, 4218, 4299, 4299, 4299, 5758, 4947, 494, 2702, 3178, 5691, 4668, 4839, 3840, 6044, 4490, 2278, 6501, 6261, 2166, 2539, 6238, 4382, 2698, 2698, 35, 1587, 1045, 734, 3799, 160, 1697, 716, 5102, 3541, 6537, 6537, 2906, 5084, 5774, 6421, 6420, 4299, 3651, 2922, 5109, 5838, 5109, 816, 2922, 6052, 5322, 4440, 4606, 5021, 3563, 1212, 84, 3015, 4852, 4763, 2576, 380, 1271, 1848, 1950, 162, 3321, 2763, 4772, 4615, 3563, 1862, 5889, 6048, 6048, 4104, 6557, 5830, 5080, 4871, 4860, 4860, 4860, 3411, 2681, 1358, 1837, 306, 3161, 569, 2417, 2085, 5408, 2445, 3481, 1294, 1203, 5244, 6057, 5979, 5340, 3074, 1272, 32, 1679, 1942, 1691, 734, 29, 775, 1479, 3648, 3975, 4299, 6486, 6486, 6486, 6405, 6405, 6405, 6405, 4218]
audio_tokens = torch.tensor(audio_tokens)
audio_tokens = audio_tokens.unsqueeze(0)

# audio_wav_path="/nfs/yangguanrou.ygr/data/Premium/WenetSpeech4TTS_Premium_6/wavs/X0000014889_106422087_S00077-S00078.wav"
# audio_wav = load_wav(audio_wav_path, 16000)
# audio_tokens=my_tok.encode(audio_wav)[0]
# audio_tokens= audio_tokens.view(1, -1)
print(audio_tokens.shape)
print(audio_tokens)

audio = my_tok.decode(audio_tokens, prompt_audio)
speech_sample_rate=24000
sf.write("/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/debug_ygr.wav", audio.squeeze().cpu().numpy(), speech_sample_rate)

#[3707, 6486, 6486, 6486, 6486, 6486, 6405, 6405, 6405, 6405, 6405, 6405, 4218, 2148, 6376, 2261, 5273, 2326, 60, 5019, 4849, 2861, 593, 1970, 1805, 1481, 2050, 6453, 6537, 688, 5594, 4879, 4966, 2841, 4947, 502, 1946, 1460, 224, 3705, 5319, 883, 2909, 3590, 6425, 6421, 4233, 4299, 4218, 3648, 3975, 4218, 4299, 4299, 4299, 5758, 4947, 494, 2702, 3178, 5691, 4668, 4839, 3840, 6044, 4490, 2278, 6501, 6261, 2166, 2539, 6238, 4382, 2698, 2698, 35, 1587, 1045, 734, 3799, 160, 1697, 716, 5102, 3541, 6537, 6537, 2906, 5084, 5774, 6421, 6420, 4299, 3651, 2922, 5109, 5838, 5109, 816, 2922, 6052, 5322, 4440, 4606, 5021, 3563, 1212, 84, 3015, 4852, 4763, 2576, 380, 1271, 1848, 1950, 162, 3321, 2763, 4772, 4615, 3563, 1862, 5889, 6048, 6048, 4104, 6557, 5830, 5080, 4871, 4860, 4860, 4860, 3411, 2681, 1358, 1837, 306, 3161, 569, 2417, 2085, 5408, 2445, 3481, 1294, 1203, 5244, 6057, 5979, 5340, 3074, 1272, 32, 1679, 1942, 1691, 734, 29, 775, 1479, 3648, 3975, 4299, 6486, 6486, 6486, 6405, 6405, 6405, 6405, 4218]