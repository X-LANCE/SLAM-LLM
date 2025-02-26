import sys
import os
import json
import soundfile as sf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
import torchaudio
import torch
import uuid
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 定义文件路径
input_jsonl = '/nfs/yangguanrou.ygr/data/gpt4o_third/test_cosyvoice2/test_cosyvoice2.jsonl'
# output_dir = '/nfs/yangguanrou.ygr/data/gpt4o_third/test_cosyvoice2/reconstruct_cosyvoice2/'
output_dir = '/nfs/yangguanrou.ygr/data/gpt4o_third/test_cosyvoice2/reconstruct_cosyvoice2_prompt_1/pred_audio/default_tone'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 初始化CosyVoice2
codec_decoder_path = "/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice2-0.5B"
codec_decoder = CosyVoice2(codec_decoder_path, load_jit=False, load_trt=False, fp16=False)

# audio_prompt_path="/nfs/yangguanrou.ygr/codes/CosyVoice/zero_shot_prompt.wav" #希望你以后能做的比我还好哟
audio_prompt_path="/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/prompt/prompt_1.wav"



prompt_speech_16k = load_wav(audio_prompt_path, 16000)
flow_prompt_speech_token, flow_prompt_speech_token_len = codec_decoder.frontend._extract_speech_token(prompt_speech_16k)
prompt_speech_24000 = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)(prompt_speech_16k)
prompt_speech_feat, prompt_speech_feat_len = codec_decoder.frontend._extract_speech_feat(prompt_speech_24000)
flow_embedding = codec_decoder.frontend._extract_spk_embedding(prompt_speech_16k)

# 设置音频生成速度
speed = 1.0

# 打开 JSONL 文件并逐行处理
with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        key = data['key']
        audio_tokens = data['answer_cosyvoice_speech_token']

        # 将音频令牌转换为张量
        audio_tokens = torch.tensor(audio_tokens).unsqueeze(0)

        # 生成音频
        audio_hat = codec_decoder.model.token2wav(
            token=audio_tokens,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=str(uuid.uuid1()),
            token_offset=0,
            finalize=True,
            speed=speed
        )

        # 生成的音频文件名
        output_file = os.path.join(output_dir, f"{key}.wav")

        # 保存音频
        sf.write(output_file, audio_hat.squeeze().cpu().numpy(), 24000)


print("所有音频文件已生成！")