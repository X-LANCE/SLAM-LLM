import os
import whisper
from tqdm import tqdm
import torch
import argparse



device = "cuda" if torch.cuda.is_available() else "cpu"
# 初始化 whisper-large-v3 模型
model = whisper.load_model("/nfs/maziyang.mzy/models/Whisper/large-v3.pt", device=device)
audio_filepath = "/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/belle_pretrain/s2s_epoch_2_step_40841/tts_decode_test_rp_seed_greedy_secap_test/pred_audio/default_tone/tx_xiao_0200106000982.wav"

# 单句转录
result = model.transcribe(audio_filepath, language="zh")
transcription = result['text'].strip()
# 写入文件名和转录文本
print(transcription)

