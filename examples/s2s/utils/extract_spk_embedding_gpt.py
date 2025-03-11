import json
import sys
import os
import soundfile as sf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
import torchaudio
import torch
import uuid
from tqdm import tqdm
from cosyvoice.utils.file_utils import load_wav

codec_decoder_path="/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice-300M-SFT"
codec_decoder = CosyVoice(codec_decoder_path, load_jit=False, load_trt=False, fp16=False)

data_jsonl = "/nfs/yangguanrou.ygr/data/gpt4o/test.jsonl"
output_data_jsonl = "/nfs/yangguanrou.ygr/data/gpt4o/test_with_spk.jsonl"

with open(data_jsonl, 'r', encoding='utf-8') as infile, open(output_data_jsonl, 'w', encoding='utf-8') as outfile:
    for line in tqdm(infile):
        item = json.loads(line)
        target_wav = item.get('target_wav')

        prompt_speech_16k = load_wav(target_wav, 16000)
        flow_embedding = codec_decoder.frontend._extract_spk_embedding(prompt_speech_16k) # (1,192)
        # print(flow_embedding.shape)
        item["spk_embedding" ]= flow_embedding.tolist()
        json.dump(item, outfile)
        outfile.write('\n') 
