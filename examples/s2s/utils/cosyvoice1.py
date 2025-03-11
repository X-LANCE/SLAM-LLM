import sys
sys.path.append('/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/utils/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import pandas as pd
import os
from multiprocessing import Process
import re
import json

cosyvoice = CosyVoice("/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice-300M-Instruct")

json_file_path = '/nfs/yangguanrou.ygr/data/gpt4o_final_paper/test.jsonl'
# output_dir='/nfs/yangguanrou.ygr/data/MM/cosyvoice/1/prompt1/pred_audio/gt_tone'
# output_dir='/nfs/yangguanrou.ygr/data/MM/cosyvoice/1/prompt2/pred_audio/gt_tone'
# output_dir='/nfs/yangguanrou.ygr/data/MM/cosyvoice/1/prompt3/pred_audio/gt_tone'
output_dir='/nfs/yangguanrou.ygr/data/MM/cosyvoice/1/prompt4/pred_audio/gt_tone'
os.makedirs(output_dir, exist_ok=True)

with open(json_file_path, 'r', encoding='utf-8') as json_file:
    for line in json_file:
        row = json.loads(line.strip())
        audio_id = row['key']
        text = row['source_text']
        emotion_text_prompt = row['emotion_text_prompt']
        emotion_text_prompt = re.sub(r'[。！？\.,!\?]$', '', emotion_text_prompt) #从字符串的结尾处移除标点符号。
        emotion_text_prompt = re.sub(r'\.(?=.)', ',', emotion_text_prompt) #把句中的. 换成,
        # instruct_str = f"Say this sentence with emotion of {emotion_text_prompt}."
        # instruct_str = f"Please speak with {emotion_text_prompt} emotion."
        # instruct_str = f"Read the following sentence with a tone of {emotion_text_prompt}."
        instruct_str = f"Say this with {emotion_text_prompt} emotion."

        target_wav = row["target_wav"]
        prompt_speech_16k = load_wav(target_wav, 16000)


        results = cosyvoice.my_inference_instruct(text, prompt_speech_16k, instruct_str, stream=False)
        for i, output_chunk in enumerate(results):
            tts_speech = output_chunk['tts_speech']
            # 以 `id_序号.wav` 存储
            # if len(results) == 1:
            #     out_wav_path = f"{output_dir}/{audio_id}.wav"
            # else:
            out_wav_path = f"{output_dir}/{audio_id}_{i}.wav"
            
            torchaudio.save(out_wav_path, tts_speech, cosyvoice.sample_rate)
            
            print(f"[cosy1]已保存: {out_wav_path}")

    print("[cosy1] cosyvoice1所有语音已生成完毕！")





