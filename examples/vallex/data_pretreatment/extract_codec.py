import os
import argparse
import numpy as np
from tqdm import tqdm
from encodec import EncodecModel
import math
import torch
import torchaudio
import sys
from fairseq import search, utils
import pandas as pd
import librosa as lib
from multiprocessing import  Process

def get_codec(model, audio_path, device, resampleers):
    with torch.no_grad():
        audio, sr = torchaudio.load(audio_path)
        if audio.size(1) < 16000:
            return None
        # audio, sr = lib.load(audio_path, sr=16000)
        # audio = torch.tensor([audio])
        # duration = len(audio[0]) / sr
        en_audio = audio.unsqueeze(0).to(device)
        en_audio = convert_audio(en_audio, sr, model.channels, resampleers)
        encoded_frames = model.encode(en_audio)
        # dim, nframe
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze(0).detach().cpu().numpy()
        return codes # dim, nframe
    
def convert_audio(wav: torch.Tensor, sr: int, target_channels: int, resampleers):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    # wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    wav = resampleers[sr](wav)
    return wav
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-home', type=str, help='out home')
    parser.add_argument('--tsv', type=str, help='out home')
    parser.add_argument('--pro-idx', type=int, help='out home')
    parser.add_argument('--pro-total', type=int, help='out home')
    args = parser.parse_args()
    
    model = EncodecModel.encodec_model_24khz().cuda().eval()
    model.set_target_bandwidth(6.0) # 1.5, 3, 6, 12, 24
    device = next(model.parameters()).device
    
    tsv = os.path.join(args.tsv)
    infos = utils.read_file(tsv) # path, dur
    
    resampleers = {
        16000: torchaudio.transforms.Resample(16000, model.sample_rate).to(device).eval(),
        44100: torchaudio.transforms.Resample(44100, model.sample_rate).to(device).eval(),
        48000: torchaudio.transforms.Resample(48000, model.sample_rate).to(device).eval(),
        24000: torchaudio.transforms.Resample(24000, model.sample_rate).to(device).eval(),
        8000: torchaudio.transforms.Resample(8000, model.sample_rate).to(device).eval(),
        22050: torchaudio.transforms.Resample(22050, model.sample_rate).to(device).eval(),
    }
    
    slice_len = len(infos) // args.pro_total
    start = args.pro_idx * slice_len
    if args.pro_idx == args.pro_total - 1:
        infos = tqdm(infos[start:])
    else:
        end = (args.pro_idx + 1) * slice_len
        infos = tqdm(infos[start:end])
    print("start:%d, len:%d"%(start, len(infos)))
    
    wfs = [open(os.path.join(args.save_home, "%d_codec%d.tsv"%(args.pro_idx, i)), "w") for i in range(8)]
    
    for step, row in enumerate(infos):
        row = row.strip()
        temp_path, dur = row.split("\t")
        codec = get_codec(model, temp_path, device, resampleers)
        if codec is None:
            continue
        for i in range(8):
            codecli = codec[i, :].flatten().astype(str)
            codecli = " ".join(codecli)
            print("%s\t%s"%(temp_path, codecli), file=wfs[i])
            
    for i in wfs:
        i.close()