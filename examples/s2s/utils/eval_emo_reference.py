import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os
import torch.nn.functional as F
import soundfile as sf
from torchaudio.transforms import Resample
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel

def load_tsv(path):
    with open(path, "r") as rf:
        lines = rf.readlines()
    return lines
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-t', '--tsv', type=str)
    parser.add_argument('-o', '--out_home', type=str)
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.out_home, "emo.log")):
        with open(os.path.join(args.out_home, "emo.log"), "r") as rf:
            if len(rf.readlines()) > 500:
                exit()
    model = AutoModel(model="iic/emotion2vec_plus_large")

    with open(args.tsv, "r") as rf:
        lines = rf.readlines()
    simis = []
    with torch.no_grad():
        with open(os.path.join(args.out_home, "emo.log"), "w") as f:
            for idx, line in enumerate(tqdm(lines)):
                path, trans, dur, emo, trans_len, spk, \
                tgt_path, tgt_trans, tgt_dur, tgt_emo, tgt_translen, tgt_spk = line.strip().split("\t")
                save_name = str(idx) + '_' + os.path.basename(path).split(".")[0] + "_srcspk%s"%spk + "_tgtspk%s"%tgt_spk + "_srcemo%s"%emo + "_tgtemo%s"%tgt_emo + ".wav"
                
                generated_wav = os.path.join(args.out_home, save_name)
                
                generated_emb = model.generate(generated_wav, granularity="utterance", extract_embedding=True)[0]["feats"] # 1024
                tgt_emb = model.generate(tgt_path, granularity="utterance", extract_embedding=True)[0]["feats"] # 1024
                simi = float(F.cosine_similarity(torch.FloatTensor([generated_emb]), torch.FloatTensor([tgt_emb])).item())
                simis.append(simi)
                
                print("%s %s %f"%(generated_wav, tgt_path, simi), file=f)
            print("------------------------------------------", file=f)
            print("emo2vec large:", np.mean(simis), file=f)
        