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
import json

def load_tsv(path):
    with open(path, "r") as rf:
        lines = rf.readlines()
    return lines
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--gt', type=str, default="/nfs/yangguanrou.ygr/data/Emotion/all/cosyvoice_semantic_token_50HZ/dev_random_200.jsonl")
    parser.add_argument('--pred', type=str)
    args = parser.parse_args()

    pred_dir = os.path.join(args.pred, "pred_audio/default_tone")
    output_path = os.path.join(args.pred, "emo.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel(model="iic/emotion2vec_plus_large").to(device)
    simis = []

    with torch.no_grad():
        with open(args.gt, "r") as rf, open(output_path, "w") as f:
            for line in rf:
                data = json.loads(line.strip())
                id =data["key"]
                gt_path=data["target_wav"]
                pred_path=pred_dir+'/'+id+'.wav'
         
                if not os.path.exists(pred_path):
                    print(pred_path)
                    continue
                    exit()

                pred_emb = model.generate(pred_path, granularity="utterance", extract_embedding=True)[0]["feats"] # 1024
                tgt_emb = model.generate(gt_path, granularity="utterance", extract_embedding=True)[0]["feats"] # 1024
                print(pred_emb.device)
                print(tgt_emb.device)
                # simi = float(F.cosine_similarity(torch.FloatTensor([pred_emb]), torch.FloatTensor([tgt_emb])).item())
                simi = float(F.cosine_similarity(pred_emb.unsqueeze(0), tgt_emb.unsqueeze(0)).item())
                simis.append(simi)

                
                print("%s %s %f"%(pred_path, gt_path, simi), file=f)
                exit()
            print("------------------------------------------", file=f)
            print("len:", len(simis),file=f)
            print("emo2vec large:", np.mean(simis), file=f)
        