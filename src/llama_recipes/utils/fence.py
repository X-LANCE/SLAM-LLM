from ast import parse
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from fense.evaluator import Evaluator

def get_system_score(evaluator, ref_dir, cands_dir, dataset):
    cands_df = pd.read_csv(cands_dir)
    if dataset == 'audiocaps':
        ref_df = pd.read_csv(ref_dir)
        assert len(cands_df) == (len(ref_df) // 5), "Number of captions should match"
        id2order = {}
        list_refs = []
        for rid, row in ref_df.iterrows():
            id0 = row["youtube_id"]
            caption = row["caption"]
            if id0 in id2order:
                list_refs[id2order[id0]].append(caption)
            else:
                id2order[id0] = len(id2order)
                list_refs.append([caption])
        cands = ["" for _ in range(len(id2order))]
        for rid, row in cands_df.iterrows():
            id0 = row["youtube_id"]
            caption = row["caption"]
            cands[id2order[id0]] = caption
        score = evaluator.corpus_score(cands, list_refs)
    
    elif dataset == 'clotho':
        ref_df = pd.read_csv(ref_dir)
        list_refs = ref_df.iloc[:, 1:].values.tolist()
        id2order = {id0: order for order, id0 in enumerate(ref_df["file_name"])}
        cands = ["" for _ in range(len(id2order))]
        for rid, row in cands_df.iterrows():
            id0 = row["file_name"]
            caption = row["caption"]
            cands[id2order[id0]] = caption
        score = evaluator.corpus_score(cands, list_refs)

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sbert_model", default="paraphrase-TinyBERT-L6-v2")
    parser.add_argument("--echecker_model", default="echecker_clotho_audiocaps_base", choices=["echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny"])
    parser.add_argument("--cands_dir", default="./test_data/audiocaps_cands.csv")
    parser.add_argument("--ref_dir", default="./test_data/audiocaps_eval.csv")
    parser.add_argument("--dataset", default="audiocaps", choices=["audiocaps", "clotho"])
    args = parser.parse_args()
    print(args)
    evaluator = Evaluator(device=args.device, sbert_model=args.sbert_model, echecker_model=args.echecker_model)
    score = get_system_score(evaluator, args.ref_dir, args.cands_dir, args.dataset)
    print(f"Avg FENSE score on {args.dataset}: {score:.5f}")
