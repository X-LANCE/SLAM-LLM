import os

import numpy as np
from sklearn import metrics

from tqdm import tqdm
import openai

openai.api_key = "your-openai-api-key"

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return np.array(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'])

def calculate_stats(output, target):
    classes_num = target.shape[-1]
    stats = []

    for k in range(classes_num):
        avg_precision = metrics.average_precision_score(target[:, k], output[:, k], average=None)
        dict = {'AP': avg_precision}
        stats.append(dict)

    return stats

labels_path = 'https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/class_labels_indices_subset.csv'
embeds_npy_path = 'https://huggingface.co/datasets/zhisheng01/SpatialAudio/blob/main/SpatialSoundQA/AudioSet/metadata/audioset_class_embeds.npy'

label2id = {}
with open(labels_path) as f:
    for idx, line in enumerate(f.readlines()[1:]):
        label = line.strip().split(',', 2)[-1]
        label2id[label.lower()] = idx
#         label2emb.append(get_embedding(label))

# label2emb = np.stack(label2emb)
# np.save(embeds_npy_path, label2emb)

total_labels_embeddings = np.load(embeds_npy_path)

one_hot_embeds = np.eye(355)

with open("decode_eval-stage2-classification_beam4_gt") as gt_f:
    gt_lines = gt_f.readlines()
    targets = []
    for line in gt_lines:
        target = np.array([one_hot_embeds[label2id[i]] for i in line.strip().split('\t', 1)[1].split("; ")]).sum(axis=0)
        targets.append(target)
    targets = np.stack(targets)


with open("decode_eval-stage2-classification_beam4_pred") as pred_f:
    pred_lines = pred_f.readlines()
    preds = []
    for line in tqdm(pred_lines):
        pred = line.strip().split('\t', 1)[1]
        pred = get_embedding(pred)
        pred = np.array([cosine_similarity(pred, embed) for embed in total_labels_embeddings])
        preds.append(pred)

    preds = np.stack(preds)

stats = calculate_stats(preds, targets)

AP = [stat['AP'] for stat in stats]
mAP = np.mean([stat['AP'] for stat in stats])
print("mAP: {:.6f}".format(mAP))
