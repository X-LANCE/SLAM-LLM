from datasets import load_dataset, load_from_disk
import pdb
from tqdm import tqdm
import json
import numpy as np
from scipy.io.wavfile import write

# train_data_path = "/nfs/yangguanrou.ygr/data/Belle"
train_data_path = "/nfs/yangguanrou.ygr/data/Belle/part_0.parquet"

load_from_cache_file = True
seed=42
split_size=0.01
if load_from_cache_file:       
    # ds = load_dataset(train_data_path)       # load_from huggingface datasets
    ds = load_dataset("parquet", data_files=train_data_path)
else:
    ds = load_from_disk(train_data_path)   # load_from local disk

train_val_split = ds['train'].train_test_split(test_size=split_size, seed=seed)
train_data_list = train_val_split["train"]
val_data_list = train_val_split['test']

train_json_path = "/nfs/yangguanrou.ygr/data/Belle_debug/manifest/train.jsonl"
val_json_path = "/nfs/yangguanrou.ygr/data/Belle_debug/manifest/val.jsonl"



with open(val_json_path, 'w') as out_f:
    for data in tqdm(val_data_list,total=len(val_data_list)):
        text = data['answer'].replace('\n', '')     #算了我觉得先不处理了
        data_dict = {
            'key': data['index'],
            'source_text': text,
            'target_text': text,
            'answer_cosyvoice_speech_token': data['answer_cosyvoice_speech_token'],
        }
        out_f.write(json.dumps(data_dict,ensure_ascii=False) + '\n')
        # pdb.set_trace()

with open(train_json_path, 'w') as out_f:
    for data in tqdm(train_data_list,total=len(train_data_list)):
        text = data['answer'].replace('\n', '')   
        data_dict = {
            'key': data['index'],
            'source_text': text,
            'target_text': text,
            'answer_cosyvoice_speech_token': data['answer_cosyvoice_speech_token'],
        }
        out_f.write(json.dumps(data_dict,ensure_ascii=False) + '\n')
