import argparse
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import json

def process_data(id, load_from_cache_file=True, seed=42, split_size=0.003):
    train_data_path = f"/nfs/yangguanrou.ygr/data/Belle/part_{id}.parquet"

    if load_from_cache_file:       
        ds = load_dataset("parquet", data_files=train_data_path)
    else:
        ds = load_from_disk(train_data_path)

    train_val_split = ds['train'].train_test_split(test_size=split_size, seed=seed)
    train_data_list = train_val_split["train"]
    val_data_list = train_val_split['test']

    train_json_path = f"/nfs/yangguanrou.ygr/data/Belle_manifest/train_{id}.jsonl"
    val_json_path = f"/nfs/yangguanrou.ygr/data/Belle_manifest/val_{id}.jsonl"

    with open(val_json_path, 'w') as out_f:
        for data in tqdm(val_data_list, total=len(val_data_list)):
            data_dict = {
                'key': data['index'],
                'source_text': data['answer'],
                'target_text': data['answer'],
                'answer_cosyvoice_speech_token': data['answer_cosyvoice_speech_token'],
            }
            out_f.write(json.dumps(data_dict,ensure_ascii=False) + '\n')

    with open(train_json_path, 'w') as out_f:
        for data in tqdm(train_data_list, total=len(train_data_list)):
            data_dict = {
                'key': data['index'],
                'source_text': data['answer'],
                'target_text': data['answer'],
                'answer_cosyvoice_speech_token': data['answer_cosyvoice_speech_token'],
            }
            out_f.write(json.dumps(data_dict,ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data files based on the given ID.")
    parser.add_argument('id', type=int, help='The ID of the data part to process')
    args = parser.parse_args()

    process_data(args.id)