from datasets import load_dataset, load_from_disk
import pdb
from tqdm import tqdm
import json
import numpy as np
from scipy.io.wavfile import write

# train_data_path = "/nfs/yangguanrou.ygr/data/VoiceAssistant-400K-v2/VoiceAssistant-400K-v2"
# train_data_path = "/nfs/yangguanrou.ygr/data/Belle"
train_data_path = "/nfs/yangguanrou.ygr/data/Belle_debug/data"
# train_data_path = "/nfs/yangguanrou.ygr/data/parquet_data_test/en"
load_from_cache_file = True
seed=42
split_size=0.01
if load_from_cache_file:       
    ds = load_dataset(train_data_path)       # load_from huggingface datasets
else:
    ds = load_from_disk(train_data_path)   # load_from local disk

train_val_split = ds['train'].train_test_split(test_size=split_size, seed=seed)
train_data_list = train_val_split["train"]
val_data_list = train_val_split['test']

# train_json_path="/nfs/yangguanrou.ygr/data/VoiceAssistant-400K-v2/train.jsonl"
# val_json_path="/nfs/yangguanrou.ygr/data/VoiceAssistant-400K-v2/val.jsonl"
# train_json_path = "/nfs/yangguanrou.ygr/data/Belle/train.jsonl"
# val_json_path = "/nfs/yangguanrou.ygr/data/Belle/val.jsonl"
train_json_path = "/nfs/yangguanrou.ygr/data/Belle_debug/manifest/train.jsonl"
val_json_path = "/nfs/yangguanrou.ygr/data/Belle_debug/manifest/val.jsonl"
# train_json_path = "/nfs/yangguanrou.ygr/data/parquet_data_test/train_debug.jsonl"
# val_json_path = "/nfs/yangguanrou.ygr/data/parquet_data_test/val_debug.jsonl"
root_path="/cpfs_speech2/yangguanrou.ygr/Belle/question_audio/"

with open(val_json_path, 'w') as out_f:
    for data in tqdm(val_data_list,total=len(val_data_list)):
        audio_array = np.array(data['question_audio']['array'])
        sampling_rate=data['question_audio']['sampling_rate']
        path=data['question_audio']['path']
        # write(root_path+str(data['index'])+'.wav', sampling_rate, audio_array)
        # exit()
        data_dict = {
            'key': data['index'],
            'question': data['question'],
            'source_text': data['answer'],
            'target_text': data['answer'],
            'answer_cosyvoice_speech_token': data['answer_cosyvoice_speech_token'],
        }
        out_f.write(json.dumps(data_dict) + '\n')
        # pdb.set_trace()

with open(train_json_path, 'w') as out_f:
    for data in tqdm(train_data_list,total=len(train_data_list)):
        audio_array = np.array(data['question_audio']['array'], dtype=np.int16)
        sampling_rate=data['question_audio']['sampling_rate']
        path=data['question_audio']['path']
        # write(root_path+path, sampling_rate, audio_array)
        data_dict = {
            'key': data['index'],
            'question': data['question'],
            'source_text': data['answer'],
            'target_text': data['answer'],
            'answer_cosyvoice_speech_token': data['answer_cosyvoice_speech_token'],
        }
        out_f.write(json.dumps(data_dict) + '\n')
