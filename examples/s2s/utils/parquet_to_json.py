from datasets import load_dataset, load_from_disk

train_data_path = "/nfs/yangguanrou.ygr/data/VoiceAssistant-400K-v2/VoiceAssistant-400K-v2"
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

train_json_path="/nfs/yangguanrou.ygr/data/VoiceAssistant-400K-v2/train.json"
val_json_path="/nfs/yangguanrou.ygr/data/VoiceAssistant-400K-v2/val.jsonl"

val_data_list.to_json(val_json_path)
train_data_list.to_json(train_json_path)