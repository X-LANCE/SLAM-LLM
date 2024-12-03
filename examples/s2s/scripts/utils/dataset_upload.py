from datasets import load_dataset, load_from_disk

parquet_dir = "/tmp/amlt-code-download/VoiceAssistant-400K-v2"
ds = load_dataset(parquet_dir)
# ds = load_from_disk(parquet_dir)
print(ds)
print(ds['train'][0])

ds.push_to_hub("worstchan/VoiceAssistant-400K-SLAM-Omni")