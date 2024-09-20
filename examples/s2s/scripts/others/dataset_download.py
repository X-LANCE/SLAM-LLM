from datasets import load_dataset
from datasets import load_from_disk

# ds = load_dataset("gpt-omni/VoiceAssistant-400K")

save_path = "/valleblob/v-wenxichen/data/s2s/VoiceAssistant-400K"
# ds.save_to_disk(save_path)

# print(f"数据集已保存到 {save_path}")

ds = load_from_disk(save_path)
print(ds)
print(ds['train'][0])
for key in ds[0].keys():
    print(key, ds[0][key])