from datasets import load_dataset, DatasetDict
from datasets import load_from_disk

# ds = load_dataset("gpt-omni/VoiceAssistant-400K")
# save_path = "/valleblob/v-wenxichen/data/s2s"
# ds.save_to_disk(save_path)
# print(f"数据集已保存到 {save_path}")

# 选择训练集的 0.1% 数据
# fraction = 0.001
# small_train_ds = ds['train'].select(range(int(len(ds['train']) * fraction)))

# 创建一个新的 DatasetDict 并将小规模数据集放入 'train' 键中
# small_ds = DatasetDict({'train': small_train_ds})

# # 保存抽取的部分数据到指定路径
# save_path = "/valleblob/v-wenxichen/data/s2s/simple_speech_dataset"
# small_ds.save_to_disk(save_path)

# print(f"已将 0.1% 的训练数据集保存到 {save_path}/train")

# ds = load_from_disk(save_path)
# print(ds)
# print(ds['train'][0])
# for key in ds[0].keys():
#     print(key, ds[0][key])


parquet_dir = "/valleblob/v-wenxichen/data/s2s/ultrachat/parquet"
ds = load_dataset(parquet_dir)
print(ds)
print(ds['train'][0])

save_dir = "/valleblob/v-wenxichen/data/s2s/ultrachat-v1"
ds.save_to_disk(save_dir)