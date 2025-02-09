import torch

file_path = '/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/belle/tts/tn_remake/embed_tokens_infer.pth'
data = torch.load(file_path, map_location=torch.device('cpu'))
print(data)