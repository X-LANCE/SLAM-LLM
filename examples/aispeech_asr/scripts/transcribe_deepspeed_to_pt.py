import argparse
import torch
import torch_npu
import  sys
in_path = sys.argv[1]
out_path = sys.argv[2]
weight_dict = torch.load(in_path)["module"]
torch.save(weight_dict, f"{out_path}/model.pt")
print("[Finish]")