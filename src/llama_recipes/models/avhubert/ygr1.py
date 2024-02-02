import torch

import fairseq
import hubert_pretraining, hubert, hubert_asr  #果然加上这个就可以了！！！

import torch
# import argparse
# parser = argparse. Argumentarser (description= 'Process some integers.')
# parser.add_argument ('--user_dir',default="/mnt/lustre/sjtu/home/gry10/FastHuBERT")

# args = parser.parse_args ()
# import_user_module(args)

# audio = torch.rand((1, 67360))
audio = torch.rand((1, 67360, 1))
# visual = torch.rand((1, 106, 1, 112, 112))
visual = torch.rand((1, 1, 106, 112, 112))
batch={}
batch['audio'] = None
batch['video'] = visual
padding_mask = torch.zeros((1, 67360), dtype=torch.bool)

ckpt_path = "/nfs/yangguanrou.ygr/av_hubert/large_vox_433h.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
result = model(source=batch, padding_mask=None, prev_output_tokens=None)
# result = model(source=batch)
print("yes")