# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import argparse
import torch
from fairseq.data import Dictionary, encoders

def add_task_state(ckpt_path):
    std = torch.load(ckpt_path)
    cfg = std['cfg']
    if cfg['model']['_name'] == 'av_hubert':
        dictionaries = [Dictionary.load(f"{cfg['task']['label_dir']}/dict.{label}.txt") for label in cfg['task']['labels']]
        std['cfg']['task']['fine_tuning'] = False
        std['task_state'] = {'dictionaries': dictionaries}
        print(dictionaries, std['cfg']['task'])
    else:
        prt = torch.load(std['cfg']['model']['w2v_path'])
        std['cfg']['model']['w2v_args'] = prt['cfg']
        std['cfg']['task']['fine_tuning'] = True
        dictionaries = [Dictionary.load(f"{prt['cfg']['task']['label_dir']}/dict.{label}.txt") for label in prt['cfg']['task']['labels']]
        target_dictionary = Dictionary.load(f"{cfg['task']['label_dir']}/dict.wrd.txt")
        tokenizer_fn = std['cfg']['task']['tokenizer_bpe_model']
        bpe_args = argparse.Namespace(**{'bpe': 'sentencepiece', f"sentencepiece_model": tokenizer_fn})
        bpe_tokenizer = encoders.build_bpe(bpe_args)
        std['task_state'] = {'dictionaries': dictionaries, 'target_dictionary': target_dictionary, 's2s_tokenizer': bpe_tokenizer}
    torch.save(std, ckpt_path)
    return

if __name__ == '__main__':
    ckpt_paths = sys.argv[1:]
    for ckpt_path in ckpt_paths:
        add_task_state(ckpt_path)
