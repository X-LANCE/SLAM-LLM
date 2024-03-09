# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math, time
import os, sys, subprocess, glob, re
import numpy as np
from collections import defaultdict
from scipy.io import wavfile
from tqdm import tqdm

def make_musan_tsv(musan_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = 16_000
    min_dur, max_dur = 3*sample_rate, 11*sample_rate
    part_ratios = zip(['train', 'valid', 'test'], [0.8, 0.1, 0.1])
    all_fns = {}
    nfs = f"{musan_root}/nframes.audio"
    nfs = dict([x.strip().split('\t') for x in open(nfs).readlines()])
    for category in ['babble', 'music', 'noise']:
        wav_fns = glob.glob(f"{musan_root}/{category}/*/*wav")
        target_fns = []
        for wav_fn in tqdm(wav_fns):
            dur = int(nfs[os.path.abspath(wav_fn)])
            if dur >= min_dur and dur < max_dur:
                target_fns.append(wav_fn)
        print(f"{category}: {len(target_fns)}/{len(wav_fns)}")
        all_fns[category] = target_fns
        output_subdir = f"{output_dir}/{category}"
        os.makedirs(output_subdir, exist_ok=True)
        num_train, num_valid, num_test = int(0.8*len(target_fns)), int(0.1*len(target_fns)), int(0.1*len(target_fns))
        if category in {'music', 'noise'}:
            np.random.shuffle(target_fns)
            train_fns, valid_fns, test_fns = target_fns[:num_train], target_fns[num_train:num_train+num_valid], target_fns[num_train+num_valid:]
        elif category == 'babble':
            train_fns, valid_fns, test_fns = [], [], []
            for wav_fn in target_fns:
                split_id = os.path.basename(wav_fn)[:-4].split('-')[0]
                if split_id == 'train':
                    train_fns.append(wav_fn)
                elif split_id == 'valid':
                    valid_fns.append(wav_fn)
                elif split_id == 'test':
                    test_fns.append(wav_fn)
        for x in ['train', 'valid', 'test']:
            x_fns = eval(f"{x}_fns")
            x_fns = [os.path.abspath(x_fn) for x_fn in x_fns]
            print(os.path.abspath(output_subdir), x, len(x_fns))
            with open(f"{output_subdir}/{x}.tsv", 'w') as fo:
                fo.write('\n'.join(x_fns)+'\n')
    return

def combine(input_tsv_dirs, output_dir):
    output_subdir = f"{output_dir}/all"
    os.makedirs(output_subdir, exist_ok=True)
    num_train_per_cat = 20_000
    train_fns, valid_fns, test_fns = [], [], []
    for input_tsv_dir in input_tsv_dirs:
        train_fn, valid_fn, test_fn = [ln.strip() for ln in open(f"{input_tsv_dir}/train.tsv").readlines()], [ln.strip() for ln in open(f"{input_tsv_dir}/valid.tsv").readlines()], [ln.strip() for ln in open(f"{input_tsv_dir}/test.tsv").readlines()]
        num_repeats = int(np.ceil(num_train_per_cat/len(train_fn)))
        train_fn_ = []
        for i in range(num_repeats):
            train_fn_.extend(train_fn)
        train_fn = train_fn_[:num_train_per_cat]
        train_fns.extend(train_fn)
        valid_fns.extend(valid_fn)
        test_fns.extend(test_fn)
    for x in ['train', 'valid', 'test']:
        x_fns = eval(f"{x}_fns")
        print(os.path.abspath(output_subdir), x, len(x_fns))
        with open(f"{output_subdir}/{x}.tsv", 'w') as fo:
            fo.write('\n'.join(x_fns)+'\n')
    return

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Set up noise manifest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--musan', type=str, help='MUSAN root')
    parser.add_argument('--lrs3', type=str, help='LRS3 root')
    args = parser.parse_args()
    short_musan, output_tsv_dir = f"{args.musan}/short-musan", f"{args.musan}/tsv"
    print(f"Make tsv for babble, music, noise")
    make_musan_tsv(short_musan, output_tsv_dir)
    print(f"Combine tsv")
    input_tsv_dirs = [f"{output_tsv_dir}/{x}" for x in ['noise', 'music', 'babble']] + [f"{args.lrs3}/noise/speech"]
    combine(input_tsv_dirs, output_tsv_dir)
    return


if __name__ == '__main__':
    main()
