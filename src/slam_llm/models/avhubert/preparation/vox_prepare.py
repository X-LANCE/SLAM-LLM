# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, sys, glob, subprocess, json, math
import numpy as np
from scipy.io import wavfile
from os.path import basename, dirname
from tqdm import tqdm
import tempfile, shutil

def get_filelist(root_dir):
    fids = []
    for split in ['dev', 'test']:
        all_fns = glob.glob(f"{root_dir}/{split}/mp4/*/*/*mp4")
        for fn in all_fns:
            fids.append('/'.join(fn.split('/')[-5:])[:-4])
    output_fn = f"{root_dir}/file.list"
    with open(output_fn, 'w') as fo:
        fo.write('\n'.join(fids)+'\n')
    return

def prep_wav(root_dir, wav_dir, flist, ffmpeg, rank, nshard):
    input_dir, output_dir = root_dir, wav_dir
    os.makedirs(output_dir, exist_ok=True)
    fids = [ln.strip() for ln in open(flist).readlines()]
    num_per_shard = math.ceil(len(fids)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    fids = fids[start_id: end_id]
    print(f"{len(fids)} videos")
    for i, fid in enumerate(tqdm(fids)):
        video_fn = f"{input_dir}/{fid}.mp4"
        audio_fn = f"{output_dir}/{fid}.wav"
        os.makedirs(os.path.dirname(audio_fn), exist_ok=True)
        cmd = ffmpeg + " -i " + video_fn + " -f wav -vn -y " + audio_fn + ' -loglevel quiet'
        # print(cmd)
        subprocess.call(cmd, shell=True)
        # print(f"{video_fn} -> {audio_fn}")
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='VoxCeleb2 data preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vox', type=str, help='VoxCeleb2 dir')
    parser.add_argument('--ffmpeg', type=str, help='ffmpeg path')
    parser.add_argument('--step', type=int, help='Steps(1: get file list, 2: extract audio)')
    parser.add_argument('--rank', type=int, help='rank id')
    parser.add_argument('--nshard', type=int, help='number of shards')
    args = parser.parse_args()
    if args.step == 1:
        print(f"Get file list")
        get_filelist(args.vox)
    elif args.step == 2:
        print(f"Extract audio")
        output_dir = f"{args.vox}/audio"
        manifest = f"{args.vox}/file.list"
        prep_wav(args.vox, output_dir, manifest, args.ffmpeg, args.rank, args.nshard)
