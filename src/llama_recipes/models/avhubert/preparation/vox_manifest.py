# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path

def main():
    import argparse
    parser = argparse.ArgumentParser(description='VoxCeleb2 tsv preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vox', type=str, help='VoxCeleb2 root dir')
    parser.add_argument('--en-ids', type=str, help='a list of English-utterance ids')
    args = parser.parse_args()
    file_list = f"{args.vox}/file.list"
    assert os.path.isfile(file_list) , f"{file_list} not exist -> run vox_prepare.py first"
    nframes_audio_file, nframes_video_file = f"{args.vox}/nframes.audio", f"{args.vox}/nframes.video"
    assert os.path.isfile(nframes_audio_file) , f"{nframes_audio_file} not exist -> run count_frames.py first"
    assert os.path.isfile(nframes_video_file) , f"{nframes_video_file} not exist -> run count_frames.py first"

    audio_dir, video_dir = f"{args.vox}/audio", f"{args.vox}/video"

    def setup_target(target_dir, train):
        for name, data in zip(['train'], [train]):
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, nf_audio, nf_video in data:
                    fo.write('\t'.join([fid, os.path.abspath(f"{video_dir}/{fid}.mp4"), os.path.abspath(f"{audio_dir}/{fid}.wav"), str(nf_video), str(nf_audio)])+'\n')
        return

    fids = [x.strip() for x in open(file_list).readlines()]
    nfs_audio, nfs_video = [x.strip() for x in open(nframes_audio_file).readlines()], [x.strip() for x in open(nframes_video_file).readlines()]
    en_fids = set([x.strip() for x in open(args.en_ids).readlines()])
    train_all, train_sub = [], []
    for fid, nf_audio, nf_video in zip(fids, nfs_audio, nfs_video):
        if fid in en_fids:
            train_sub.append([fid, nf_audio, nf_video])
        train_all.append([fid, nf_audio, nf_video])
    dir_en = f"{args.vox}/en_data"
    print(f"Set up English-only dir")
    os.makedirs(dir_en, exist_ok=True)
    setup_target(dir_en, train_sub)
    dir_all = f"{args.vox}/all_data"
    print(f"Set up all data dir")
    os.makedirs(dir_all, exist_ok=True)
    setup_target(dir_all, train_all)
    return


if __name__ == '__main__':
    main()
