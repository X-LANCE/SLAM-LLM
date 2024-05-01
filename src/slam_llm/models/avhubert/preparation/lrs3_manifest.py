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
from gen_subword import gen_vocab
from tempfile import NamedTemporaryFile

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LRS3 tsv preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    parser.add_argument('--valid-ids', type=str, help='a list of valid ids')
    parser.add_argument('--vocab-size', type=int, default=1000, help='a list of valid ids')
    args = parser.parse_args()
    file_list, label_list = f"{args.lrs3}/file.list", f"{args.lrs3}/label.list"
    assert os.path.isfile(file_list) , f"{file_list} not exist -> run lrs3_prepare.py first"
    assert os.path.isfile(label_list) , f"{label_list} not exist -> run lrs3_prepare.py first"
    nframes_audio_file, nframes_video_file = f"{args.lrs3}/nframes.audio", f"{args.lrs3}/nframes.video"
    assert os.path.isfile(nframes_audio_file) , f"{nframes_audio_file} not exist -> run count_frames.py first"
    assert os.path.isfile(nframes_video_file) , f"{nframes_video_file} not exist -> run count_frames.py first"
    print(f"Generating sentencepiece units")
    vocab_size = args.vocab_size
    vocab_dir = (Path(f"{args.lrs3}")/f"spm{vocab_size}").absolute()
    # out_root = Path(vocab_dir).absolute()
    vocab_dir.mkdir(exist_ok=True)
    spm_filename_prefix = f"spm_unigram{vocab_size}"
    with NamedTemporaryFile(mode="w") as f:
        label_text = [ln.strip() for ln in open(label_list).readlines()]
        for t in label_text:
            f.write(t.lower() + "\n")
        gen_vocab(Path(f.name), vocab_dir/spm_filename_prefix, 'unigram', args.vocab_size)
    vocab_path = (vocab_dir/spm_filename_prefix).as_posix()+'.txt'

    audio_dir, video_dir = f"{args.lrs3}/audio", f"{args.lrs3}/video"

    def setup_target(target_dir, train, valid, test):
        for name, data in zip(['train', 'valid', 'test'], [train, valid, test]):
            with open(f"{target_dir}/{name}.tsv", 'w') as fo:
                fo.write('/\n')
                for fid, _, nf_audio, nf_video in data:
                    fo.write('\t'.join([fid, os.path.abspath(f"{video_dir}/{fid}.mp4"), os.path.abspath(f"{audio_dir}/{fid}.wav"), str(nf_video), str(nf_audio)])+'\n')
            with open(f"{target_dir}/{name}.wrd", 'w') as fo:
                for _, label, _, _ in data:
                    fo.write(f"{label}\n")
        shutil.copyfile(vocab_path, f"{target_dir}/dict.wrd.txt")
        return

    fids, labels = [x.strip() for x in open(file_list).readlines()], [x.strip().lower() for x in open(label_list).readlines()]
    nfs_audio, nfs_video = [x.strip() for x in open(nframes_audio_file).readlines()], [x.strip() for x in open(nframes_video_file).readlines()]
    valid_fids = set([x.strip() for x in open(args.valid_ids).readlines()])
    train_all, train_sub, valid, test = [], [], [], []
    for fid, label, nf_audio, nf_video in zip(fids, labels, nfs_audio, nfs_video):
        part = fid.split('/')[0]
        # print(part)
        if part == 'test':
            test.append([fid, label, nf_audio, nf_video])
        else:
            if fid in valid_fids:
                valid.append([fid, label, nf_audio, nf_video])
            else:
                train_all.append([fid, label, nf_audio, nf_video])
                if part == 'trainval':
                    train_sub.append([fid, label, nf_audio, nf_video])
    dir_30h = f"{args.lrs3}/30h_data"
    print(f"Set up 30h dir")
    os.makedirs(dir_30h, exist_ok=True)
    setup_target(dir_30h, train_sub, valid, test)
    dir_433h = f"{args.lrs3}/433h_data"
    print(f"Set up 433h dir")
    os.makedirs(dir_433h, exist_ok=True)
    setup_target(dir_433h, train_all, valid, test)
    return


if __name__ == '__main__':
    main()
