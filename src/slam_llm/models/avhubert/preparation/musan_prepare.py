# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import tempfile
import shutil
import submitit
import os, sys, subprocess, glob, re
import numpy as np
from collections import defaultdict
from scipy.io import wavfile
from tqdm import tqdm

def split_musan(musan_root, rank, nshard):
    wav_fns = glob.glob(f"{musan_root}/speech/*/*wav") + glob.glob(f"{musan_root}/music/*/*wav") + glob.glob(f"{musan_root}/noise/*/*wav")
    num_per_shard = math.ceil(len(wav_fns)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    wav_fns = wav_fns[start_id: end_id]
    print(f"{len(wav_fns)} raw audios")
    output_dir = f"{musan_root}/short-musan"
    dur = 10
    for wav_fn in tqdm(wav_fns):
        sample_rate, wav_data = wavfile.read(wav_fn)
        assert sample_rate == 16_000 and len(wav_data.shape) == 1
        if len(wav_data) > dur * sample_rate:
            num_split = int(np.ceil(len(wav_data) / (dur*sample_rate)))
            for i in range(num_split):
                filename = '/'.join(wav_fn.split('/')[-3:])[:-4]
                output_wav_fn = os.path.join(output_dir, filename + f'-{i}.wav')
                sub_data = wav_data[i*dur*sample_rate: (i+1)*dur*sample_rate]
                os.makedirs(os.path.dirname(output_wav_fn), exist_ok=True)
                wavfile.write(output_wav_fn, sample_rate, sub_data.astype(np.int16))
    return

def mix_audio(wav_fns):
    wav_data = [wavfile.read(wav_fn)[1] for wav_fn in wav_fns]
    wav_data_ = []
    min_len = min([len(x) for x in wav_data])
    for item in wav_data:
        wav_data_.append(item[:min_len])
    wav_data = np.stack(wav_data_).mean(axis=0).astype(np.int16)
    return wav_data

def get_speaker_info(musan_root):
    wav_fns = glob.glob(f"{musan_root}/speech/*/*wav")
    spk2wav = {}
    for wav_fn in tqdm(wav_fns):
        speaker = '-'.join(os.path.basename(wav_fn).split('-')[:-1])
        if speaker not in spk2wav:
            spk2wav[speaker] = []
        spk2wav[speaker].append(wav_fn)
    speakers = sorted(list(spk2wav.keys()))
    print(f"{len(speakers)} speakers")
    np.random.shuffle(speakers)
    output_dir = f"{musan_root}/speech/"
    num_train, num_valid = int(len(speakers)*0.8), int(len(speakers)*0.1)
    train_speakers, valid_speakers, test_speakers = speakers[:num_train], speakers[num_train: num_train+num_valid], speakers[num_train+num_valid:]
    for split in ['train', 'valid', 'test']:
        speakers = eval(f"{split}_speakers")
        with open(f"{output_dir}/spk.{split}", 'w') as fo:
            fo.write('\n'.join(speakers)+'\n')
    return

def make_musan_babble(musan_root, rank, nshard):
    babble_dir = f"{musan_root}/babble/wav/"
    num_per_mixture = 30
    sample_rate = 16_000
    num_train, num_valid, num_test = 8000, 1000, 1000
    os.makedirs(babble_dir, exist_ok=True)
    wav_fns = glob.glob(f"{musan_root}/speech/*/*wav")
    spk2wav = {}
    for wav_fn in tqdm(wav_fns):
        speaker = '-'.join(os.path.basename(wav_fn).split('-')[:-1])
        if speaker not in spk2wav:
            spk2wav[speaker] = []
        spk2wav[speaker].append(wav_fn)
    for split in ['train', 'valid', 'test']:
        speakers = [ln.strip() for ln in open(f"{musan_root}/speech/spk.{split}").readlines()]
        num_split = eval(f"num_{split}")
        wav_fns = []
        for x in speakers:
            wav_fns.extend(spk2wav[x])
        print(f"{split} -> # speaker {len(speakers)}, # wav {len(wav_fns)}")
        num_per_shard = math.ceil(num_split/nshard)
        start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
        for i in tqdm(range(num_split)):
            if not (i >= start_id and i < end_id):
                continue
            np.random.seed(i)
            perm = np.random.permutation(len(wav_fns))[:num_per_mixture]
            output_fn = f"{babble_dir}/{split}-{str(i+1).zfill(5)}.wav"
            wav_data = mix_audio([wav_fns[x] for x in perm])
            wavfile.write(output_fn, sample_rate, wav_data)
    return

def count_frames(wav_fns, rank, nshard):
    num_per_shard = math.ceil(len(wav_fns)/nshard)
    start_id, end_id = num_per_shard*rank, num_per_shard*(rank+1)
    wav_fns = wav_fns[start_id: end_id]
    nfs = []
    for wav_fn in tqdm(wav_fns):
        sample_rate, wav_data = wavfile.read(wav_fn)
        nfs.append(len(wav_data))
    return nfs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MUSAN audio preparation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--musan', type=str, help='MUSAN root')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    parser.add_argument('--slurm_partition', type=str, default='cpu', help='slurm partition')
    args = parser.parse_args()
    tmp_dir = tempfile.mkdtemp(dir='./')
    executor = submitit.AutoExecutor(folder=tmp_dir)
    executor.update_parameters(slurm_array_parallelism=100, slurm_partition=args.slurm_partition, timeout_min=240)
    ranks = list(range(0, args.nshard))
    print(f"Split raw audio")
    jobs = executor.map_array(split_musan, [args.musan for _ in ranks], ranks, [args.nshard for _ in ranks])
    [job.result() for job in jobs]
    short_musan = os.path.join(args.musan, 'short-musan')
    print(f"Get speaker info")
    get_speaker_info(short_musan)
    print(f"Mix audio")
    jobs = executor.map_array(make_musan_babble, [short_musan for _ in ranks], ranks, [args.nshard for _ in ranks])
    [job.result() for job in jobs]
    print(f"Count number of frames")
    wav_fns = glob.glob(f"{short_musan}/babble/*/*wav") + glob.glob(f"{short_musan}/music/*/*wav") + glob.glob(f"{short_musan}/noise/*/*wav")
    jobs = executor.map_array(count_frames, [wav_fns for _ in ranks], ranks, [args.nshard for _ in ranks])
    nfs = [job.result() for job in jobs]
    nfs_ = []
    for nf in nfs:
        nfs_.extend(nf)
    nfs = nfs_
    num_frames_fn = f"{short_musan}/nframes.audio"
    with open(num_frames_fn, 'w') as fo:
        for wav_fn, nf in zip(wav_fns, nfs):
            fo.write(os.path.abspath(wav_fn)+'\t'+str(nf)+'\n')
    shutil.rmtree(tmp_dir)
