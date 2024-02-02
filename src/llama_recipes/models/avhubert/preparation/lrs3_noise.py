# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def mix_audio(wav_fns):
    wav_data = [wavfile.read(wav_fn)[1] for wav_fn in wav_fns]
    wav_data_ = []
    min_len = min([len(x) for x in wav_data])
    for item in wav_data:
        wav_data_.append(item[:min_len])
    wav_data = np.stack(wav_data_).mean(axis=0).astype(np.int16)
    return wav_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generating babble and speech noise from LRS3', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='lrs3 root dir')
    args = parser.parse_args()
    tsv_fn = os.path.join(args.lrs3, '433h_data', 'train.tsv')

    output_wav = os.path.join(args.lrs3, 'noise', 'babble', 'noise.wav')
    output_tsvs = [os.path.join(args.lrs3, 'noise', 'babble', 'valid.tsv'), os.path.join(args.lrs3, 'noise', 'babble', 'test.tsv')]
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    for output_tsv in output_tsvs:
        os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    print(f"Generating babble noise -> {output_tsvs}")
    num_samples = 30
    sample_rate = 16_000
    min_len = 15*sample_rate
    lns = open(tsv_fn).readlines()[1:]
    wav_fns = [(ln.strip().split('\t')[2], int(ln.strip().split('\t')[-1])) for ln in lns]
    wav_fns = list(filter(lambda x: x[1]>min_len, wav_fns))
    indexes = np.random.permutation(len(wav_fns))[:num_samples]
    wav_fns = [wav_fns[i][0] for i in indexes]
    wav_data = mix_audio(wav_fns)
    wavfile.write(output_wav, sample_rate, wav_data)
    for output_tsv in output_tsvs:
        with open(output_tsv, 'w') as fo:
            fo.write(os.path.abspath(output_wav)+'\n')

    min_len = 20*sample_rate
    speech_tsv_dir, speech_wav_dir = os.path.join(args.lrs3, 'noise', 'speech'), os.path.join(args.lrs3, 'noise', 'speech', 'wav')
    os.makedirs(speech_tsv_dir, exist_ok=True)
    os.makedirs(speech_wav_dir, exist_ok=True)
    print(f'Generating speech noise -> {speech_tsv_dir}')
    lns = open(tsv_fn).readlines()[1:]
    wav_fns = [(ln.strip().split('\t')[2], int(ln.strip().split('\t')[-1])) for ln in lns]
    wav_fns = list(filter(lambda x: x[1]>min_len, wav_fns))
    wav_fns = [x[0] for x in wav_fns]
    print(f"# speech noise audios: {len(wav_fns)}")
    noise_fns = []
    for wav_fn in tqdm(wav_fns):
        sample_rate, wav_data = wavfile.read(wav_fn)
        wav_data = wav_data[:min_len]
        filename = '_'.join(wav_fn.split('/')[-2:])
        noise_fn = f"{speech_wav_dir}/{filename}"
        noise_fns.append(noise_fn)
        wavfile.write(noise_fn, sample_rate, wav_data.astype(np.int16))

    num_train, num_valid, num_test = int(len(noise_fns)*0.6), int(len(noise_fns)*0.2), int(len(noise_fns)*0.2)
    prev = 0
    for split in ['train', 'valid', 'test']:
        split_fns = []
        num_x, tsv_x = eval(f"num_{split}"), f"{speech_tsv_dir}/{split}.tsv"
        for fn in noise_fns[prev: prev+num_x]:
            split_fns.append(os.path.abspath(fn))
        with open(tsv_x, 'w') as fo:
            fo.write('\n'.join(split_fns)+'\n')
        prev += num_x
    return


if __name__ == '__main__':
    main()
