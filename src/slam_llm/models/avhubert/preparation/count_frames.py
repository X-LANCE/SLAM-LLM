# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2, math, os
import tempfile
import shutil
from tqdm import tqdm
from scipy.io import wavfile

def count_frames(fids, audio_dir, video_dir):
    total_num_frames = []
    for fid in tqdm(fids):
        wav_fn = f"{audio_dir}/{fid}.wav"
        video_fn = f"{video_dir}/{fid}.mp4"
        num_frames_audio = len(wavfile.read(wav_fn)[1])
        cap = cv2.VideoCapture(video_fn)
        num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_num_frames.append([num_frames_audio, num_frames_video])
    return total_num_frames

def check(fids, audio_dir, video_dir):
    missing = []
    for fid in tqdm(fids):
        wav_fn = f"{audio_dir}/{fid}.wav"
        video_fn = f"{video_dir}/{fid}.mp4"
        is_file = os.path.isfile(wav_fn) and os.path.isfile(video_fn)
        if not is_file:
            missing.append(fid)
    return missing

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='count number of frames', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root', type=str, help='root dir')
    parser.add_argument('--manifest', type=str, help='a list of filenames')
    parser.add_argument('--nshard', type=int, default=1, help='number of shards')
    parser.add_argument('--rank', type=int, default=0, help='rank id')
    args = parser.parse_args()
    fids = [ln.strip() for ln in open(args.manifest).readlines()]
    print(f"{len(fids)} files")
    audio_dir, video_dir = f"{args.root}/audio", f"{args.root}/video"
    ranks = list(range(0, args.nshard))
    fids_arr = []
    num_per_shard = math.ceil(len(fids)/args.nshard)
    for rank in ranks:
        sub_fids = fids[rank*num_per_shard: (rank+1)*num_per_shard]
        if len(sub_fids) > 0:
            fids_arr.append(sub_fids)
    if args.rank >= len(fids_arr):
        open(f"{args.root}/nframes.audio.{args.rank}", 'w').write('')
        open(f"{args.root}/nframes.video.{args.rank}", 'w').write('')
    else:
        fids = fids_arr[args.rank]
        missing_fids = check(fids, audio_dir, video_dir)
        if len(missing_fids) > 0:
            print(f"Some audio/video files not exist, see {args.root}/missing.list.{args.rank}")
            with open(f"{args.root}/missing.list.{args.rank}", 'w') as fo:
                fo.write('\n'.join(missing_fids)+'\n')
        else:
            num_frames = count_frames(fids, audio_dir, video_dir)
            audio_num_frames = [x[0] for x in num_frames]
            video_num_frames = [x[1] for x in num_frames]
            with open(f"{args.root}/nframes.audio.{args.rank}", 'w') as fo:
                fo.write(''.join([f"{x}\n" for x in audio_num_frames]))
            with open(f"{args.root}/nframes.video.{args.rank}", 'w') as fo:
                fo.write(''.join([f"{x}\n" for x in video_num_frames]))
