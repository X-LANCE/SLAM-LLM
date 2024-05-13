import os.path as osp
import random
import json, yaml
import copy

import numpy as np
from scipy import signal
import soundfile as sf

import torch
import torchaudio
from torch.utils.data import Dataset
import whisper

import itertools
import os
import sys
import time
from typing import Any, List, Optional, Union
import torch.nn.functional as F
from fairseq.data import data_utils
from python_speech_features import logfbank
from scipy.io import wavfile
import slam_llm.utils.custom_utils as custom_utils


import logging
logger = logging.getLogger(__name__)

def load_audio_visual(manifest_path, max_keep, min_keep, frame_rate, label_paths, label_rates, tol=0.1):
    def is_audio_label_aligned(audio_dur, label_durs):
        return all([abs(audio_dur - label_dur)<tol for label_dur in label_durs])

    n_long, n_short, n_unaligned = 0, 0, 0
    names, inds, sizes = [], [], []
    dur_from_label_list = []
    is_seq_label = any([x==-1 for x in label_rates])
    for label_path, label_rate in zip(label_paths, label_rates):
        label_lengths = [len(line.rstrip().split())/label_rate for line in open(label_path).readlines()]
        dur_from_label_list.append(label_lengths)
    dur_from_label_list = list(zip(*dur_from_label_list))

    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            sz = int(items[-2])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            elif (not is_seq_label) and (not is_audio_label_aligned(sz/frame_rate, dur_from_label_list[ind])):
                n_unaligned += 1
            else:
                video_path = items[1]
                audio_path = items[2]
                audio_id = items[0]
                names.append((video_path, audio_path+':'+audio_id))
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long and {n_unaligned} unaligned, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes

def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )

class AVHubertdataset(torch.utils.data.Dataset):
    def __init__(self,dataset_config,tokenizer=None,split='train',):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        self.prompt = dataset_config.get("prompt", None)
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.data= dataset_config.data
        self.manifest = f"{self.data}/{split}.tsv"

        paths = [
            f"{dataset_config.label_dir}/{split}.{l}" for l in dataset_config.labels
        ]
        label_paths=paths
        image_aug = False
        noise_fn, noise_snr = f"{self.cfg.noise_wav}/{split}.tsv" if dataset_config.noise_wav is not None else None, eval(dataset_config.noise_snr)
        noise_num = dataset_config.noise_num
        store_labels=False


        self.label_rates = (
            [dataset_config.label_rate for _ in range(len(label_paths))]
            if isinstance(dataset_config.label_rate, int)
            else dataset_config.label_rate
        )
        self.modalities = set(dataset_config.modalities)
        self.audio_root, self.names, inds, tot, self.sizes = load_audio_visual(self.manifest, dataset_config.max_sample_size, dataset_config.min_sample_size, frame_rate=dataset_config.sample_rate, label_paths=label_paths, label_rates=self.label_rates)
        self.sample_rate = dataset_config.sample_rate
        self.stack_order_audio = dataset_config.stack_order_audio
        self.shuffle = dataset_config.shuffle
        self.random_crop = dataset_config.random_crop

        self.num_labels = len(label_paths)
        self.single_target = dataset_config.single_target
        self.store_labels = False
        self.is_s2s = dataset_config.is_s2s
        self.noise_wav, self.noise_prob, self.noise_snr, self.noise_num = [ln.strip() for ln in open(noise_fn).readlines()] if noise_fn is not None else [], dataset_config.noise_prob, noise_snr, noise_num

        assert self.single_target == (self.label_rates[0] == -1), f"single target should be equivalent to sequence label (label_rate==-1)"
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        if not dataset_config.skip_verify:
            for label_path, label_rate in zip(label_paths, self.label_rates):
                verify_label_lengths(self.sizes, self.sample_rate, label_path, label_rate, inds, tot)
        else:
            logger.info(f"Skip label alignment verifying")

        self.max_sample_size = (
            dataset_config.max_sample_size if dataset_config.max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = dataset_config.pad_audio
        self.normalize = dataset_config.normalize
        if image_aug:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.RandomCrop((dataset_config.image_crop_size, dataset_config.image_crop_size)),
                custom_utils.HorizontalFlip(0.5),
                custom_utils.Normalize(dataset_config.image_mean, dataset_config.image_std) ])
        else:
            self.transform = custom_utils.Compose([
                custom_utils.Normalize( 0.0,255.0 ),
                custom_utils.CenterCrop((dataset_config.image_crop_size, dataset_config.image_crop_size)),
                custom_utils.Normalize(dataset_config.image_mean, dataset_config.image_std) ])
        logger.info(f"image transform: {self.transform}")

        logger.info(
            f"pad_audio={self.pad_audio}, random_crop={self.random_crop}, "
            f"normalize={self.normalize}, max_sample_size={self.max_sample_size}, "
            f"seqs2seq data={self.is_s2s},")
        logger.info(
            f"Noise wav: {noise_fn}->{len(self.noise_wav)} wav, Prob: {self.noise_prob}, SNR: {self.noise_snr}, Number of mixture: {self.noise_num}"
        )

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def load_feature(self, mix_name):
        """
        Load image and audio feature
        Returns:
        video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
        """
        def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats
        video_fn, audio_fn = mix_name
        if 'video' in self.modalities:
            video_feats = self.load_video(video_fn) # [T, H, W, 1]
        else:
            video_feats = None
        if 'audio' in self.modalities:
            audio_fn = audio_fn.split(':')[0]
            sample_rate, wav_data = wavfile.read(audio_fn)
            assert sample_rate == 16_000 and len(wav_data.shape) == 1
            if np.random.rand() < self.noise_prob:
                wav_data = self.add_noise(wav_data)
            audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
            audio_feats = stacker(audio_feats, self.stack_order_audio) # [T/stack_order_audio, F*stack_order_audio]
        else:
            audio_feats = None
        if audio_feats is not None and video_feats is not None:
            diff = len(audio_feats) - len(video_feats)
            if diff < 0:
                audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
            elif diff > 0:
                audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_video(self, audio_name):
        feats = custom_utils.load_video(os.path.join(self.audio_root, audio_name))
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def select_noise(self):
        rand_indexes = np.random.randint(0, len(self.noise_wav), size=self.noise_num)
        noise_wav = []
        for x in rand_indexes:
            noise_wav.append(wavfile.read(self.noise_wav[x])[1].astype(np.float32))
        if self.noise_num == 1:
            return noise_wav[0]
        else:
            min_len = min([len(x) for x in noise_wav])
            noise_wav = [x[:min_len] for x in noise_wav]
            noise_wav = np.floor(np.stack(noise_wav).mean(axis=0))
            return noise_wav

    def add_noise(self, clean_wav):
        clean_wav = clean_wav.astype(np.float32)
        noise_wav = self.select_noise()
        if type(self.noise_snr) == int or type(self.noise_snr) == float:
            snr = self.noise_snr
        elif type(self.noise_snr) == tuple:
            snr = np.random.randint(self.noise_snr[0], self.noise_snr[1]+1)
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
        adjusted_noise_rms = clean_rms / (10**(snr/20))
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav

        #Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
                reduction_rate = max_int16 / mixed.max(axis=0)
            else :
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * (reduction_rate)
        mixed = mixed.astype(np.int16)
        return mixed

    def __getitem__(self, index):  
        video_feats, audio_feats = self.load_feature(self.names[index]) 
        audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
        if self.normalize and 'audio' in self.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        labels = self.get_labels(index)
        target = labels[0].replace("\n", "")
        fid = self.names[index][1].split(':')[1]

        if self.dataset_config.modal=="AV":
            prompt = "Transcribe video to text. "
        elif self.dataset_config.modal=="AO":
            prompt = "Transcribe speech to text. "
        elif self.dataset_config.modal=="VO":
            prompt = self.dataset_config.prompt #"Transcribe the silent speech in this video to text by lip-reading the speaker's clear and visible lip movements."

        prompt = self.prompt_template.format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)

        audio_length = video_feats.shape[0]

        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        else:
            audio_length = audio_length // 5 # ad-hoc for 5x fc downsample
        audio_pseudo = torch.full((audio_length,), -1) # placeholder

        if self.inference_mode:
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
            example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio,prompt]
            example_mask = example_ids.ge(-1)  # [True,True]

            return {
                "id": index,
                "input_ids": example_ids,
                "attention_mask": example_mask,
                'audio_source': audio_feats,
                "video_source": video_feats,
                'audio_length': audio_length,
                'key': fid,
                'target': target,
            }    

        answer = self.answer_template.format(target)
        example = prompt + answer  # FIX(MZY): avoid putting a bos token before answer.
        example_ids = self.tokenizer.encode(example)  # [prompt,answer]
        example_ids.append(self.tokenizer.eos_token_id)  # [prompt,answer,eos]
        example_ids = torch.tensor(
            example_ids, dtype=torch.int64
        )
        example_ids = torch.cat((audio_pseudo, example_ids))  # [audio,prompt,answer,eos]

        labels_ids = copy.deepcopy(example_ids)  # [audio,prompt,answer,eos]
        labels_ids[:audio_length + prompt_length] = -1  # [-1,-1,answer,eos];
        example_mask = example_ids.ge(-1)  # FIX(GZF): [True,True,True,True]

        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        example_ids[~example_mask] = 0  # [audio,prompt,answer,eos]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,answer,eos]

        return {
            "id": index,
            "input_ids": example_ids,
            "labels": labels_ids,
            "attention_mask": example_mask,
            'audio_source': audio_feats,
            "video_source": video_feats,
            'audio_length': audio_length,
        }
                             
    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size, start=None):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        # longer utterances
        if start is None:
            start, end = 0, target_size
            if self.random_crop:
                start = np.random.randint(0, diff + 1)
                end = size - diff + start
        else:
            end = start + target_size
        return wav[start:end], start

    def collator(self, samples):
        samples = [s for s in samples if s["id"] is not None]
        if len(samples) == 0:
            return {}

        audio_source, video_source = [s["audio_source"] for s in samples], [s["video_source"] for s in samples]
        if audio_source[0] is None:
            audio_source = None
        if video_source[0] is None:
            video_source = None
        if audio_source is not None:
            audio_sizes = [len(s) for s in audio_source]
        else:
            audio_sizes = [len(s) for s in video_source]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        if audio_source is not None:
            collated_audios, padding_mask, audio_starts = self.collater_audio(audio_source, audio_size)
        else:
            collated_audios, audio_starts = None, None
        if video_source is not None:
            collated_videos, padding_mask, audio_starts = self.collater_audio(video_source, audio_size, audio_starts)
        else:
            collated_videos = None

        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([self.pad(s['input_ids'], input_ids_max_length, self.tokenizer.pad_token_id) for s in samples])    
        attention_mask = torch.stack([self.pad(s['attention_mask'], input_ids_max_length, False) for s in samples])

        modality_mask = torch.zeros_like(attention_mask)
        for line, sample in enumerate(samples):
            modality_mask[line, :sample['audio_length']] = 1
        
        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'modality_mask': modality_mask,
                'keys': keys,
                'targets': targets,

                "audio": collated_audios,
                "audio_mask": padding_mask,
                "visual": collated_videos,
                "visual_mask": padding_mask,
            }        

        labels = torch.stack([self.pad(s['labels'], input_ids_max_length, self.IGNORE_INDEX) for s in samples])

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'modality_mask': modality_mask,
    
            "audio": collated_audios,
            "audio_mask": padding_mask,
            "visual": collated_videos,
            "visual_mask": padding_mask,
        }     

    def collater_audio(self, audios, audio_size, audio_starts=None):
        audio_feat_shape = list(audios[0].shape[1:])
        collated_audios = audios[0].new_zeros([len(audios), audio_size]+audio_feat_shape)
        padding_mask = (
            torch.BoolTensor(len(audios), audio_size).fill_(False)
        )
        start_known = audio_starts is not None
        audio_starts = [0 for _ in audios] if not start_known else audio_starts
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat(
                    [audio, audio.new_full([-diff]+audio_feat_shape, 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size, audio_starts[i] if start_known else None
                )
        if len(audios[0].shape) == 2:
            collated_audios = collated_audios.transpose(1, 2) # [B, T, F] -> [B, F, T]
        else:
            collated_audios = collated_audios.permute((0, 4, 1, 2, 3)).contiguous() # [B, T, H, W, C] -> [B, C, T, H, W]
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate # num label per sample
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label_s2s(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        pad, eos = self.label_processors[0].dictionary.pad(), self.label_processors[0].dictionary.eos()
        targets_ = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False)
        prev_output_tokens = data_utils.collate_tokens(targets, pad_idx=pad, eos_idx=eos, left_pad=False, move_eos_to_beginning=True)
        return (targets_, prev_output_tokens), lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                if self.is_s2s:
                    targets, lengths, ntokens = self.collater_seq_label_s2s(targets, pad)
                else:
                    targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def pad(self, sequence, max_length, padding_idx=0):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

def get_audio_dataset(dataset_config, tokenizer, split):
    dataset = AVHubertdataset(dataset_config, tokenizer, split)
    return dataset