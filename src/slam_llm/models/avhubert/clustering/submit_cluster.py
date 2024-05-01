# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, subprocess
import submitit
import argparse
from argparse import Namespace

def dump_av_hubert(*args, **kwargs):
    from dump_hubert_feature import dump_feature
    import fairseq
    import sys
    av_hubert_dir = os.path.join(os.getcwd(), '..')
    fairseq.utils.import_user_module(Namespace(user_dir=av_hubert_dir))
    sys.path.append(av_hubert_dir)
    import utils as custom_utils
    kwargs.update({'custom_utils': custom_utils})
    args = args[0]
    dump_feature(*args, **kwargs)
    return


def dump_mfcc(*args, **kwargs):
    from dump_mfcc_feature import dump_feature
    args = args[0]
    dump_feature(*args, **kwargs)
    return

def run_kmeans(*args, **kwargs):
    import sys
    from learn_kmeans import learn_kmeans
    learn_kmeans(*args, **kwargs)
    return

def apply_kmeans(*args, **kwargs):
    import sys
    from dump_km_label import dump_label
    args = args[0]
    dump_label(*args, **kwargs)
    return

def concatenate(*args, **kwargs):
    from concat import main as concat_fn
    args = args[0]
    concat_fn(*args, **kwargs)
    return

def main():
    parser = argparse.ArgumentParser(description='clustering', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tsv', type=str, help='tsv dir')
    parser.add_argument('--output', type=str, help='output dir (labels)')
    parser.add_argument('--ckpt', type=str, help='checkpoint of last iteration')
    parser.add_argument('--nlayer', type=int, default=12, help='layer index for clustering')
    parser.add_argument('--ncluster', type=int, default=500, help='number of clusters') 
    parser.add_argument('--nshard', type=int, default=100, help='number of shards') 
    parser.add_argument('--percent', type=float, default=0.05, help='Percentage for clustering') 
    parser.add_argument('--mfcc', action='store_true', help='extracting MFCC feature')
    parser.add_argument('--slurm-partition', type=str, help='slurm partitions')
    args = parser.parse_args()
    tsv_dir = args.tsv
    output_dir = args.output
    km_dir = output_dir
    feat_dir = output_dir
    ckpt_path = args.ckpt
    nlayer = args.nlayer
    nshard = args.nshard
    n_clusters = args.ncluster
    slurm_partition = args.slurm_partition
    is_mfcc = args.mfcc
    timeout_min = 240
    percent = 0.1
    log_folder = "log_submit/%j"
    km_path = f"{km_dir}/kmeans.mdl"
    os.makedirs(output_dir, exist_ok=True)
    ext = submitit.AutoExecutor(folder=log_folder)

    args_array = []
    if is_mfcc:
        print(f"Dump MFCC feature")
        for rank in range(nshard):
            args = [tsv_dir, 'train', nshard, rank, output_dir]
            args_array.append(args)
        args_array.append([tsv_dir, 'valid', 1, 0, output_dir])
        ext.update_parameters(timeout_min=60, slurm_partition=slurm_partition, cpus_per_task=1, slurm_array_parallelism=100)
        jobs = ext.map_array(dump_mfcc, args_array)
    else:
        print(f"Dump AV-Hubert feature")
        for rank in range(nshard):
            args = [tsv_dir, 'train', ckpt_path, nlayer, nshard, rank, output_dir, 1600000]
            args_array.append(args)
        args_array.append([tsv_dir, 'valid', ckpt_path, nlayer, 1, 0, output_dir, 1600000])
        ext.update_parameters(timeout_min=60, slurm_partition=slurm_partition, cpus_per_task=1, gpus_per_node=1, slurm_array_parallelism=100)
        jobs = ext.map_array(dump_av_hubert, args_array)
    [job.result() for job in jobs]

    print(f"Learn K-means")
    percent, batch_size = percent, 20000
    ext.update_parameters(timeout_min=timeout_min, slurm_partition=slurm_partition, cpus_per_task=8, mem_gb=128)
    args, kwargs = [feat_dir, 'train', nshard, km_path, n_clusters], vars(Namespace(seed=0, percent=percent, init="k-means++", max_iter=100, batch_size=batch_size, tol=0.0, n_init=20, reassignment_ratio=0.0, max_no_improvement=100))
    print(args, kwargs)
    job = ext.submit(run_kmeans, *args, **kwargs)
    job.result()

    print(f"Apply K-means")
    args_array = []
    for rank in range(nshard):
        args = [feat_dir, 'train', km_path, nshard, rank, output_dir]
        args_array.append(args)
    args_array.append([feat_dir, 'valid', km_path, 1, 0, output_dir])
    ext.update_parameters(timeout_min=10, slurm_partition=slurm_partition, cpus_per_task=1, slurm_array_parallelism=500)
    jobs = ext.map_array(apply_kmeans, args_array)
    [job.result() for job in jobs]

    print(f"Concatenate labels")
    cont = f"for rank in $(seq 0 {nshard-1}); do cat {output_dir}/train_${{rank}}_{nshard}.km; done > {output_dir}/train.km"
    print(cont)
    subprocess.call(cont, shell=True)
    cont = f"cp {output_dir}/valid*.km {output_dir}/valid.km"
    print(cont)
    subprocess.call(cont, shell=True)
    with open(f"{output_dir}/dict.km.txt", 'w') as fo:
        for i in range(n_clusters):
            fo.write(f"{i} {10000}\n")
    print(f"Please delete intermediate files to save space: rm {output_dir}/*npy")
    return


if __name__ == '__main__':
    main()
