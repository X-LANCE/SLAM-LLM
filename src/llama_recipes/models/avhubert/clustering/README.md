# AV-HuBERT Label Preparation

This folder contains scripts for preparing AV-HUBERT labels from tsv files, the
steps are:
1. feature extraction
2. k-means clustering
3. k-means application

## Installation
To prepare labels, you need some additional packages:
```
pip install -r requirements.txt
```

## Data preparation

`*.tsv` files contains a list of audio, where each line is the root, and
following lines are the subpath and number of frames of each video and audio separated by `tab`:
```
<root-dir>
<id-1> <video-path-1> <audio-path-1> <video-number-frames-1> <audio-number-frames-1>
<id-2> <video-path-2> <audio-path-2> <video-number-frames-2> <audio-number-frames-2>
...
```
See [here](../preparation/) for data preparation for LRS3 and VoxCeleb2. 

## Feature extraction

### MFCC feature
Suppose the tsv file is at `${tsv_dir}/${split}.tsv`. To extract 39-D
mfcc+delta+ddelta features for the 1st iteration AV-HuBERT training, run:
```sh
python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${feat_dir}
```
This would shard the tsv file into `${nshard}` and extract features for the
`${rank}`-th shard, where rank is an integer in `[0, nshard-1]`. Features would
be saved at `${feat_dir}/${split}_${rank}_${nshard}.{npy,len}`.


### AV-HuBERT feature
To extract features from the `${layer}`-th transformer layer of a trained
AV-HuBERT model saved at `${ckpt_path}`, run:
```sh
python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir} --user_dir `pwd`/../
```
Features would also be saved at `${feat_dir}/${split}_${rank}_${nshard}.{npy,len}`.

- if out-of-memory, decrease the chunk size with `--max_chunk`


## K-means clustering
To fit a k-means model with `${n_clusters}` clusters on 10% of the `${split}` data, run
```sh
python learn_kmeans.py ${feat_dir} ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1
```
This saves the k-means model to `${km_path}`.

- set `--precent -1` to use all data
- more kmeans options can be found with `-h` flag


## K-means application
To apply a trained k-means model `${km_path}` to obtain labels for `${split}`, run
```sh
python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
```
This would extract labels for the `${rank}`-th shard out of `${nshard}` shards
and dump them to `${lab_dir}/${split}_${rank}_${shard}.km`


Finally, merge shards for `${split}` by running
```sh
for rank in $(seq 0 $((nshard - 1))); do
  cat $lab_dir/${split}_${rank}_${nshard}.km
done > $lab_dir/${split}.km
```
and create a dictionary of cluster indexes by running
```sh
for i in $(seq 1 $((n_cluster-1)));do 
    echo $i 10000
done > $lab_dir/dict.{mfcc,km}.txt
```


## Clustering on slurm
If you are on slurm, you can combine the above steps (feature extraction + K-means clustering + K-means application) by:

- MFCC feature cluster:
```sh
python submit_cluster.py --tsv ${tsv_dir} --output ${lab_dir} --ncluster ${n_cluster} \
  --nshard ${nshard} --mfcc --percent 0.1
```

- AV-HuBERT feature cluster:
```sh
python submit_cluster.py --tsv ${tsv_dir} --output ${lab_dir} --ckpt ${ckpt_path} --nlayer ${layer} \
  --ncluster ${n_cluster} --nshard ${nshard} --percent 0.1
```

This would  dump labels to `${lab_dir}/{train,valid}.km`.
