# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class speech_dataset:
    dataset: str = "speech_dataset"
    file: str = "src/llama_recipes/datasets/speech_dataset.py:get_speech_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = None
    max_words: int = None
    train_data_path: str = None
    val_data_path: str = None
    max_words: int = None
    max_mel: int = None
    fix_length_audio: int = -1


@dataclass
class audio_dataset:
    dataset: str = "audio_dataset"
    file: str = "src/llama_recipes/datasets/audio_dataset.py:get_audio_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = None
    fbank_mean: float = 15.41663
    fbank_std: float = 6.55582
    max_words: int = None
    train_data_path: str = None
    val_data_path: str = None
    max_words: int = None
    max_mel: int = None
    fix_length_audio: int = -1


@dataclass
class avsr_dataset:
    dataset: str = "avsr_dataset"
    file: str = "examples/avsr_dataset.py"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/nfs/yangguanrou.ygr/" #"/home/oss/yangguanrou.ygr/"
    h5file: str =  "/nfs/yangguanrou.ygr/LRS3/LRS3.h5"       # "/home/oss/yangguanrou.ygr/LRS3/LRS3.h5"
    noiseFile : str = "/nfs/yangguanrou.ygr/AVSR/LRS3/Noise.h5" #"/home/oss/yangguanrou.ygr/AVSR/LRS3/Noise.h5"
    noiseProb: float = 0.
    noiseSNR: float = 5
    stepSize: int = 16384
    charToIx : str = "x"   #应该没用了  TypeError: Object of type NotImplementedType is not JSON serializable 但这个是上面的问题
    modal: str = "AV"
    pretrain_subset: str = "LRS3/pretrain.txt"
    train_subset: str = "LRS3/train.txt"
    valid_subset: str = "LRS3/val.txt"
    test_subset: str = "LRS3/test.txt"
    reqInpLen: str = 80
