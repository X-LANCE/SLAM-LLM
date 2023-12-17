# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from collections import defaultdict

    
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
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
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
    # charToIx={" ": 1, "'": 22, "1": 30, "0": 29, "3": 37, "2": 32, "5": 34, "4": 38, "7": 36, "6": 35, "9": 31, "8": 33, "A": 5, "C": 17,
    #                      "B": 20, "E": 2, "D": 12, "G": 16, "F": 19, "I": 6, "H": 9, "K": 24, "J": 25, "M": 18, "L": 11, "O": 4, "N": 7, "Q": 27,
    #                      "P": 21, "S": 8, "R": 10, "U": 13, "T": 3, "W": 15, "V": 23, "Y": 14, "X": 26, "Z": 28, "<EOS>": 39}
    charToIx : str = "x"   #应该没用了  TypeError: Object of type NotImplementedType is not JSON serializable 但这个是上面的问题
    modal: str = "AV"
    pretrain_subset: str = "LRS3/pretrain.txt"
    train_subset: str = "LRS3/train.txt"
    valid_subset: str = "LRS3/val.txt"
    test_subset: str = "LRS3/test.txt"
