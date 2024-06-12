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
from slam_llm.utils.compute_utils import calculate_output_length_1d


# config
probability_threshold = 0.95
word_num=5
filter_type="char"
dis_list=[100]
log_filename = "fix/char/fix_char_{}_{}_{}_only_match_rare_words.log".format(dis_list, word_num, probability_threshold)


import logging
# logger = logging.getLogger(__name__)

import difflib
from functools import lru_cache
from tqdm import tqdm
import Levenshtein
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode='w'
)

logger = logging.getLogger()  
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(filename=log_filename , mode='w')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

logger.handlers[0].setLevel(logging.INFO)
console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger.handlers[0].setFormatter(console_formatter) 

logger.addHandler(file_handler)


def build_ngram_index(names, n=2):
    """构建N-Gram倒排索引"""
    index = {}
    for name in names:
        for i in range(len(name) - n + 1):
            ngram = name[i:i+n].lower()
            index.setdefault(ngram, set()).add(name)
    return index

def find_candidate_names(sentence, ngram_index, n=2):
    """通过N-Gram倒排索引找到候选人名"""
    candidates = set()
    for i in range(len(sentence) - n + 1):
        ngram = sentence[i:i+n].lower()
        candidates.update(ngram_index.get(ngram, []))       
    return candidates

def build_ngram_index_phn(names, n=2):
    """构建N-Gram倒排索引"""
    index = {}
    for name in names:
        phonemes = name.split()
        for i in range(len(phonemes) - n + 1):
            ngram = ' '.join(phonemes[i:i+n])  # 不用小写
            index.setdefault(ngram, set()).add(name)
    return index

def find_candidate_names_phn(phonemes, ngram_index, n=2):
    """通过N-Gram倒排索引找到候选人名"""
    candidates = set()
    phonemes = phonemes.split()
    for i in range(len(phonemes) - n + 1):
        ngram = ' '.join(phonemes[i:i+n])
        candidates.update(ngram_index.get(ngram, []))       
    return candidates

@lru_cache(maxsize=100000)
def similarity(name, sentence):
    return Levenshtein.ratio(name, sentence)  #速度主要来源于这个函数的更换

def generate_ngrams(sentence, n):
    """生成长度为n的n-grams"""
    sentence = sentence.split()
    return [' '.join(sentence[i:i+n]) for i in range(len(sentence)-n+1)]

def calculate_similarity_score(name, sentence, length_tolerance=3):
    max_similarity = 0
    name_sentence = name.split()
    name_length = len(name_sentence)
    sentence_ngrams = generate_ngrams(sentence, name_length) #9
    
    for ngram in sentence_ngrams:
        if abs(len(ngram) - len(name)) <= length_tolerance:
            sim = similarity(name.lower(), ngram.lower())
            max_similarity = max(max_similarity, sim)
    return max_similarity

def score_candidates(candidates, sentence):
    """为候选人名计算得分"""
    scores = {}
    for candidate in candidates:
        score = calculate_similarity_score(candidate, sentence)
        scores[candidate] = score
    return scores

mismatch=0
chongfu=0

common_words_5k=set()
with open("/nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/words/common_words_5k.txt") as f:
    for line in f:
        word = line.strip()
        common_words_5k.add(word)

logger.info("word_num: %d", word_num)
logger.info("probability_threshold: %f", probability_threshold)
for N in dis_list:
    for ref_split in ["test_clean","test_other"]:
        logger.info(str(N)+'\t'+ref_split)
        val_data_path="/nfs/maziyang.mzy/data/librispeech/librispeech_{}.jsonl".format(ref_split)
        infer_file="/nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/my_ref/{}.biasing_{}.tsv".format(ref_split,N)
        ctc_file="/nfs/yangguanrou.ygr/data/librispeech_my_infer/wavlm_ft_libri960_{}_char.txt".format(ref_split)


        data_list = []
        with open(val_data_path, encoding='utf-8') as fin:
            for line in fin:
                data_dict = json.loads(line.strip())
                data_list.append(data_dict)

        hotwords_list=[]
        biaswords_list=[]
        with open(infer_file,'r') as fref:
            for line in fref:
                line=line.strip().split('\t')
                # id = line[0]
                # label = line[1]
                hotwords = line[2]
                biaswords= line[3]
                hotwords_list.append(hotwords)
                biaswords_list.append(biaswords)

        infer_list=[]
        with open(ctc_file,'r') as finfer:
            for line in finfer:
                infer_list.append(line.strip())

        hotwords_num=0
        miss_words_num=0
        not_in_infer_num=0

        for index,data_dict in tqdm(enumerate(data_list)):
            target = data_dict.get("target", None)
            key = data_dict.get("key", None)

            gt=eval(hotwords_list[index])  #['B R UW1 Z D', 'K AE1 R AH0 T S', 'F AE1 T AH0 N D', 'L EY1 D AH0 L D', 'M AH1 T AH0 N', 'P EH1 P ER0 D', 'S T UW1', 'T ER1 N AH0 P S']
            if filter_type=="char":
                infer_sentence=infer_list[index].lower()
            else:
                infer_sentence=infer_list[index]  #'HH IY1 HH OW1 P T DH EH1 R W UH1 D B IY1 S T UW1 F AO1 R D IH1 N ER0 T ER1 N AH0 P S AH0 N D K AE1 R AH0 T S AH0 N D B R UW1 Z D P AH0 T EY1 T OW0 Z AH0 N D F AE1 T M AH1 T AH0 N P IY1 S AH0 Z T UW1 B IY1 L EY1 D AH0 L D AW1 T IH0 N TH IH1 K P EH1 P ER0 D F L AW1 ER0 F AE1 T AH0 N D S AO1 S'
            # infer_sentence=infer_list[index].lower()

            # ================  处理一下 infer_sentence =============
            words_list = infer_sentence.split()
            filtered_words = [word for word in words_list if word not in common_words_5k]
            infer_sentence = ' '.join(filtered_words)
            if len(gt)!=len(filtered_words):
                mismatch+=1
                if set(gt)==set(filtered_words):
                    chongfu+=1
                else:
                    logger.info("gt: %s", str(gt))
                    logger.info("filtered_words: %s", str(filtered_words))
                    logger.info("infer_sentence_list: %s", " ".join(words_list))
            # ================  处理一下 infer_sentence =============

            biaswords=eval(biaswords_list[index]) #['AH0 B AE1 T IH0 S', 'AE1 SH M IY2 D', 'AH0 T R EH1 M IH0 NG L', 'AA2 Z ER0 B AY0 JH AA1 N', 'B IH1 TH AO0 R', 'B R UW1 Z D', 'K EY1 D', 'K AH0 D UW1 T OW0', 'K AE1 R AH0 T S', 'K AE1 R UW0 TH', 'K AO1 SH AH0 N IH0 NG', 'S AH0 L IY1 N', 'CH AE1 G F ER0 D', 'K R AA1 B AH0 L', 'S IH1 N AH0 D AA2 N', 'D EH1 B AH0 T', 'D IH0 L IY1 T', 'D AA1 JH IY0', 'D AA1 L F IH0 N', 'D AH1 S T IH0 NG', 'IH0 L EH1 K T R AH0 M', 'EH1 M AH0 N EY2 T', 'IH0 N G R EY1 V IH0 NG Z', 'IY1 S AO2', 'IH0 G Z AE1 K SH AH0 N Z', 'IH0 K S T ER2 M AH0 N EY1 SH AH0 N', 'F AE1 T AH0 N D', 'F ER1 M ER0', 'F EH1 V R AH0 S', 'F IH1 SH M AE2 N', 'F L AE0 M B OW1 Z', 'F R AE1 T IH0 JH', 'G ER0 AA1 ZH', 'G L EH1 N K EH2 R N', 'G AO1 R SH K AO2 V', 'G R EY1 S T IY2 L', 'G AH1 S IH0 V', 'HH AE1 S K IH0 T', 'HH ER1 K Y AH0 L', 'Y ER0 S IH1 N IY0 AH0 N', 'HH EH1 V AH0 N', 'HH IH1 L T AA2 P S', 'HH AA1 B Z', 'AY0 S L AE1 N D IH0 K', 'IH0 N OW0 CH EH1 N CH IY0 OW0', 'IH0 R AH0 P R EH1 S T AH0 B AH0 L Z', 'IH1 S M AA0 R S', 'JH AO1 R S', 'K ER0 EH1 R IY0', 'K IH1 JH IH0 N', 'L EY1 D AH0 L D', 'L EH1 G Z', 'L EH1 K W EH0 L', 'L AO1 R S IY0', 'L AO0 R EH0 N Z UW1 N IY0', 'L UW1 S AH0 F ER0 Z', 'M EH1 R IY0 AH0 N Z', 'M AE1 T IY0', 'M IY1 N S T', ...]
            if filter_type=="char":
                ngram_index=build_ngram_index(biaswords)
                candidates = find_candidate_names(infer_sentence, ngram_index) #第一个len11
            elif filter_type=="phn":
                ngram_index=build_ngram_index_phn(biaswords)
                candidates = find_candidate_names_phn(infer_sentence, ngram_index) #第一个len11
            scores = score_candidates(candidates, infer_sentence)
            sorted_dict = sorted(scores.items(), key=lambda item: item[1],  reverse=True)
            high_score_items = [(k, value) for k, value in sorted_dict if value > probability_threshold] 
            if len(high_score_items) < word_num:
                high_score_items = sorted_dict[:word_num]
            keys_list = [k for k, _ in high_score_items]

            if len(high_score_items)>word_num:
                logger.info("longer than %d candidates, cand_num: %d", word_num,len(high_score_items))

            # ======== count recall
            miss=False
            for name in gt:

                hotwords_num+=1
                if name not in keys_list:
                    logger.info("miss name: %s", name)
                    miss_words_num+=1
                    miss=True
                    if name not in infer_sentence:
                        not_in_infer_num+=1
                        logger.info("not in infer_sentence: %s", name)
            if miss:
                logger.info("key: %s", key)
                logger.info("infer sentence: %s",infer_sentence)
                logger.info("target sentence: %s", target)
                logger.info("name: %s, gt: %s, keys_list: %s", name, gt, keys_list)

        logger.info("total_hotwords_num: %d, miss_hotwords_num: %d", hotwords_num, miss_words_num)
        logger.info("not_in_infer_num: %d", not_in_infer_num)
        logger.info("mismatch: %d, chongfu: %d", mismatch, chongfu)
        logger.info("======================================================================================================================================================")
