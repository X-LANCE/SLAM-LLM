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
import kaldiio

# config
first=1
filter_type="char"

log_filename = "fix_giga/char/baseline/fix_char_{}_only_match_rare_words_match_for_each_word_debug1_5000_1.log".format(first)
prompt_word_num=0


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
            ngram = name[i:i+n]#.lower()
            index.setdefault(ngram, set()).add(name)
    return index

def find_candidate_names(sentence, ngram_index, n=2):
    """通过N-Gram倒排索引找到候选人名"""
    candidates = set()
    for i in range(len(sentence) - n + 1):
        ngram = sentence[i:i+n]#.lower()
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
    sentence_ngrams = generate_ngrams(sentence, name_length)
    
    for ngram in sentence_ngrams:
        if abs(len(ngram) - len(name)) <= length_tolerance:
            sim = similarity(name, ngram) #sim = similarity(name.lower(), ngram.lower())
            max_similarity = max(max_similarity, sim)
    return max_similarity

def score_candidates(candidates, sentence):
    """为候选人名计算得分"""
    scores = {}
    for candidate in candidates:
        score = calculate_similarity_score(candidate, sentence)
        scores[candidate] = score
    return scores

# def score_candidates_for_each_word(candidates, sentence):
#     keys_list = []
#     for word in sentence.split():
#         scores = {}
#         for candidate in candidates:
#             # if word in candidate and candidate not in keys_list:   #不能是简单的in  要是其中一个词
#             #     keys_list.append(candidate)
#             #     continue
#             if word in candidate.split() and candidate not in keys_list:   #不能是简单的in  要是其中一个词
#                 keys_list.append(candidate)
#                 continue
#             score = similarity(word,candidate)
#             scores[candidate] = score
#         sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
#         first_two_items =  sorted_items[:first]
#         keys_list.extend([item[0] for item in first_two_items])
#         # if lenkeys_list
#     return keys_list

def score_candidates_for_each_word(candidates, sentence):
    keys_list = []
    for word in sentence.split():
        scores = {}
        for candidate in candidates:
            if word in candidate.split():   #不能是简单的in  要是其中一个词
                score = calculate_similarity_score(candidate,sentence)
            else:
                score = similarity(word,candidate)
            scores[candidate] = score
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        first_two_items =  sorted_items[:first]
        keys_list.extend([item[0] for item in first_two_items])  #这里还可以去重一下  infer_sentence 不好说
        keys_list=list(set(keys_list))
    if len(keys_list) > len(sentence.split())*first:
        logger.info("what")
    return keys_list


conversational_filler = ['UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE' , 'ACH', 'EEE', 'EW']
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = conversational_filler + unk_tags + gigaspeech_punctuations + gigaspeech_garbage_utterance_tags

def asr_text_post_processing(text):
    # 1. convert to uppercase
    # text = text.upper()  #本来就是大写的

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace('-', ' ')

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return ' '.join(remaining_words)


logger.info("first: %d", first)

        
ctc_file="/nfs/yangguanrou.ygr/data/gigaspeech_my_infer/wavlm_ft_char_giga1000.txt"

data_list = []
label_list = []
key_list = []
line_name_list =[]
name_list=[]
single_name_list=[]

with open("/nfs/yangguanrou.ygr/data/ner/giga_name_test/2/giga_ner_wsplit.txt",'r') as f:
    for line in f:
        line = line.strip().split('\t')

        key_list.append(line[0])
        data_list.append(line[1])
        label_list.append(line[2]) 
        line_name_list.append(line[3]) 

# with open("/nfs/yangguanrou.ygr/data/ner/giga_name_test/person_uniq_my",'r') as f:
with open("/nfs/yangguanrou.ygr/data/ner/giga_name_test/person_uniq_my_tn",'r') as f:
    for line in f:
        line = line.strip()
        name_list.append(line)

        for word in line.split(): 
            single_name_list.append(word) #2122
ngram_index = build_ngram_index(name_list)
single_name_list=list(set(single_name_list)) #1435

infer_list=[]
with open(ctc_file,'r') as finfer:
    for line in finfer:
        infer_list.append(line.strip())



common_words_5k=set()
# with open("/nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/words/common_words_5k.txt") as f:
with open("/nfs/yangguanrou.ygr/data/fbai-speech/is21_deep_bias/words/common_words_10000.txt") as f:  
    for line in f:
        word = line.strip().upper()
        if word not in single_name_list:
            common_words_5k.add(word)
print(len(common_words_5k))  #4768





hotwords_num=0
miss_words_num=0
not_in_infer_num=0

for index in tqdm(range(len(data_list))):
    target = label_list[index] #'KIM WAS NOT DOWN WITH THE CRITIQUE'
    key = key_list[index] #'1012-133424-0005'

    gt=line_name_list[index]  #['B R UW1 Z D', 'K AE1 R AH0 T S', 'F AE1 T AH0 N D', 'L EY1 D AH0 L D', 'M AH1 T AH0 N', 'P EH1 P ER0 D', 'S T UW1', 'T ER1 N AH0 P S']
    if filter_type=="char":
        infer_sentence=infer_list[index]
    else:
        infer_sentence=infer_list[index]  #'HH IY1 HH OW1 P T DH EH1 R W UH1 D B IY1 S T UW1 F AO1 R D IH1 N ER0 T ER1 N AH0 P S AH0 N D K AE1 R AH0 T S AH0 N D B R UW1 Z D P AH0 T EY1 T OW0 Z AH0 N D F AE1 T M AH1 T AH0 N P IY1 S AH0 Z T UW1 B IY1 L EY1 D AH0 L D AW1 T IH0 N TH IH1 K P EH1 P ER0 D F L AW1 ER0 F AE1 T AH0 N D S AO1 S'

     v

    words_list = infer_sentence.split()
    filtered_words = [word for word in words_list if word not in common_words_5k]
    infer_sentence = ' '.join(filtered_words)
    
    
    
    
    candidates = find_candidate_names(infer_sentence, ngram_index) #第一个len11
    # scores = score_candidates(candidates, infer_sentence)
    # sorted_dict = sorted(scores.items(), key=lambda item: item[1],  reverse=True)
    # high_score_items = [(k, value) for k, value in sorted_dict if value > probability_threshold] 
    # if len(high_score_items) < word_num:
    #     high_score_items = sorted_dict[:word_num]
    # prompt_word_num += len(high_score_items)
    # keys_list = [k for k, _ in high_score_items]

    keys_list = score_candidates_for_each_word(candidates, infer_sentence)
    prompt_word_num += len(keys_list)

    # if len(high_score_items)>word_num:
    #     logger.info("longer than %d candidates, cand_num: %d", word_num,len(high_score_items))

    # ======== count recall
    miss=False
    for name in gt.split('|'):
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
        logger.info("gt: %s, keys_list: %s",gt, keys_list)

logger.info("total_hotwords_num: %d, miss_hotwords_num: %d", hotwords_num, miss_words_num)
logger.info("not_in_infer_num: %d", not_in_infer_num)
logger.info("avg_prompt_word_num: %f", float(prompt_word_num)/len(data_list))
logger.info("======================================================================================================================================================")
