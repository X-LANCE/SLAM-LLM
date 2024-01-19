import sys
import os
import re
import string
# from whisper_normalizer.english import EnglishTextNormalizer

# english_normalizer = EnglishTextNormalizer()

import json
with open('/nfs/maziyang.mzy/data/whisper_normalizer/openai_whisper.json', 'r') as file:
    english_normalizer_dict = json.load(file)

def english_normalizer(sentence):
    words= sentence.split()  #['BUT', 'EVENTUALLY', 'THEY', 'DID', 'COME', 'AROUND']
    new_words=[]

    for word in words:
        word = word.lower()
        if word in english_normalizer_dict:
            new_word=english_normalizer_dict[word]
            print(word,new_word)
        else:
            new_word=word
        new_words.append(new_word)
    
    new_words=' '.join(new_words)
    return new_words


def normalize_text(srcfn, dstfn):
    with open(srcfn, "r") as f_read, open(dstfn, "w") as f_write:
        all_lines = f_read.readlines()
        for line in all_lines:
            line = line.strip()
            line_arr = line.split()
            key = line_arr[0]   #'/nfs/yangguanrou.ygr/LRS3/test/fIICVeGW4RY/00003.txt'
            conts = " ".join(line_arr[1:])  #'BUT EVENTUALLY THEY DID COME AROUND'
            # print(conts)
            normalized_conts = english_normalizer(conts)
            f_write.write("{0}\t{1}\n".format(key, normalized_conts))

if __name__ == "__main__":
    srcfn = sys.argv[1]
    dstfn = sys.argv[2]
    normalize_text(srcfn, dstfn)