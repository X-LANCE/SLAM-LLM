import sys
import os
import re
import string
from whisper_normalizer.english import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()

def reduce_repeated_words(text):
    pattern ="."
    for i in range(1, 50):
        p = pattern * i
        text = re.sub(f'({p})' + r'\1{4,200}', r'\1', text)
    for i in range (50, 100):
        p = pattern * i
        text = re.sub(f'({p})' + r'\1{3,200}', r'\1', text)
    return text

def normalize_text(srcfn, dstfn):
    with open(srcfn, "r") as f_read, open(dstfn, "w") as f_write:
        all_lines = f_read.readlines()
        for line in all_lines:
            line = line.strip()
            line_arr = line.split()
            key = line_arr[0]
            conts = " ".join(line_arr[1:])
            normalized_conts = english_normalizer(conts)
            reduced_conts = reduce_repeated_words(normalized_conts)
            f_write.write("{0}\t{1}\n".format(key, reduced_conts))

if __name__ == "__main__":
    srcfn = sys.argv[1]
    dstfn = sys.argv[2]
    normalize_text(srcfn, dstfn)