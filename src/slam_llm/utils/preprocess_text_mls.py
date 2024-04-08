
import sys
import re
import string

in_f = sys.argv[1]
out_f = sys.argv[2]
language = sys.argv[3]

def remove_punctuation(text):
    # 使用正则表达式删除所有标点符号
    return re.sub(r'[^\w\s]', '', text)


with open(in_f, "r", encoding="utf-8") as f:
  lines = f.readlines()

with open(out_f, "w", encoding="utf-8") as f:
  for line in lines:
    outs = line.strip().split("\t", 1)
    if len(outs) == 2:
      idx, text = outs
      text = remove_punctuation(text).lower()
      if language.lower() == "de":
        text = text.replace("ß","ss")
        text = text.replace("thüre","türe")
    else:
      idx = outs[0]
      text = " "

    # text = [x for x in text]
    # text = " ".join(text)
    out = "{} {}\n".format(idx, text)
    f.write(out)
