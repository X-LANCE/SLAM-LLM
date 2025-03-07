
import sys
import re
import string

in_f = sys.argv[1]
out_f = sys.argv[2]


with open(in_f, "r", encoding="utf-8") as f:
  lines = f.readlines()

with open(out_f, "w", encoding="utf-8") as f:
  for line in lines:
    outs = line.strip().split("\t", 1)
    if len(outs) == 2:
      idx, text = outs
      text = re.sub("<|", "", text)
      text = re.sub("|>", "", text)
      text = re.sub("—", "", text)
      # text = re.sub("<s>", "", text)
      # text = re.sub("@@", "", text)
      # text = re.sub("@", "", text)
      # text = re.sub("<unk>", "", text)
      # text = re.sub(" ", "", text)
      # text = text.lower()
      translator = str.maketrans('', '', string.punctuation.replace("'", ""))
      result = text.translate(translator)
      text = result.upper()
    else:
      idx = outs[0]
      text = " "

    # text = [x for x in text]
    # text = " ".join(text)
    out = "{} {}\n".format(idx, text)
    f.write(out)
