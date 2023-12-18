
import sys
from num2words import num2words

file = sys.argv[1]
out_file = sys.argv[2]

with open(file) as f:
    lines = f.readlines()

with open(out_file, "w") as fw:
    for line in lines:
        key, content = line.strip().split(maxsplit=1)
        new_content = ""
        for ct in content.split():
            if ct.isdigit():
                ct = num2words(ct)
            new_content += ct + " "
        fw.write(key + " " + new_content + "\n")