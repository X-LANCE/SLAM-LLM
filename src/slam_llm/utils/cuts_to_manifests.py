import json
import sys


part = sys.argv[1]
input_path = f"librispeech_cuts_{part}.jsonl"
output_path = f"librispeech_{part}.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(
    output_path, "w", encoding="utf-8"
) as outfile:
    for line in infile:
        data = json.loads(line.strip())

        key = data["id"]
        source = data["recording"]["sources"][0]["source"].replace(
            "download", "/home/v-yifyang"
        )
        target = data["supervisions"][0]["text"]

        new_data = {"key": key, "source": source, "target": target}

        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")
