import json
import os
import sys
from tqdm import tqdm
import sys
import os
import soundfile as sf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/Matcha-TTS"))
from cosyvoice.cli.cosyvoice import CosyVoice,CosyVoice2
import torchaudio
import torch
import argparse


def main(input_file_path, output_file_path):
    # 初始化CosyVoice
    my_cosyvoice = CosyVoice("/nfs/yangguanrou.ygr/ckpts/CosyVoice/CosyVoice-300M-Instruct")

    # 读取JSONL文件并处理
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile):
            # 解析JSON对象
            record = json.loads(line.strip())
            
            # 获取source_text
            source_text = record.get("source_text")
            
            # 对source_text进行标准化
            normalized_text = my_cosyvoice.frontend.text_normalize(source_text)
            normalized_text = "".join(normalized_text)
            
            # 更新record中的source_text和target_text
            record["source_text"] = normalized_text
            record["target_text"] = normalized_text
            # 写入新的JSONL文件
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Text normalization completed. Normalized data saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Normalize text in JSONL files using CosyVoice.')
    parser.add_argument('--input', required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output', required=True, help='Path to the output JSONL file.')

    args = parser.parse_args()

    main(args.input, args.output)