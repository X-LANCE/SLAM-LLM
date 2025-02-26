import os
import whisper
from tqdm import tqdm
import torch
import argparse

def main(parent_dir):
    gt_text_path = os.path.join(parent_dir, "gt_text")
    audio_dir = os.path.join(parent_dir, "pred_audio/default_tone")  # 根据目录结构得出
    output_dir = os.path.join(parent_dir, "pred_whisper_text")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 初始化 whisper-large-v3 模型
    model = whisper.load_model("/nfs/maziyang.mzy/models/Whisper/large-v3.pt", device=device)
    
    with open(gt_text_path, 'r') as file, open(output_dir, 'w') as f:
        for line in tqdm(file):
            id = line.split('\t')[0]
            audio_filename = id + '.wav'
            audio_filepath = os.path.join(audio_dir, audio_filename)
            try:
                # 加载音频并进行识别
                result = model.transcribe(audio_filepath, language='en')
                transcription = result['text'].strip()

                # 写入文件名和转录文本
                f.write(f"{id}\t{transcription}\n")
            except Exception as e:
                print(f"Error processing {audio_filepath}: {e}")
                f.write(f"{id}\t\n")

    print("Transcription completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio files.')
    parser.add_argument('--parent_dir', type=str, required=True, help='Path to the parent directory.')
    args = parser.parse_args()

    main(args.parent_dir)