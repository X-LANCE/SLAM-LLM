import os
import glob
import whisper
from tqdm import tqdm
import torch
import argparse

def main(parent_dir):
    gt_text_path = os.path.join(parent_dir, "gt_text")
    audio_dir = os.path.join(parent_dir, "pred_audio/default_tone")  # 根据目录结构得出
    # audio_dir = os.path.join(parent_dir, "pred_audio/zero_shot_prompt")  # 根据目录结构得出
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
                result = model.transcribe(audio_filepath, language='zh')
                transcription = result['text'].strip()

                # 写入文件名和转录文本
                f.write(f"{id}\t{transcription}\n")
            except Exception as e:
                print(f"Error processing {audio_filepath}: {e}")
                f.write(f"{id}\t\n")

    print("Transcription completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio files.')
    parser.add_argument('--parent_dir', type=str, required=True, help='Path to the wav list file.')
    args = parser.parse_args()

    main(args.parent_dir)


# /nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/tts_emotion_ft_simple_form_libritts_50HZ_wenxi_parameter_epoch_28_step_21_lr5e-5_all_emobox_100k/s2s_epoch_31_step_1800/tts_decode_test_rp_seed_greedy_emodev_ang/gt_text
# /nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/tts_emotion_ft_simple_form_libritts_50HZ_wenxi_parameter_epoch_28_step_21_lr5e-5_all_emobox_100k/s2s_epoch_31_step_1800/tts_decode_test_rp_seed_greedy_emodev_ang/pred_audio/default_tone