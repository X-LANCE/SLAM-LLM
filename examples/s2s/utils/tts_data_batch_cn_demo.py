import os
import json
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from cosyvoice.cli.cosyvoice import CosyVoice
import sys
from tqdm import tqdm  # Import tqdm for progress visualization
from cosyvoice.utils.file_utils import load_wav
import random  # Import random for selecting a random prompt file
import argparse

# Add python path for Matcha-TTS
sys.path.append('third_party/Matcha-TTS')
os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'

# Initialize CosyVoice (Assuming you already have this setup in your environment)
cosyvoice_sft = CosyVoice('/valleblob/v-wenxichen/models/CosyVoice/CosyVoice-300M-SFT', load_jit=True, load_onnx=False, fp16=True)
cosyvoice_base = CosyVoice('/valleblob/v-wenxichen/models/CosyVoice/CosyVoice-300M')
sampling_rate = 22050

prompt_speech_dir = "/valleblob/v-wenxichen/data/prompt/seedtts_testset/zh/prompt-wavs"
print("The number of available prompt files is:", len(os.listdir(prompt_speech_dir)))
prompt_files = os.listdir(prompt_speech_dir)
# cache the prompt speech
prompt_speech_16k_list = {}
for prompt_file in prompt_files:
    prompt_wav_path = os.path.join(prompt_speech_dir, prompt_file)
    prompt_speech_16k = load_wav(prompt_wav_path, 16000)
    prompt_speech_16k_list[prompt_file] = prompt_speech_16k


def synthesize_and_store_parquet(jsonl_path, save_dir):
    """
    Process a JSONL file to synthesize question_audio and answer_cosyvoice_speech_token fields, and save to Parquet.
    
    Parameters:
    - jsonl_path (str): Path to the JSONL input file with conversation data.
    - save_dir (str): Directory to save the processed Parquet file.
    """
    # Load JSONL data
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    processed_data = []
    count = 0

    # Process each entry in JSONL data with tqdm progress bar
    for entry in tqdm(data, desc=f"Processing {os.path.basename(jsonl_path)}"):
        index = entry.get("id", "")
        conversations = entry.get("conversations", [])

        history = ''
        for i in range(0, len(conversations), 2):
            if conversations[i]["from"] != "human" or conversations[i+1]["from"]!= "assistant":
                print(f"Invalid conversation format at index {index}, round {i // 2 + 1}")
                continue

            question_text = conversations[i]["value"]
            answer_text = conversations[i + 1]["value"]

            # Select a random prompt wav file
            prompt_speech_16k_key = random.choice(prompt_files)
            prompt_speech_16k = prompt_speech_16k_list[prompt_speech_16k_key]

            # Generate question_audio using CosyVoice-300M
            question_speech_segments = []
            for result in cosyvoice_base.inference_cross_lingual(question_text, prompt_speech_16k, stream=False):
                question_speech_segments.append(result['tts_speech'])
            
            combined_question_speech = torch.cat(question_speech_segments, dim=-1)
            combined_question_speech_np = combined_question_speech.cpu().numpy().flatten()

            # if the combined_question_speech is longer than 30 seconds, skip this entry
            if len(combined_question_speech_np) > 30 * sampling_rate:
                print(f"Skipping entry due to long question: {len(combined_question_speech_np) / sampling_rate:.2f} seconds")
                history = history + ' USER: ' + question_text + ' ASSISTANT: ' + answer_text
                continue

            question_audio = {
                'array': combined_question_speech_np,
                'path': 'placeholder_path',  # Placeholder for path as requested
                'sampling_rate': sampling_rate
            }

            # Generate answer_cosyvoice_speech_token using CosyVoice-300M-SFT
            answer_speech_tokens = []
            for result in cosyvoice_sft.inference_sft(answer_text, '中文女', stream=False):
                answer_speech_tokens.append(result['tts_speech_token'])
            
            combined_answer_tokens = torch.cat(answer_speech_tokens, dim=-1).cpu().numpy().flatten()

            # if the combined_answer_tokens is longer than 3000 tokens, skip this entry
            if len(combined_answer_tokens) > 3000:
                print(f"Skipping entry due to long answer: {len(combined_answer_tokens)} tokens")
                continue

            # Update entry with synthesized data
            output_entry = {
                'split_name': 'train_3.5M_CN_ready4cosy_wo_code_switching',
                'index': index,
                'round': i // 2 + 1,
                'question': history + ' <USER> ' + question_text,
                'question_audio': question_audio,
                'answer': answer_text,
                'answer_cosyvoice_speech_token': combined_answer_tokens,
                'answer_snac': "",
            }
            processed_data.append(output_entry)

            # Update history for the next round
            history = history + ' USER: ' + question_text + ' ASSISTANT: ' + answer_text

        count += 1
        if count == 5:
            break

    # Convert processed data to a DataFrame for saving as Parquet
    df = pd.DataFrame(processed_data)

    # Set up output filename
    base_name = os.path.basename(jsonl_path).replace(".json", "")
    output_path = os.path.join(save_dir, f"{base_name}.parquet")
    
    # Save the DataFrame to Parquet
    df.to_parquet(output_path, index=False)
    print(f"Processed data saved to {output_path}")


def batch_process_jsonl_files(jsonl_dir, save_dir, start_index, end_index):
    """
    Process a range of JSONL files in the specified directory.
    
    Parameters:
    - jsonl_dir (str): Directory containing JSONL files.
    - save_dir (str): Directory to save the processed Parquet files.
    - start_index (int): Start index of the JSONL files to process.
    - end_index (int): End index of the JSONL files to process.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(start_index, end_index):
        jsonl_filename = f"part_{i}.json"
        jsonl_path = os.path.join(jsonl_dir, jsonl_filename)

        if os.path.exists(jsonl_path):
            print(f"Processing file: {jsonl_path}")
            synthesize_and_store_parquet(jsonl_path, save_dir)
        else:
            print(f"File not found: {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process JSONL files for TTS data.")
    parser.add_argument('--start_index', type=int, required=True, help='Start index of the JSONL files to process.')
    parser.add_argument('--end_index', type=int, required=True, help='End index of the JSONL files to process.')
    parser.add_argument('--jsonl_dir', type=str, required=True, help='Directory containing JSONL files.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the processed Parquet files.')
    
    args = parser.parse_args()
    
    batch_process_jsonl_files(args.jsonl_dir, args.save_dir, args.start_index, args.end_index)