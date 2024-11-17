import argparse
from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio
import os
import sys
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
import numpy as np

# Add python path for Matcha-TTS
sys.path.append('third_party/Matcha-TTS')
os.environ['PYTHONPATH'] = 'third_party/Matcha-TTS'

# Initialize CosyVoice
cosyvoice = CosyVoice('/valleblob/v-wenxichen/models/CosyVoice/CosyVoice-300M-SFT', load_jit=True, load_onnx=False, fp16=True)

def process_data_batch(start_idx, end_idx, save_dir):
    sampling_rate = 22050

    # Loop through the specified range of .arrow files
    for i in range(start_idx, end_idx):
        # Construct file path for the current .arrow file
        file_path = f"/valleblob/v-wenxichen/data/s2s/VoiceAssistant-400K/train/data-{i:05d}-of-00442.arrow"
        dataset = Dataset.from_file(file_path)

        # Collect updated items
        updated_items = []

        # Track progress with process_count

        # Process each data item in the current file
        for j in range(len(dataset)):
            print(f"Processing item {j} in file {i:05d}-of-00442.arrow")
            data_item = dataset[j]
            target_text = data_item['answer']

            # Run CosyVoice inference
            speech_segments = []
            speech_tokens = []
            for result in cosyvoice.inference_sft(target_text, '英文女', stream=False):
                speech_segments.append(result['tts_speech'])
                speech_tokens.append(result['tts_speech_token'])

            combined_speech = torch.cat(speech_segments, dim=-1)
            combined_tokens = torch.cat(speech_tokens, dim=-1)

            # Ensure array is (length,) and convert to numpy
            combined_speech_np = combined_speech.cpu().numpy().flatten()  # Flatten to (length,)
            combined_tokens_np = combined_tokens.cpu().numpy().flatten()  # Flatten tokens to (length,)

            # Create the dictionary format for speech and store token as array directly
            speech_dict = {
                'array': combined_speech_np,
                'sampling_rate': sampling_rate
            }

            # Update the data item with new fields
            data_item['answer_cosyvoice_speech_single_tone'] = speech_dict
            data_item['answer_cosyvoice_speech_token'] = combined_tokens_np  # Store token directly

            # Append modified data item to the updated items list
            updated_items.append(data_item)


        # Incremental save: overwrite only the processed part
        arrays = {k: [item[k] for item in updated_items] for k in updated_items[0].keys()}
        table = pa.Table.from_pydict(arrays)

        # Save the updated dataset directly to the target directory using Parquet
        save_path = os.path.join(save_dir, f"data-{i:05d}-of-00442.parquet")
        pq.write_table(table, save_path)
        print(f"Saved updated dataset to {save_path}")
        print(f"Processed all items in file {i:05d}-of-00442.arrow")

# Specify start and end indices manually for the range you want to process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save TTS data batches")
    parser.add_argument('--start_idx', type=int, required=True, help='Starting index for batch processing')
    parser.add_argument('--end_idx', type=int, required=True, help='Ending index for batch processing (exclusive)')
    parser.add_argument('--save_dir', type=str, default="/valleblob/v-wenxichen/data/s2s/VoiceAssistant-400K-v1/train", help='Directory to save the output files')

    args = parser.parse_args()

    # Ensure the save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Call the function to process the data batch
    process_data_batch(args.start_idx, args.end_idx, args.save_dir)

# python -m debugpy --listen 5678 --wait-for-client tts_data_batch.py
# python tts_data_batch.py