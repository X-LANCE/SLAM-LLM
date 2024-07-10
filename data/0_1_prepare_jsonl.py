# %%
import sys
import os
import argparse
import re
import json
import torch
import logging
import pandas as pd
import numpy as np

# from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
from evaluate import load
from tqdm import tqdm
from datetime import datetime

speakers_to_process = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']


for speaker_id in speakers_to_process:
    print(f"Processing speaker {speaker_id}")
    # %%
    # Define the speaker to be used for the test set
    test_speaker = speaker_id 
    keep_all_data = False
    debug = False

    # %%
    torgo_csv_path = './torgo.csv'
    data_df = pd.read_csv(torgo_csv_path)
    dataset_csv = load_dataset('csv', data_files=torgo_csv_path)

    # %%
    # Check if the following columns exist in the dataset ['session', 'audio', 'text', 'speaker_id']
    expected_columns = ['session', 'audio', 'text', 'speaker_id']
    not_found_columns = []
    for column in expected_columns:
        if column not in dataset_csv['train'].column_names:
            not_found_columns.append(column)

    if len(not_found_columns) > 0:
        logging.error(
            "The following columns are not found in the dataset:" + " [" + ", ".join(not_found_columns) + "]")
        sys.exit(1)

    # %%
    print("Splitting the dataset into training / validation / test sets...")

    # Extract the unique speakers in the dataset
    speakers = data_df['speaker_id'].unique()

    print("Unique speakers found in the dataset:")
    print(str(speakers) + '\n')

    if test_speaker not in speakers:
        print("Test Speaker not found in the dataset.")
        sys.exit(1)

    valid_speaker = 'F03' if test_speaker != 'F03' else 'F04'
    train_speaker = [s for s in speakers if s not in [
        test_speaker, valid_speaker]]

    print("Train speakers:", train_speaker)
    print("Validation speaker:", valid_speaker)
    print("Test speaker:", test_speaker)

    torgo_dataset = DatasetDict()
    torgo_dataset['train'] = dataset_csv['train'].filter(
        lambda x: x in train_speaker, input_columns=['speaker_id'])
    torgo_dataset['validation'] = dataset_csv['train'].filter(
        lambda x: x == valid_speaker, input_columns=['speaker_id'])
    torgo_dataset['test'] = dataset_csv['train'].filter(
        lambda x: x == test_speaker, input_columns=['speaker_id'])

    print("Dataset split completed.")

    # %%
    original_data_count = {
        'train': len(torgo_dataset['train']),
        'validation': len(torgo_dataset['validation']),
        'test': len(torgo_dataset['test'])
    }

    if not keep_all_data:
        # Update the three dataset splits (if ['test_data'] == 1, keep in test, if ['test_data'] == 0, keep in train and validation)
        torgo_dataset['train'] = torgo_dataset['train'].filter(
            lambda x: x['test_data'] == 0)
        torgo_dataset['validation'] = torgo_dataset['validation'].filter(
            lambda x: x['test_data'] == 0)
        torgo_dataset['test'] = torgo_dataset['test'].filter(
            lambda x: x['test_data'] == 1)

        # Drop the 'test_data' column
        torgo_dataset['train'] = torgo_dataset['train'].remove_columns([
                                                                       'test_data'])
        torgo_dataset['validation'] = torgo_dataset['validation'].remove_columns([
                                                                                 'test_data'])
        torgo_dataset['test'] = torgo_dataset['test'].remove_columns([
                                                                     'test_data'])

        print("After removal of repeated prompts, the number of data in each dataset is:")
        print(
            f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
        print(
            f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
        print(
            f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)\n')

    # %%
    # Remove special characters from the text
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\`\�0-9]'

    def remove_special_characters(batch):
        batch['text'] = re.sub(chars_to_ignore_regex,
                               ' ', batch['text']).lower()
        return batch

    torgo_dataset = torgo_dataset.map(remove_special_characters)

    # %%

    # Define the output directory
    output_dir = '.'

    # Ensure the output directory exists
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def process_dataset(dataset):
        for split_name, split_data in dataset.items():
            jsonl_filename = f"{speaker_id}_{split_name}.jsonl"
            with open(jsonl_filename, 'w') as jsonl_file:
                for entry in split_data:
                    json_entry = {
                        "key": entry['session'],
                        "source": '/work/van-speech-nlp/data/torgo'+entry['audio'],
                        "target": entry['text']
                    }
                    jsonl_file.write(json.dumps(json_entry) + '\n')
            print(f"{split_name} dataset saved to {jsonl_filename}")

    process_dataset(torgo_dataset)
