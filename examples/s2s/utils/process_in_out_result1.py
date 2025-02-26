import os
import argparse

def load_keys(file_path):
    """Reads keys from a file, one key per line."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

def process_emo_log(emo_log_path, keys, output_path):
    """Processes the emo.log file and calculates the average of the third column for matching keys."""
    sum_values = 0.0
    count = 0
    with open(emo_log_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip() == '' or line.startswith('---') or line.startswith('len') or line.startswith('emo2vec'):
                continue
            parts = line.strip().split()
            # Extract the key from the file path
            audio_path = parts[1]
            key = os.path.basename(audio_path).split('.')[0]
            
            if key in keys:
                value = float(parts[2])
                sum_values += value
                count += 1
                outfile.write(line)  # Write matching line to output

    average_value = sum_values / count if count > 0 else 0.0
    return average_value, count

def filter_file(input_path, keys, output_path):
    """Filters a text file to include only lines with specified keys."""
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line_key = line.split()[0]  # Assuming key is the first word in the line
            if line_key in keys:
                outfile.write(line)

def main(args):
    # Load keys from test_out_id and test_in_id
    test_out_keys = load_keys(args.test_out_id_path)
    test_in_keys = load_keys(args.test_in_id_path)

    # Define paths
    emo_log_path = os.path.join(args.dir, 'emo.log')
    gt_text_path = os.path.join(args.dir, 'gt_text')
    pred_whisper_text_path = os.path.join(args.dir, 'pred_whisper_text')
    pred_text_path = os.path.join(args.dir, 'pred_text')

    # Create output directories if they don't exist
    split_path = os.path.join(args.dir, 'split')
    os.makedirs(os.path.join(split_path, 'out'), exist_ok=True)
    os.makedirs(os.path.join(split_path, 'in'), exist_ok=True)
    os.makedirs(split_path, exist_ok=True)

    # Process emo.log for test_out_id
    out_emo_log_path = os.path.join(split_path, 'out_emo.log')
    average_out, count_out = process_emo_log(emo_log_path, test_out_keys, out_emo_log_path)

    # Process emo.log for test_in_id
    in_emo_log_path = os.path.join(split_path, 'in_emo.log')
    average_in, count_in = process_emo_log(emo_log_path, test_in_keys, in_emo_log_path)

    # Output averages and counts
    with open(out_emo_log_path, 'a', encoding='utf-8') as file:
        file.write(f"Average: {average_out}\nCount: {count_out}\n")

    with open(in_emo_log_path, 'a', encoding='utf-8') as file:
        file.write(f"Average: {average_in}\nCount: {count_in}\n")

    # Filter gt_text, pred_whisper_text, pred_text for test_out_id keys
    out_gt_text_path = os.path.join(split_path, 'out', 'gt_text')
    filter_file(gt_text_path, test_out_keys, out_gt_text_path)
    out_pred_whisper_text_path = os.path.join(split_path, 'out', 'pred_whisper_text')
    filter_file(pred_whisper_text_path, test_out_keys, out_pred_whisper_text_path)
    out_pred_text_path = os.path.join(split_path, 'out', 'pred_text')
    filter_file(pred_text_path, test_out_keys, out_pred_text_path)

    # Filter gt_text, pred_whisper_text, pred_text for test_in_id keys
    in_gt_text_path = os.path.join(split_path, 'in', 'gt_text')
    filter_file(gt_text_path, test_in_keys, in_gt_text_path)
    in_pred_whisper_text_path = os.path.join(split_path, 'in', 'pred_whisper_text')
    filter_file(pred_whisper_text_path, test_in_keys, in_pred_whisper_text_path)
    in_pred_text_path = os.path.join(split_path, 'in', 'pred_text')
    filter_file(pred_text_path, test_in_keys, in_pred_text_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process emo.log and related files based on keys.")
    parser.add_argument('--dir', type=str, default="/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/exp/tts/belle_pretrain_remake/s2s_epoch_2_step_31841/tts_decode_test_rp_seed_greedy_secap_test", help="Directory containing emo.log and related files.")
    parser.add_argument('--test_out_id_path', type=str, required=True, help="Path to the test_out_id file.")
    parser.add_argument('--test_in_id_path', type=str, required=True, help="Path to the test_in_id file.")
    args = parser.parse_args()

    main(args)