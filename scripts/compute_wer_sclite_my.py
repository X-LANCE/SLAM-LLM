import os
import sys
import re
from pathlib import Path

# res_file="/root/SLAM-LLM/scripts/test_wer.txt"
ref_file = "/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_gt"
hyp_file = "/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_pred"
# ref_file = "/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_gt.proc"
# hyp_file = "/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_pred.proc"

decode_logfile="/nfs/yangguanrou.ygr/experiments_librispeech/vicuna-7b-v1.5-WavLM-Large-giga1000-ft-phn-hotwords-20240528/asr_epoch_1_step_48000/decode_test_name_beam4_filter_pred_notn.proc.wer"
res_file=hyp_file+".form.wer"

conversational_filler = ['UH', 'UHH', 'UM', 'EH', 'MM', 'HM', 'AH', 'HUH', 'HA', 'ER', 'OOF', 'HEE' , 'ACH', 'EEE', 'EW']
unk_tags = ['<UNK>', '<unk>']
gigaspeech_punctuations = ['<COMMA>', '<PERIOD>', '<QUESTIONMARK>', '<EXCLAMATIONPOINT>']
gigaspeech_garbage_utterance_tags = ['<SIL>', '<NOISE>', '<MUSIC>', '<OTHER>']
non_scoring_words = conversational_filler + unk_tags + gigaspeech_punctuations + gigaspeech_garbage_utterance_tags

def asr_text_post_processing(text):
    # 1. convert to uppercase
    text = text.upper()

    # 2. remove hyphen
    #   "E-COMMERCE" -> "E COMMERCE", "STATE-OF-THE-ART" -> "STATE OF THE ART"
    text = text.replace('-', ' ')

    # 3. remove non-scoring words from evaluation
    remaining_words = []
    for word in text.split():
        if word in non_scoring_words:
            continue
        remaining_words.append(word)

    return ' '.join(remaining_words)

with open(ref_file, 'r') as f_ref:
    ref_proc_file = ref_file + ".form"
    with open(ref_proc_file,'w') as f_ref_form:
        for line in f_ref:
            line = line.strip().split('\t')
            id = line[0]
            text = line [1]
            text = asr_text_post_processing(text)
            f_ref_form.write(f"{text} ({id})\n")


with open(hyp_file, 'r') as f_hyp:
    hyp_proc_file = hyp_file + ".form"
    with open(hyp_proc_file,'w') as f_hyp_form:
        for line in f_hyp:
            line = line.strip().split('\t')
            id = line[0]
            text = line [1]
            text = asr_text_post_processing(text)
            f_hyp_form.write(f"{text} ({id})\n")

        

os.system(f"sclite -r {ref_proc_file} -h {hyp_proc_file} -i swb > {res_file}")

print ("---------------------------------------")
print (f"Result file: {res_file}")
print ("---------------------------------------")
print (f"WER from edit distance:")
# os.system(f"grep %WER: {decode_logfile}")
os.system(f"grep '%WER' {decode_logfile}")

print ("---------------------------------------")
print (f"WER from sclite:")
os.system(f"grep Avg {res_file}")
print ("---------------------------------------")
