import os
import sys
import re
from pathlib import Path

### compute WER using sclite using fairseq generate log file
### e.g. python scripts/compute_wer_sclite.py outputs/decoding/hydra_generate.log

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


decode_logfile = sys.argv[1]
dirname = Path(decode_logfile).absolute().parent
ref_file = dirname / 'REF'
hyp_file = dirname / 'HYP'
res_file = dirname / "RESULT"

with open(decode_logfile, 'r') as f_in, \
     open(ref_file, 'w') as f_ref, \
     open(hyp_file, 'w') as f_hyp:
    lines = f_in.read().strip().split('\n')
    for line in lines:
        m = re.search(" - T-(\S+)\s+(.*)", line)
        # m = re.search("T-(\S+)\s+(.*)", line)
        if m is not None:
            utt_id_ref = m.group(1)
            sent_ref   = m.group(2).strip().strip()

            # cols = sent_ref.split()
            # text = asr_text_post_processing(' '.join(cols[0:-1]))
            text = asr_text_post_processing(sent_ref)

            f_ref.write (f"{text} ({utt_id_ref}_1_1)\n")

        else:
            m = re.search(" - D-(\S+)\s+\S+(\s*.*)", line)
            # m = re.search("D-(\S+)\s+\S+(\s*.*)", line)
            if m is not None:
                utt_id_hyp = m.group(1).strip()
                sent_hyp = m.group(2).strip()

                # cols = sent_hyp.split()
                # text = asr_text_post_processing(' '.join(cols[0:-1]))
                text = asr_text_post_processing(sent_hyp)

                f_hyp.write (f"{text} ({utt_id_hyp}_1_1)\n")

os.system(f"sclite -r {ref_file} -h {hyp_file} -i swb > {res_file}")

print ("---------------------------------------")
print (f"Result file: {res_file}")
print ("---------------------------------------")
print (f"WER from edit distance:")
os.system(f"grep WER: {decode_logfile}")
print ("---------------------------------------")
print (f"WER from sclite:")
os.system(f"grep Avg {res_file}")
print ("---------------------------------------")
