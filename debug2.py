from whisper.normalizers import EnglishTextNormalizer

from whisper.normalizers.english import EnglishNumberNormalizer
import jiwer

normalizer = EnglishTextNormalizer()


pred_list=[]
with open("/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/decode_dev_whisper_1_pred_proc",'r') as f1:
    for line in f1:
        line=line.strip().split('\t')
        pred_list.append(line[1])


gt_list=[]
with open("/root/SLAM-LLM/slides_script/3.1/whisper_decode_results/decode_dev_whisper_1_gt_proc",'r') as f:
    for line in f:
        line=line.strip().split('\t')
        gt_list.append(line[1])



wer = jiwer.wer(gt_list, pred_list )
print(f"WER: {wer * 100:.2f} %")


# EnglishTextNormalizer 包含这俩
# self.standardize_numbers = EnglishNumberNormalizer()
# self.standardize_spellings = EnglishSpellingNormalizer()
