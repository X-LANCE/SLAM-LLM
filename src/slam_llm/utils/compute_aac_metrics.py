from aac_metrics import evaluate
import sys

def compute_wer(ref_file,
                hyp_file):
    pred_captions = []
    gt_captions = []

    with open(hyp_file, 'r') as hyp_reader:
        for line in hyp_reader:
            key = line.strip().split()[0]
            value = line.strip().split()[1:]
            pred_captions.append(value)
    with open(ref_file, 'r') as ref_reader:
        for line in ref_reader:
            key = line.strip().split()[0]
            value = line.strip().split()[1:]
            gt_captions.append(value)

    print('Used lines:', len(pred_captions))
    candidates: list[str] = pred_captions
    mult_references: list[list[str]] = [[gt] for gt in gt_captions]

    corpus_scores, _ = evaluate(candidates, mult_references)
    print(corpus_scores)
    # dict containing the score of each metric: "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider"
    # {"bleu_1": tensor(0.4278), "bleu_2": ..., ...}


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage : python compute_aac_metrics.py test.ref test.hyp")
        sys.exit(0)

    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    cer_detail_file = sys.argv[3]
    compute_wer(ref_file, hyp_file, cer_detail_file)