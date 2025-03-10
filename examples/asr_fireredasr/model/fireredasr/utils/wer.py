#!/usr/bin/env python3

import argparse
import re
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument("--ref", type=str, required=True)
parser.add_argument("--hyp", type=str, required=True)
parser.add_argument("--print_sentence_wer", type=int, default=0)
parser.add_argument("--do_tn", type=int, default=0, help="simple tn by cn2an")
parser.add_argument("--rm_special", type=int, default=0, help="remove <\|.*?\|>")


def main(args):
    uttid2refs = read_uttid2tokens(args.ref, args.do_tn, args.rm_special)
    uttid2hyps = read_uttid2tokens(args.hyp, args.do_tn, args.rm_special)
    uttid2wer_info, wer_stat, en_dig_stat = compute_uttid2wer_info(
        uttid2refs, uttid2hyps, args.print_sentence_wer)
    wer_stat.print()
    en_dig_stat.print()


def read_uttid2tokens(filename, do_tn=False, rm_special=False):
    print(f">>> Read uttid to tokens: {filename}", flush=True)
    uttid2tokens = OrderedDict()
    uttid2text = read_uttid2text(filename, do_tn, rm_special)
    for uttid, text in uttid2text.items():
        tokens = text2tokens(text)
        uttid2tokens[uttid] = tokens
    return uttid2tokens


def read_uttid2text(filename, do_tn=False, rm_special=False):
    uttid2text = OrderedDict()
    with open(filename, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            cols = line.split()
            if len(cols) == 0:
                print("[WARN] empty line, continue", i, flush=True)
                continue
            assert cols[0] not in uttid2text, f"repeated uttid: {line}"
            if len(cols) == 1:
                uttid2text[cols[0]] = ""
                continue
            txt = " ".join(cols[1:])
            if rm_special:
                txt = " ".join([t for t in re.split("<\|.*?\|>", txt) if t.strip() != ""])
            if do_tn:
                import cn2an
                txt = cn2an.transform(txt, "an2cn")
            uttid2text[cols[0]] = txt
    return uttid2text


def text2tokens(text):
    PUNCTUATIONS = "，。？！,\.?!＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·｡\":" + "()\[\]{}/;`|=+"
    if text == "":
        return []
    tokens = []

    text = re.sub("<unk>", "", text)
    text = re.sub(r"[%s]+" % PUNCTUATIONS, " ", text)

    pattern = re.compile(r'([\u4e00-\u9fff])')
    parts = pattern.split(text.strip().upper())
    parts = [p for p in parts if len(p.strip()) > 0]
    for part in parts:
        if pattern.fullmatch(part) is not None:
            tokens.append(part)
        else:
            for word in part.strip().split():
                tokens.append(word)
    return tokens


def compute_uttid2wer_info(refs, hyps, print_sentence_wer=False):
    print(f">>> Compute uttid to wer info", flush=True)

    uttid2wer_info = OrderedDict()
    wer_stat = WerStats()
    en_dig_stat = EnDigStats()

    for uttid, ref in refs.items():
        if uttid not in hyps:
            print(f"[WARN] No hyp for {uttid}", flush=True)
            continue
        hyp = hyps[uttid]

        if len(hyp) - len(ref) >= 8:
            print(f"[BidLengthDiff]: {uttid} {len(ref)} {len(hyp)}#{' '.join(ref)}#{' '.join(hyp)}")
            #continue

        wer_info = compute_one_wer_info(ref, hyp)
        uttid2wer_info[uttid] = wer_info
        ns = count_english_ditgit(ref, hyp, wer_info)
        wer_stat.add(wer_info)
        en_dig_stat.add(*ns)
        if print_sentence_wer:
            print(f"{uttid} {wer_info}")

    return uttid2wer_info, wer_stat, en_dig_stat


COST_SUB = 3
COST_DEL = 3
COST_INS = 3

ALIGN_CRT = 0
ALIGN_SUB = 1
ALIGN_DEL = 2
ALIGN_INS = 3
ALIGN_END = 4


def compute_one_wer_info(ref, hyp):
    """Impl minimum edit distance and backtrace.
    Args:
        ref, hyp: List[str]
    Returns:
        WerInfo
    """
    ref_len = len(ref)
    hyp_len = len(hyp)

    class _DpPoint:
        def __init__(self, cost, align):
            self.cost = cost
            self.align = align

    dp = []
    for i in range(0, ref_len + 1):
        dp.append([])
        for j in range(0, hyp_len + 1):
            dp[-1].append(_DpPoint(i * j, ALIGN_CRT))

    # Initialize
    for i in range(1, hyp_len + 1):
        dp[0][i].cost = dp[0][i - 1].cost + COST_INS;
        dp[0][i].align = ALIGN_INS
    for i in range(1, ref_len + 1):
        dp[i][0].cost = dp[i - 1][0].cost + COST_DEL
        dp[i][0].align = ALIGN_DEL

    # DP
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            min_cost = 0
            min_align = ALIGN_CRT
            if hyp[j - 1] == ref[i - 1]:
                min_cost = dp[i - 1][j - 1].cost
                min_align = ALIGN_CRT
            else:
                min_cost = dp[i - 1][j - 1].cost + COST_SUB
                min_align = ALIGN_SUB

            del_cost = dp[i - 1][j].cost + COST_DEL
            if del_cost < min_cost:
                min_cost = del_cost
                min_align = ALIGN_DEL

            ins_cost = dp[i][j - 1].cost + COST_INS
            if ins_cost < min_cost:
                min_cost = ins_cost
                min_align = ALIGN_INS

            dp[i][j].cost = min_cost
            dp[i][j].align = min_align

    # Backtrace
    crt = sub = ins = det = 0
    i = ref_len
    j = hyp_len
    align = []
    while i > 0 or j > 0:
        if dp[i][j].align == ALIGN_CRT:
            align.append((i, j, ALIGN_CRT))
            i -= 1
            j -= 1
            crt += 1
        elif dp[i][j].align == ALIGN_SUB:
            align.append((i, j, ALIGN_SUB))
            i -= 1
            j -= 1
            sub += 1
        elif dp[i][j].align == ALIGN_DEL:
            align.append((i, j, ALIGN_DEL))
            i -= 1
            det += 1
        elif dp[i][j].align == ALIGN_INS:
            align.append((i, j, ALIGN_INS))
            j -= 1
            ins += 1

    err = sub + det + ins
    align.reverse()
    wer_info = WerInfo(ref_len, err, crt, sub, det, ins, align)
    return wer_info



class WerInfo:
    def __init__(self, ref, err, crt, sub, dele, ins, ali):
        self.r = ref
        self.e = err
        self.c = crt
        self.s = sub
        self.d = dele
        self.i = ins
        self.ali = ali
        r = max(self.r, 1)
        self.wer = 100.0 * (self.s + self.d + self.i) / r

    def __repr__(self):
        s = f"wer {self.wer:.2f} ref {self.r:2d} sub {self.s:2d} del {self.d:2d} ins {self.i:2d}"
        return s


class WerStats:
    def __init__(self):
        self.infos = []

    def add(self, wer_info):
        self.infos.append(wer_info)

    def print(self):
        r = sum(info.r for info in self.infos)
        if r <= 0:
            print(f"REF len is {r}, check")
            r = 1
        s = sum(info.s for info in self.infos)
        d = sum(info.d for info in self.infos)
        i = sum(info.i for info in self.infos)
        se = 100.0 * s / r
        de = 100.0 * d / r
        ie = 100.0 * i / r
        wer = 100.0 * (s + d + i) / r
        sen = max(len(self.infos), 1)
        errsen = sum(info.e > 0 for info in self.infos)
        ser = 100.0 * errsen / sen
        print("-"*80)
        print(f"ref{r:6d} sub{s:6d} del{d:6d} ins{i:6d}")
        print(f"WER{wer:6.2f} sub{se:6.2f} del{de:6.2f} ins{ie:6.2f}")
        print(f"SER{ser:6.2f} = {errsen} / {sen}")
        print("-"*80)


class EnDigStats:
    def __init__(self):
        self.n_en_word = 0
        self.n_en_correct = 0
        self.n_dig_word = 0
        self.n_dig_correct = 0

    def add(self, n_en_word, n_en_correct, n_dig_word, n_dig_correct):
        self.n_en_word += n_en_word
        self.n_en_correct += n_en_correct
        self.n_dig_word += n_dig_word
        self.n_dig_correct += n_dig_correct

    def print(self):
        print(f"English #word={self.n_en_word}, #correct={self.n_en_correct}\n"
              f"Digit #word={self.n_dig_word}, #correct={self.n_dig_correct}")
        print("-"*80)



def count_english_ditgit(ref, hyp, wer_info):
    patt_en = "[a-zA-Z\.\-\']+"
    patt_dig = "[0-9]+"
    patt_cjk = re.compile(r'([\u4e00-\u9fff])')
    n_en_word = 0
    n_en_correct = 0
    n_dig_word = 0
    n_dig_correct = 0
    ali = wer_info.ali
    for i, token in enumerate(ref):
        if re.match(patt_en, token):
            n_en_word += 1
            for y in ali:
                if y[0] == i+1 and y[2] == ALIGN_CRT:
                    j = y[1] - 1
                    n_en_correct += 1
                    break
        if re.match(patt_dig, token):
            n_dig_word += 1
            for y in ali:
                if y[0] == i+1 and y[2] == ALIGN_CRT:
                    j = y[1] - 1
                    n_dig_correct += 1
                    break
        if not re.match(patt_cjk, token) and not re.match(patt_en, token) \
           and not re.match(patt_dig, token):
            print("[WiredChar]:", token)
    return n_en_word, n_en_correct, n_dig_word, n_dig_correct



if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    main(args)
