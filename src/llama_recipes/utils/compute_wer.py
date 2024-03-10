import os
import numpy as np
import sys
import editdistance

def compute_wer(refs, hyps):
    n_err, n_total = 0, 0
    assert len(hyps) == len(refs)
    for hypo, ref in zip(hyps, refs):
        hypo, ref = hypo.strip().split(), ref.strip().split()
        n_err += editdistance.eval(hypo, ref)
        n_total += len(ref)
    return 100 * n_err / n_total
