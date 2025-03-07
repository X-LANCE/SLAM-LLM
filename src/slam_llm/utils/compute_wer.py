import os
import numpy as np
import sys

def build_diff(ref, hyp, path):
    result = []
    ref = list(map(lambda x: x.lower(), ref))
    hyp = list(map(lambda x: x.lower(), hyp))
    r_record = -1
    h_record = -1
    # path = path+[(len(ref), len(hyp))]

    for rpointer, hpointer in path:
        if rpointer!=r_record+1 or hpointer!=h_record+1:
            r_buffer = ' '.join(ref[r_record+1:rpointer])
            r_buffer = r_buffer if len(r_buffer)>0 else "*"
            h_buffer = ' '.join(hyp[h_record+1:hpointer])
            h_buffer = h_buffer if len(h_buffer)>0 else "*"
            result.append(f"({r_buffer}->{h_buffer})")

        result.append(ref[rpointer])
        r_record = rpointer
        h_record = hpointer

    if r_record<len(ref)-1 or h_record<len(hyp)-1:
        r_buffer = ' '.join(ref[r_record+1:])
        r_buffer = r_buffer if len(r_buffer)>0 else "*"
        h_buffer = ' '.join(hyp[h_record+1:])
        h_buffer = h_buffer if len(h_buffer)>0 else "*"
        result.append(f"({r_buffer}->{h_buffer})")
    return ' '.join(result)






def compute_wer(ref_file,
                hyp_file,
                cer_detail_file):
    rst = {
        'Wrd': 0,
        'Corr': 0,
        'Ins': 0,
        'Del': 0,
        'Sub': 0,
        'Snt': 0,
        'Err': 0.0,
        'S.Err': 0.0,
        'wrong_words': 0,
        'wrong_sentences': 0
    }

    hyp_dict = {}
    ref_dict = {}
    with open(hyp_file, 'r') as hyp_reader:
        for line in hyp_reader:
            key = line.strip().split()[0]
            value = line.strip().split()[1:]
            hyp_dict[key] = value
    with open(ref_file, 'r') as ref_reader:
        for line in ref_reader:
            key = line.strip().split()[0]
            value = line.strip().split()[1:]
            ref_dict[key] = value

    cer_detail_writer = open(cer_detail_file, 'w')
    for hyp_key in hyp_dict:
        if hyp_key in ref_dict:
            out_item = compute_wer_by_line(hyp_dict[hyp_key], ref_dict[hyp_key])
            # if out_item['ins'] > 10 or out_item['del'] > 10:
            #     print(hyp_key + print_cer_detail(out_item))
            #     print("ref:" + '\t' + " ".join(list(map(lambda x: x.lower(), ref_dict[hyp_key]))))
            #     print("hyp:" + '\t' + " ".join(list(map(lambda x: x.lower(), hyp_dict[hyp_key]))))
            rst['Wrd'] += out_item['nwords']
            rst['Corr'] += out_item['cor']
            rst['wrong_words'] += out_item['wrong']
            rst['Ins'] += out_item['ins']
            rst['Del'] += out_item['del']
            rst['Sub'] += out_item['sub']
            rst['Snt'] += 1
            if out_item['wrong'] > 0:
                rst['wrong_sentences'] += 1
            cer_detail_writer.write(hyp_key + print_cer_detail(out_item) + '\n')
            cer_detail_writer.write("ref:" + '\t' + " ".join(list(map(lambda x: x.lower(), ref_dict[hyp_key]))) + '\n')
            cer_detail_writer.write("hyp:" + '\t' + " ".join(list(map(lambda x: x.lower(), hyp_dict[hyp_key]))) + '\n')
            cer_detail_writer.write("diff:" + '\t' + build_diff(ref_dict[hyp_key], hyp_dict[hyp_key], out_item['path']) + '\n')

    if rst['Wrd'] > 0:
        rst['Err'] = round(rst['wrong_words'] * 100 / rst['Wrd'], 2)
    if rst['Snt'] > 0:
        rst['S.Err'] = round(rst['wrong_sentences'] * 100 / rst['Snt'], 2)

    cer_detail_writer.write('\n')
    cer_detail_writer.write("%WER " + str(rst['Err']) + " [ " + str(rst['wrong_words'])+ " / " + str(rst['Wrd']) +
                            ", " + str(rst['Ins']) + " ins, " + str(rst['Del']) + " del, " + str(rst['Sub']) + " sub ]" + '\n')
    cer_detail_writer.write("%SER " + str(rst['S.Err']) + " [ " + str(rst['wrong_sentences']) + " / " + str(rst['Snt']) + " ]" + '\n')
    cer_detail_writer.write("Scored " + str(len(hyp_dict)) + " sentences, " + str(len(hyp_dict) - rst['Snt']) + " not present in hyp." + '\n')

     
def compute_wer_by_line(hyp,
                        ref):
    hyp = list(map(lambda x: x.lower(), hyp))
    ref = list(map(lambda x: x.lower(), ref))

    len_hyp = len(hyp)
    len_ref = len(ref)

    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1

                compare_val = [substitution, insertion, deletion]

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    i = len_hyp
    j = len_ref
    rst = {
        'nwords': len_ref,
        'cor': 0,
        'wrong': 0,
        'ins': 0,
        'del': 0,
        'sub': 0,
        'path': []
    }
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                rst['cor'] += 1

            i -= 1
            j -= 1

        elif ops_matrix[i_idx][j_idx] == 2:  # insert
            i -= 1
            rst['ins'] += 1

        elif ops_matrix[i_idx][j_idx] == 3:  # delete
            j -= 1
            rst['del'] += 1

        elif ops_matrix[i_idx][j_idx] == 1:  # substitute
            i -= 1
            j -= 1
            rst['sub'] += 1

        if i < 0 and j >= 0:
            rst['del'] += 1
        elif j < 0 and i >= 0:
            rst['ins'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    rst['wrong'] = wrong_cnt
    rst['path'] = match_idx

    return rst

def print_cer_detail(rst):
    return ("(" + "nwords=" + str(rst['nwords']) + ",cor=" + str(rst['cor'])
            + ",ins=" + str(rst['ins']) + ",del=" + str(rst['del']) + ",sub="
            + str(rst['sub']) + ") corr:" + '{:.2%}'.format(rst['cor']/rst['nwords'])
            + ",cer:" + '{:.2%}'.format(rst['wrong']/rst['nwords']))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage : python compute-wer.py test.ref test.hyp test.wer")
        sys.exit(0)

    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    cer_detail_file = sys.argv[3]
    compute_wer(ref_file, hyp_file, cer_detail_file)
