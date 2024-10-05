import json
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process JSONL file')
    parser.add_argument('--file', type=str, required=True, help='Path to the JSONL file')
    parser.add_argument('--task', type=str, required=True, help='Path to the JSONL file')
    args = parser.parse_args()

    results_file = args.file
    task = args.task

    response_asr = []
    response_st = []
    gt_asr = []
    gt_st = []
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            gt = data['gt']
            response = data['response']
            source = data['source']

            
            
            # 这里可以对 gt 和 response 进行处理
            text_lan = source.split("_")[-1]
            if len(text_lan)==4:
                text_lan=text_lan[2:]
            if len(text_lan)==6:
                text_lan=text_lan[4:]
            text_lan = "<|"+text_lan+"|>"

            response = response.replace("<|en|>",text_lan)

            response_parts = response.split(text_lan)

            if task == "st":
                gt_parts = gt.split(text_lan)

                if len(response_parts)==2 and len(gt_parts) == 2:
                    # 获取英文和中文部分
                    response_language1 = response_parts[0].strip()
                    response_language2 = response_parts[1].strip()

                    response_asr.append(response_language1)
                    response_st.append(response_language2)
                else:
                    # response_asr.append("")
                    # response_st.append("")
                    print(response)
                    print(gt)
                    continue

                if len(gt_parts) == 2:
                    # 获取英文和中文部分
                    gt_language1 = gt_parts[0].strip()
                    gt_language2 = gt_parts[1].strip()

                    gt_asr.append(gt_language1)
                    gt_st.append(gt_language2)
            if task == "asr":
                gt_asr.append(gt)
                response_asr.append(response)

            


    



    from evaluate import load
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer

    wer = load("wer")
    normalizer = BasicTextNormalizer()


    wer_ortho = 100 * wer.compute(predictions=response_asr, references=gt_asr)

    pred_str_norm = [normalizer(pred) for pred in response_asr]
    label_str_norm = [normalizer(label) for label in gt_asr]

    pred_str_norm = [ pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0]
    label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0]

    wer = 100 * wer.compute(predictions=pred_str_norm, references=label_str_norm)
    wer_result = {"wer_ortho": wer_ortho, "wer": wer}
    print(wer_result)

    print("-"*50)

    text_lan = text_lan[-4:-2]
    print(text_lan)
    if text_lan == "ja":
            text_lan = "ja-mecab"
    elif text_lan == "zh":
        text_lan = "zh"
    else:
        text_lan = "13a"
    print(text_lan)

    import sacrebleu
    
    bleu = sacrebleu.corpus_bleu(response_st,[gt_st], tokenize=text_lan)
    # print(len(response_st))
    print(len(gt_st))
    print(bleu)
    bleu_score = bleu.score
    print(bleu_score)




    # 将 response_st 和 gt_st 中的内容转换为小写
    # response_st_lower = [sentence.lower() for sentence in response_st]
    # gt_st_lower = [sentence.lower() for sentence in gt_st]

    bleu = sacrebleu.corpus_bleu(response_st,[gt_st],lowercase=True, tokenize=text_lan)
    print("Count:", len(response_st))
    print("Length of response_st:", len(response_st))
    print("Length of gt_st:", len(gt_st))
    print("BLEU Score:", bleu)
    bleu_score = bleu.score
    print("BLEU Score:", bleu_score)



if __name__ == '__main__':
    main()