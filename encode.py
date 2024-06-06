# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# llm_path="/nfs/maziyang.mzy/models/vicuna-7b-v1.5"
# tokenizer = AutoTokenizer.from_pretrained(llm_path)
# tokenizer.pad_token_id = tokenizer.eos_token_id

# with open("/root/fairseq/wavlm_ft_libri960_base10h_token/decode_result/ckptbest/test_other/viterbi/wavlm_ft_libri960_test_other_token.txt","r") as f:
#     with open("/root/fairseq/wavlm_ft_libri960_base10h_token/decode_result/ckptbest/test_other/viterbi/wavlm_ft_libri960_test_other_tokentostr.txt","w") as fw:
#         for line in f:
#             line=[[int(item) for item in line.split()]]
#             str = tokenizer.batch_decode(line, add_special_tokens=False, skip_special_tokens=True)
#             text = str[0]
#             fw.write(text+'\n')
            
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_path="/nfs/maziyang.mzy/models/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(llm_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

with open("/root/fairseq/wavlm_ft_libri960_base10h_token/decode_result/ckptbest/test_clean/viterbi/wavlm_ft_libri960_test_clean_token.txt","r") as f:
    with open("/root/fairseq/wavlm_ft_libri960_base10h_token/decode_result/ckptbest/test_clean/viterbi/wavlm_ft_libri960_test_clean_tokentostr.txt","w") as fw:
        for line in f:
            line=[[int(item) for item in line.split()]]
            str = tokenizer.batch_decode(line, add_special_tokens=False, skip_special_tokens=True)
            text = str[0]
            fw.write(text+'\n')