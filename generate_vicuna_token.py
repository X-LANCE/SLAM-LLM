from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_path="/nfs/maziyang.mzy/models/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(llm_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
with open("/root/fairseq/data/bpe_txt/normal_train_960.txt",'r') as f:
    with open("/root/fairseq/data/ali/train_960.token",'w') as fw:
        for line in tqdm(f):
            label = line.strip()
            token = tokenizer.encode(label)

            token_str=" ".join(str(x) for x in token)
            fw.write(token_str+'\n')


# token = tokenizer.encode("I am very happy")
# print(token)
# # [1, 306, 626, 1407, 9796]

# token1 = tokenizer.encode("I am very sad")
# print(token1)
# # [1, 306, 626, 1407, 14610]