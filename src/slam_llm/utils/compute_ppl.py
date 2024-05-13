from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json

# MODEL_PATH = "/nfs/maziyang.mzy/models/vicuna-7b-v1.5"
MODEL_PATH = "/nfs/zhifu.gzf/ckpt/Llama-2-7b-hf"
# MODEL_PATH = "/nfs/maziyang.mzy/models/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

device = 'cuda:7'
model.to(device)
model.eval()

corpus_path = "/nfs/maziyang.mzy/data/librispeech/librispeech_test_clean_filtered.jsonl"
corpus = []
with open(corpus_path, encoding='utf-8') as fin:
    for line in fin:
        data_dict = json.loads(line.strip())
        corpus.append(data_dict.get("target", None))

cumulative_log_likelihood = 0
total_tokens = 0

for sentence in tqdm(corpus):
    inputs = tokenizer(sentence.strip().lower(), return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    # input_len = input_ids.size(1)
    input_len = len(sentence.split(" "))
    total_tokens += input_len

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        log_likelihood = outputs.loss * input_len
        cumulative_log_likelihood += log_likelihood.item()


average_log_likelihood = cumulative_log_likelihood / total_tokens
corpus_ppl = torch.exp(torch.tensor(average_log_likelihood)).item()

print(f"Model: {MODEL_PATH}")
print(f"Corpus: {corpus_path}")
print(f"Corpus Perplexity: {corpus_ppl}")
