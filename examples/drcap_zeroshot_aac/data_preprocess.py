import ruamel.yaml as yaml
import torch
from slam_llm.models.CLAP.ase_model import ASE
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torchaudio
from torchaudio.transforms import Resample
import torch.nn.functional as F
import argparse
import os

## we use dataloader to accelerate retrieval 
class caption_dataset(Dataset): 
    def __init__(self, database, remove_dup=True) -> None:
        super().__init__()
        captions = []
        for dataset in database: 
            with open(dataset, 'r') as f: 
                for line in f: 
                    data = json.loads(line)
                    caption = data['target']
                    captions.append(caption)
        if remove_dup == True: 
            seen_caps = set()
            seen_add = seen_caps.add
            self.captions = [x for x in captions if not (x in seen_caps or seen_add(x))]   # remove duplications within dataset
        else: 
            self.captions = captions
    
    def __getitem__(self, index):
        return self.captions[index]
    
    def __len__(self): 
        return len(self.captions)

class audio_dataset(Dataset): 
    def __init__(self, database) -> None:
        super().__init__()
        self.input_type = 'raw'
        self.max_length = 10
        self.sr = 32000
        wav_path = []
        for dataset in database: 
            with open(dataset, 'r') as f: 
                for line in f: 
                    data = json.loads(line)
                    wav_path.append(data['source'])
        self.wav_paths = wav_path
    
    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        wav_info = torchaudio.info(wav_path)
        waveform, sr = torchaudio.load(wav_path, num_frames=self.max_length*wav_info.sample_rate)  #[1, wav_length]
        if waveform.shape[-1] < 0.1*self.sr: 
            waveform = torch.zeros(self.max_length*self.sr)
        else: 
            waveform = waveform[0]
            if self.input_type == "raw": 
                resampler = Resample(orig_freq=sr, new_freq=32000)  # 32k for HTSAT
            elif self.input_type == "mel": 
                resampler = Resample(orig_freq=sr, new_freq=16000)  # 16k for EAT
            waveform = resampler(waveform)  
        return waveform
    
    def __len__(self): 
        return len(self.wav_paths)
    
    def collator(self, samples): 
        audio_list = []
        max_length = max([i.shape[-1] for i in samples])

        for audio in samples:   # audio: raw or mel
            if audio.dim() == 1:   # raw
                if audio.shape[-1] < max_length:
                    pad_length = max_length - audio.shape[-1]
                    audio = F.pad(audio, [0, pad_length], "constant", 0.0)
                audio_list.append(audio)
            elif audio.dim() == 2:   # mel
                audio_list.append(audio)

        audios = torch.stack(audio_list, dim=0)
        return audios

def encode_text(dl): 
    embeds, caps = [], []
    for i, b in enumerate(tqdm(dl, total=len(dl), desc="Encoding captions")): 
        embeds.append(model.encode_text(b).detach())
        caps += b
    return torch.vstack(embeds), caps

def encode_audio(dl): 
    device = torch.device("cuda")
    embeds = []
    for i, b in enumerate(tqdm(dl, total=len(dl), desc="Encoding audios")): 
        b = b.to(device)
        embeds.append(model.encode_audio(b).detach())
    return torch.vstack(embeds)

def retrieve(target, db, topn=None, min_max=None): 
    sim = target @ db.t()  # [n_train, n_db]
    if topn != None and min_max == None: 
        return torch.topk(sim, topn, dim=1)[1]  # [topn, n_train] 

    elif min_max != None: 
        min, max = min_max
        sim = sim.detach().cpu()
        mask = ((sim < max) & (sim > min))
        indices = []
        for j in tqdm(range(mask.shape[0]), total=mask.shape[0]): 
            ind = torch.nonzero(mask[j, :])
            ind = ind[torch.randperm(len(ind))][:topn].squeeze()
            # assert (ind.shape == 0), "No corresponding caption for given similarity interval."
            indices.append(ind)
        # indices = torch.vstack(indices).t()  # [topn, n_train]
        return indices
    
    
if __name__ == '__main__': 
    ## global settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_train", type=str, required=True, 
                        help="input jsonl file of the training split to be modified")
    parser.add_argument("--input_file_val", type=str, required=True, 
                        help="input jsonl file of the validation split to be modified")
    parser.add_argument("--input_file_test", type=str, required=True, 
                        help="input jsonl file of the test split to be modified")
    parser.add_argument("--clap_encoder_path", type=str, required=True, 
                        help="ckpt path of the CLAP encoder")
    parser.add_argument("--input_file_database", type=str, required=True, 
                        help="database file path composed of captions for retrieval")
    parser.add_argument("--topn", type=int, default=3, 
                        help="top n captions selected during retrieval phase")
    parser.add_argument("--sim_min", type=float, default=0.75, 
                        help="min similarity threshold for text-to-text retrieval")
    parser.add_argument("--sim_max", type=float, default=0.85, 
                        help="max similarity threshold for text-to-text retrieval")
    parser.add_argument("--clap_config", type=str, default='examples/drcap_zeroshot_aac/conf/clap_config.yaml', 
                        help="configuration file path of the CLAP encoder")
    parser.add_argument("--output_dir", type=str, default='examples/drcap_zeroshot_aac/data', 
                        help="output dir for modified jsonl file")

    args = parser.parse_args()
    input_file_train, input_file_val, input_file_test = args.input_file_train, args.input_file_val, args.input_file_test
    input_file_database = args.input_file_database
    output_dir = args.output_dir
    topn, sim_min, sim_max = args.topn, args.sim_min, args.sim_max
    clap_encoder_path, clap_config = args.clap_encoder_path, args.clap_config

    # load CLAP model
    with open(clap_config, 'r') as f: 
        config = yaml.safe_load(f)
    model = ASE(config)
    ckpt = torch.load(clap_encoder_path)['model']
    model.load_state_dict(ckpt)
    model = model.eval().cuda()

    # build dataset & dataloader for retrieval
    train_loader = DataLoader(caption_dataset([input_file_train], remove_dup=False), batch_size=1024, shuffle=False)
    val_loader = DataLoader(caption_dataset([input_file_val], remove_dup=False), batch_size=1024, shuffle=False)
    test_dataset = audio_dataset([input_file_test])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collator) 

    train_embeds,train_caps = encode_text(train_loader)
    val_embeds, val_caps = encode_text(val_loader)
    test_embeds = encode_audio(test_loader)

    # encode
    database_loader = DataLoader(caption_dataset([input_file_database]), batch_size=1024, shuffle=False)
    database_embeds, database_caps = encode_text(database_loader)

    # saving text embedding support for projection-based decoding
    text_embeds_path = os.path.dirname(clap_encoder_path) + '/text_embedding_support.pt' 
    torch.save(train_embeds, text_embeds_path)
    print(f"text support embedding saved at: {text_embeds_path}")

    # retrieval
    print("performing retrieval...")
    train_indices = retrieve(train_embeds, database_embeds, topn, (sim_min, sim_max))  # text-to-text retrieval with similarity selection
    val_indices = retrieve(val_embeds, database_embeds, topn, (sim_min, sim_max))  
    test_indices = retrieve(test_embeds, database_embeds, topn)  # audio-to-text retrieval (normal)
    indices = [train_indices, val_indices, test_indices]


    # modify the jsonl file
    for file_num, input_file in enumerate([input_file_train, input_file_val, input_file_test]): 
        input_file_name = input_file.split('/')[-1]
        name, ext = os.path.splitext(input_file_name)
        output_file = output_dir + f'/{name}_rag{ext}'
        with open(input_file, 'r') as fin, open(output_file, 'w') as fout: 
            i=0
            for line in fin: 
                data = json.loads(line)
                similar_captions = []
                if indices[file_num][i].dim() !=  0: 
                    for j in range(len(indices[file_num][i])): 
                        ind = int(indices[file_num][i][j])
                        similar_captions.append(database_caps[ind])
                data['similar_captions'] = similar_captions 
                i+=1
                fout.write(json.dumps(data)+'\n')

        print(f"Finished modifing {input_file}, result jsonl file is: {output_file}")