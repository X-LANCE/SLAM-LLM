## CLAP-refine 
from ruamel import yaml
import torch
from tqdm import tqdm
import argparse
import json
import torchaudio
from torchaudio.transforms import Resample
from slam_llm.models.CLAP.ase_model import ASE
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

## we use dataloader to accelerate data I/O
class caption_dataset(Dataset): 
    def __init__(self, captions, remove_dup=True) -> None:
        super().__init__()
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
    
def read_captions(decode_log):
    audio_ids, captions = [], []
    with open(decode_log, 'r') as f: 
        for idx, line in enumerate(f): 
            line = line.strip() 
            line_strip = [i for i in line.split('\t') if i]
            if len(line_strip) == 2: 
                audio_id, caption = line_strip
            else: 
                audio_id, caption = line_strip, ''
                print("No caption detected")
                
            audio_ids.append(audio_id)
            captions.append(caption)
            
    return audio_ids, captions

def encode_text(dl): 
    embeds, caps = [], []
    for i, b in enumerate(tqdm(dl, total=len(dl))): 
        embeds.append(model.encode_text(b).detach_())
        caps += b
    return torch.vstack(embeds), caps

def encode_audio(dl): 
    device = torch.device("cuda")
    embeds = []
    for i, b in enumerate(tqdm(dl, total=len(dl))): 
        b = b.to(device)
        embeds.append(model.encode_audio(b).detach_())
    return torch.vstack(embeds), None

    
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_beam", type=int, default=2, 
                        help="Start beam for reranking [included]")
    parser.add_argument("--end_beam", type=int, default=8,
                        help="End beam for reranking [included]")
    parser.add_argument("--clap_ckpt", type=str, required=True,
                        help="model ckpt for CLAP encoder")
    parser.add_argument("--config", type=str, required=True,
                        help="model config for CLAP encoder")
    parser.add_argument("--test_jsonl", type=str, required=True, 
                        help="jsonl file for test set to get audio sources")
    parser.add_argument("--exp_explorer", type=str, required=True,
                        help="dir to load candidate captions")
    parser.add_argument("--rank", type=int, default=1, 
                        help="rank for selecting the candidate caption")
    args = parser.parse_args()

    clap_ckpt = args.clap_ckpt
    config = args.config
    exp_explorer, test_jsonl = args.exp_explorer, args.test_jsonl
    start_beam, end_beam = args.start_beam, args.end_beam
    cand_files = [f'{exp_explorer}/decode_beam{i}_pred' for i in range(start_beam, end_beam+1)]

    print(f"--Clap re-ranking for beam {start_beam}~{end_beam}--")

    ## Load captions & audios
    cand_captions = [read_captions(log)[1] for log in cand_files]
    audio_ids, _ = read_captions(cand_files[0])

    ## Load model
    with open(config, 'r') as f: 
        config = yaml.safe_load(f)
    model = ASE(config)
    cp_dict = torch.load(clap_ckpt)['model']
    model.load_state_dict(cp_dict)
    model.cuda().eval()

    # Encode
    audio_ds = audio_dataset([test_jsonl])
    train_loader = DataLoader(audio_ds, batch_size=1, shuffle=False, collate_fn=audio_ds.collator)  # NOTE: btz should be 1, if not performance will be harmed due to zero padding
    audio_embeds, train_caps = encode_audio(train_loader) # [b, dim]

    caption_embeds= []
    for captions in cand_captions: 
        embeds = encode_text(DataLoader(caption_dataset(captions), batch_size=512, shuffle=False))[0]
        caption_embeds.append(embeds)
    caption_embeds = torch.stack(caption_embeds)  # [b, dim]

    # Select 
    sim = (audio_embeds.unsqueeze(0) * caption_embeds).sum(-1)  # [b, n]
    sorted, indices = torch.sort(sim, dim=0, descending=True)  # [b, n]
    best_captions = []
    for i in range(indices.shape[1]): 
        best_captions.append(cand_captions[int(indices[args.rank-1][i])][i])

    # Write
    output_file = exp_explorer + '/' + f"decode_beam{start_beam}-{end_beam}_pred"
    with open(output_file, 'w') as f: 
        for i, caption in enumerate(best_captions): 
            audio_id = audio_ids[i]
            line = f'{audio_id}\t{caption}'
            f.write(line + '\n')
    print(f"Clap refine finished, decode file saved at {output_file}")