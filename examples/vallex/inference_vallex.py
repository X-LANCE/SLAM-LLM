from src.slam_llm.models.vallex.vallex_config import VallexConfig
from src.slam_llm.models.vallex.vallex_model import VALLE
import torch
import string
import os
punctuation_string = string.punctuation
import re
import sentencepiece as spm
from zhon.hanzi import punctuation
import soundfile as sf
from encodec import EncodecModel
import torchaudio
from encodec.utils import convert_audio
import numpy as np
from vocos import Vocos
import argparse


def audiowrite(destpath, audio, sample_rate=16000):
    '''Function to write audio'''
    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    sf.write(destpath, audio, sample_rate)
    return

def decodec(codes, out_path, model):
    with torch.no_grad():
        inp = [(codes.unsqueeze(0), None)]
        wav = model.decode(inp)
    audiowrite(out_path, wav.detach().cpu().numpy().flatten(), sample_rate=model.sample_rate)

def sentence2token(sentence, sp):
    return " ".join(np.array(sp.encode_as_ids(sentence), dtype=str))

def sentence2sp(sentence, sp):
    return " ".join(np.array(sp.encode_as_pieces(sentence), dtype=str))

def norm_eng(txt):
    temp = re.sub('[{}]'.format(punctuation_string),"", txt)
    return re.sub('[{}]'.format(punctuation),"", temp).upper()

def norm_zh(txt):
    temp = re.sub('[{}]'.format(punctuation),"", txt)
    return re.sub('[{}]'.format(punctuation_string),"", temp)

def get_codec(model, audio_path, device):
    with torch.no_grad():
        audio, sr = torchaudio.load(audio_path)
        en_audio = audio.unsqueeze(0)
        en_audio = convert_audio(en_audio, sr, model.sample_rate, model.channels)
        encoded_frames = model.encode(en_audio.to(device))
        # dim, nframe
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze(0).detach().cpu().numpy()
        return codes

from scipy.io.wavfile import write as write_wav
@torch.no_grad()
def generate_audio_24L(model, device, prompt_txt, prompt_audio, target_txt, save_path, sp, prompt_lang, target_lang):
    ar_st_dict, ar_at_dict, nar_st_dict, nar_at_dict = model.ar_st_dict, model.ar_at_dict, model.nar_st_dict, model.nar_at_dict
    codec = EncodecModel.encodec_model_24khz().to(device)
    codec.set_target_bandwidth(6.0)
    vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)
    lang_id_dict = {
        "en": 0,
        "zh": 1,
    }
    ref_txt = prompt_txt
    ref_codec = get_codec(codec, prompt_audio, device)
    ref_at0_tokens = " ".join(ref_codec[0, :].flatten().astype(str))
    ref_all_ats = " ".join(np.expand_dims(ref_codec[:, :].T, axis=0).flatten().astype(str))
    ref_st_tokens = sentence2token(norm_zh(ref_txt), sp)
    
    tar_txt = target_txt
    tgt_st_token = sentence2token(norm_eng(tar_txt), sp)
    
    ref_at0_tokens = ar_at_dict.encode_line(ref_at0_tokens, append_eos=False).long()
    ref_at_tokens = nar_at_dict.encode_line(ref_all_ats, append_eos=False).long().reshape([-1, 8]).unsqueeze(0)
    ref_st_tokens = ar_st_dict.encode_line(ref_st_tokens, append_eos=False).long()
    tgt_st_token = ar_st_dict.encode_line(tgt_st_token, append_eos=False).long()
    src_lang_id, tgt_lang_id = lang_id_dict[prompt_lang], lang_id_dict[target_lang]
    
    if tgt_st_token[-1].data != ar_st_dict.eos():
        tgt_st_token = torch.cat([tgt_st_token, torch.LongTensor([ar_st_dict.eos()])])
    
    if ref_st_tokens[-1].data == ar_st_dict.eos():
        ref_st_tokens = ref_st_tokens[:-1]
    if ref_at0_tokens[-1].data == ar_st_dict.eos():
        ref_at0_tokens = ref_at0_tokens[:-1]
        
    st_sample = torch.cat([ref_st_tokens, tgt_st_token]).unsqueeze(0)
    text_tokens_lens = torch.IntTensor([st_sample.size(-1)])
    enroll_x_lens = torch.IntTensor([len(ref_st_tokens)])
    
    encoded_frames = model.inference_24L(
        st_sample.to(device),
        text_tokens_lens.to(device),
        ref_at_tokens.to(device),
        enroll_x_lens=enroll_x_lens.to(device),
        top_k=-100,
        temperature=1,
        prompt_language=torch.LongTensor([src_lang_id]).to(device),
        text_language=torch.LongTensor([tgt_lang_id]).to(device),
        at_eos=model.ar_at_dict.eos()
    )
    # print(encoded_frames.size())
    at_8 = encoded_frames.reshape(1, -1).long()
    at_8 = nar_at_dict.string(at_8[0])
    codec_item = torch.LongTensor(list(map(int, at_8.strip().split()))).reshape(-1, 8).transpose(0, 1).cuda()
    save_name = save_path
    # 8层token转回语音并保存
    # decodec(codec_item, save_name, codec)
    features = vocos.codes_to_features(codec_item)
    samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device)).squeeze().cpu().numpy()
    write_wav(save_name, model.config.sample_rate, samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference of vallex')
    parser.add_argument('--model_home', type=str)
    parser.add_argument('--target_txt', type=str)
    parser.add_argument('--prompt_txt', type=str)
    parser.add_argument('--prompt_audio', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--prompt_lang', type=str)
    parser.add_argument('--target_lang', type=str)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    vallex_config = VallexConfig(
        ar_at_dict=os.path.join(args.model_home, r"dict.at.txt"),
        ar_st_dict=os.path.join(args.model_home, r"dict.st.txt"),
        nar_at_dict=os.path.join(args.model_home, r"dict.at.txt"),
        nar_st_dict=os.path.join(args.model_home, r"dict.st.txt"),
    )
    model = VALLE(vallex_config)

    vallex_config = VallexConfig.from_pretrained(args.model_home)
    model = VALLE.from_pretrained(args.model_home)
    model.eval().to(device)

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.model_home, r"bpe.model"))
    generate_audio_24L(model, device, args.prompt_txt, args.prompt_audio, 
                       args.target_txt, args.save_path, sp, args.prompt_lang, args.target_lang)