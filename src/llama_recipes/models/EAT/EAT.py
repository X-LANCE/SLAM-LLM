import torch
import torchaudio


def EAT_preprocess(source, norm_mean = -4.268, norm_std = 4.569, target_length = 1024):    
    source = source - source.mean()
    source = source.unsqueeze(dim=0)
    
    source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
    
    n_frames = source.shape[1]
    diff = target_length - n_frames
    if diff > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
        source = m(source)
        
    elif diff < 0:
        source = source[0:target_length, :]
    
    # Normalize the mel spectrogram
    source = (source - norm_mean) / (norm_std * 2)
    source = source.squeeze()
    
    return source