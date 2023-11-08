import whisper
import torch
import types
import torch.nn.functional as F

# model = whisper.load_model("/home/oss/maziyang.mzy/models/Whisper/base.pt")
encoder = whisper.load_model("/home/oss/maziyang.mzy/models/Whisper/base.pt").encoder

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("/root/whisper/tests/jfk.flac")
# audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to("cuda")
print(mel.shape)

def extract_features(self, x: torch.Tensor):
    """
    x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
        the mel spectrogram of the audio
    """
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
    # x = (x + self.positional_embedding).to(x.dtype)
    x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

    for block in self.blocks:
        x = block(x)

    x = self.ln_post(x)
    return x

encoder.extract_features = types.MethodType(extract_features, encoder)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# get encoder output
mel = mel.unsqueeze(0)
encoder_out = encoder.extract_features(mel)
print(encoder_out.shape)

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)