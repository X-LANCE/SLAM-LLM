import whisper

def transcribe_audio(audio_path):
    """
    Transcribe audio to text using the whisper-large-v3 model.
    
    Args:
        audio_path: str, path to the audio file.
    
    Returns:
        str, transcribed text.
    """
    model = whisper.load_model("large-v3")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    # options = whisper.DecodingOptions(language="en", without_timestamps=True)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text
