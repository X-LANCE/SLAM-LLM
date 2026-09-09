from pathlib import Path
import numpy as np
import torch
import torchaudio


class SmartSLAMRouter:
    """Lightweight CPU-compatible prototype router for SmartSLAM."""

    TARGET_SAMPLE_RATE = 16000
    FRAME_SIZE = 1600  # 100 ms at 16 kHz

    def __init__(self):
        self.encoder_map = {
            "speech": "whisper",
            "non_speech": "eat",
        }

        # SLAM-LLM configuration profiles used by the routing layer.
        # These describe the existing repository recipes; they do not
        # load model checkpoints by themselves.
        self.encoder_profiles = {
            "whisper": {
                "encoder_name": "whisper",
                "encoder_dim": 1280,
                "encoder_projector": "linear",
                "encoder_projector_ds_rate": 5,
                "input_type": "mel",
                "mel_size": 128,
            },
            "eat": {
                "encoder_name": "eat",
                "encoder_dim": 768,
                "encoder_projector": "linear",
                "encoder_projector_ds_rate": 5,
                "input_type": "mel",
                "fbank_mean": -4.268,
                "fbank_std": 4.569,
                "target_length": 1024,
            },
        }

    def load_audio(self, audio):
        """Load audio and convert it to mono 16 kHz."""
        if isinstance(audio, (str, Path)):
            waveform, sample_rate = torchaudio.load(str(audio))
        elif isinstance(audio, torch.Tensor):
            waveform = audio
            sample_rate = self.TARGET_SAMPLE_RATE
        else:
            raise TypeError("audio must be a file path or torch.Tensor")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.TARGET_SAMPLE_RATE,
            )

        return waveform.squeeze(0).float()

    def extract_features(self, waveform):
        """Extract lightweight acoustic features."""
        x = waveform.detach().cpu().numpy().astype(np.float32)

        if x.size == 0:
            raise ValueError("Audio waveform is empty")

        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))

        signs = np.signbit(x).astype(np.float32)
        zcr = float(np.mean(np.abs(np.diff(signs)))) / 2.0

        spectrum = np.abs(np.fft.rfft(x))
        frequencies = np.fft.rfftfreq(
            len(x),
            d=1.0 / self.TARGET_SAMPLE_RATE,
        )

        spectral_energy = np.sum(spectrum) + 1e-10

        spectral_centroid = float(
            np.sum(frequencies * spectrum) / spectral_energy
        )

        frames = x[:len(x) - (len(x) % self.FRAME_SIZE)]
        if len(frames) >= self.FRAME_SIZE:
            frames = frames.reshape(-1, self.FRAME_SIZE)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            frame_rms_mean = float(np.mean(frame_rms))
            frame_rms_std = float(np.std(frame_rms))
        else:
            frame_rms_mean = rms
            frame_rms_std = 0.0

        return {
            "rms": rms,
            "peak": peak,
            "zero_crossing_rate": zcr,
            "spectral_centroid_hz": spectral_centroid,
            "frame_rms_mean": frame_rms_mean,
            "frame_rms_std": frame_rms_std,
            "duration_seconds": len(x) / self.TARGET_SAMPLE_RATE,
        }

    def classify(self, features):
        """Produce an explainable prototype routing decision."""

        rms = features["rms"]
        zcr = features["zero_crossing_rate"]
        centroid = features["spectral_centroid_hz"]
        frame_rms_std = features["frame_rms_std"]

        speech_score = 0.0
        reasoning = []

        if rms >= 0.10:
            speech_score += 0.30
            reasoning.append("Relatively high overall audio energy supports speech")
        elif rms >= 0.06:
            speech_score += 0.15
            reasoning.append("Moderate overall audio energy provides weak speech evidence")
        else:
            reasoning.append("Relatively low overall audio energy provides weak speech evidence")

        if 0.02 <= zcr <= 0.08:
            speech_score += 0.20
            reasoning.append("Zero-crossing rate is compatible with speech")
        elif zcr <= 0.12:
            speech_score += 0.05
            reasoning.append("Zero-crossing rate provides weak speech evidence")
        else:
            reasoning.append("High zero-crossing rate is less speech-like")

        if 800.0 <= centroid <= 4000.0:
            speech_score += 0.20
            reasoning.append("Spectral centroid overlaps with a speech-like range")
        elif 500.0 <= centroid <= 5000.0:
            speech_score += 0.10
            reasoning.append("Spectral centroid provides weak speech evidence")
        else:
            reasoning.append("Spectral centroid is outside the main speech range")

        if frame_rms_std >= 0.055:
            speech_score += 0.30
            reasoning.append("High temporal energy variation supports speech")
        elif frame_rms_std >= 0.025:
            speech_score += 0.10
            reasoning.append("Moderate temporal energy variation provides weak speech evidence")
        else:
            reasoning.append("Low temporal energy variation is less speech-like")

        speech_score = min(speech_score, 1.0)

        if speech_score >= 0.55:
            modality = "speech"
            confidence = speech_score
            reasoning.append("Combined acoustic evidence favors speech")
        elif speech_score <= 0.35:
            modality = "non_speech"
            confidence = 1.0 - speech_score
            reasoning.append("Combined acoustic evidence favors non-speech")
        else:
            modality = "non_speech"
            confidence = 0.5
            reasoning.append(
                "Evidence is ambiguous; using non-speech fallback"
            )

        return {
            "modality": modality,
            "encoder": self.encoder_map[modality],
            "confidence": round(float(confidence), 4),
            "reasoning": reasoning,
        }

    def route(self, audio):
        """Run preprocessing, feature extraction, and routing."""
        waveform = self.load_audio(audio)
        features = self.extract_features(waveform)
        decision = self.classify(features)
        profile = self.encoder_profiles[decision["encoder"]]

        return {
            **decision,
            "features": features,
            "slam_llm_profile": profile,
        }
