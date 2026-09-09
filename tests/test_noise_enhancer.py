"""Unit tests for the Noise Enhancement module."""

import os
import tempfile
import numpy as np
import pytest
import soundfile as sf

from slam_llm.enhancement import NoiseEnhancer, enhance_audio


@pytest.fixture
def synthetic_audio_signals():
    """Generate sample clean, low, and strong noise audio vectors."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    # Speech-like tone harmonic signal
    clean = (0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # Add envelope
    envelope = (0.5 * (1.0 + np.sin(2 * np.pi * 3 * t))) ** 2
    clean = clean * envelope

    # Generate noisy signal
    np.random.seed(42)
    strong_noise = (clean + np.random.normal(0, 0.25, size=len(clean))).astype(np.float32)
    
    return {"sr": sr, "clean": clean, "noisy": strong_noise}


def test_clean_audio_handling(synthetic_audio_signals):
    """Test that clean audio is recognized as clean and enhancement is not forced."""
    enhancer = NoiseEnhancer(snr_clean_threshold_db=20.0)
    clean = synthetic_audio_signals["clean"]
    sr = synthetic_audio_signals["sr"]

    enhanced_wav, result = enhancer.enhance(clean, sample_rate=sr)

    assert result.noise_detected is False
    assert result.enhancement_applied is False
    assert result.noise_level in ["Clean", "Low"]
    assert np.allclose(enhanced_wav, clean, atol=1e-5)


def test_noisy_audio_detection_and_enhancement(synthetic_audio_signals):
    """Test that noisy audio triggers noise detection and enhancement."""
    enhancer = NoiseEnhancer(snr_clean_threshold_db=20.0)
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    enhanced_wav, result = enhancer.enhance(noisy, sample_rate=sr)

    assert result.noise_detected is True
    assert result.enhancement_applied is True
    assert result.estimated_snr_after > result.estimated_snr_before
    assert len(enhanced_wav) == len(noisy)


def test_output_wav_file_validity(synthetic_audio_signals):
    """Test that saving output WAV results in valid, readable audio on disk."""
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "input.wav")
        output_wav = os.path.join(tmp_dir, "output.wav")
        
        sf.write(input_wav, noisy, sr)
        assert os.path.exists(input_wav)

        _, result = enhance_audio(input_wav, output_path=output_wav)

        assert os.path.exists(output_wav)
        assert os.path.getsize(output_wav) > 0
        
        read_audio, read_sr = sf.read(output_wav)
        assert read_sr == 16000
        assert len(read_audio) > 0
        assert np.max(np.abs(read_audio)) <= 1.0


def test_metadata_structure(synthetic_audio_signals):
    """Test that result object conforms to required metadata specification."""
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    _, result = enhance_audio(noisy, sample_rate=sr)
    res_dict = result.to_dict()

    expected_keys = [
        "input_path",
        "output_path",
        "noise_detected",
        "noise_level",
        "estimated_snr_before",
        "estimated_snr_after",
        "enhancement_applied",
        "method",
        "sample_rate",
        "processing_time",
        "duration_sec",
    ]
    for key in expected_keys:
        assert key in res_dict
    assert isinstance(result.processing_time, float)
    assert result.processing_time >= 0.0


def test_sample_rate_resampling(synthetic_audio_signals):
    """Test that audio recorded at different sample rate (e.g., 44.1 kHz) resamples correctly."""
    clean = synthetic_audio_signals["clean"]
    sr_orig = 44100
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_wav = os.path.join(tmp_dir, "sr44100.wav")
        sf.write(input_wav, clean, sr_orig)

        enhancer = NoiseEnhancer()
        loaded_wav, loaded_sr = enhancer.load_audio(input_wav, target_sr=16000)

        assert loaded_sr == 16000
        assert len(loaded_wav) > 0


def test_enhancement_disabled_bypass(synthetic_audio_signals):
    """Test that enabled=False acts as a pure identity passthrough without alteration."""
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    enhancer = NoiseEnhancer(enabled=False)
    enhanced_wav, result = enhancer.enhance(noisy, sample_rate=sr)

    assert result.enhancement_applied is False
    assert result.method == "Bypass (Disabled)"
    assert np.allclose(enhanced_wav, noisy, atol=1e-5)

    # Also test via enhance_audio wrapper
    wav2, res2 = enhance_audio(noisy, sample_rate=sr, enabled=False)
    assert res2.enhancement_applied is False
    assert np.allclose(wav2, noisy, atol=1e-5)


def test_stereo_and_multichannel_handling():
    """Test stereo (2, N) and (N, 2) multi-channel inputs and shape preservation."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig1 = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sig2 = (0.5 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)
    
    np.random.seed(42)
    noisy_ch1 = sig1 + np.random.normal(0, 0.2, size=len(sig1)).astype(np.float32)
    noisy_ch2 = sig2 + np.random.normal(0, 0.2, size=len(sig2)).astype(np.float32)

    # Test (2, N)
    stereo_2xN = np.stack([noisy_ch1, noisy_ch2], axis=0)
    enhancer = NoiseEnhancer()
    out_2xN, res_2xN = enhancer.enhance(stereo_2xN, sample_rate=sr)

    assert out_2xN.shape == stereo_2xN.shape
    assert res_2xN.enhancement_applied is True
    assert np.all(np.isfinite(out_2xN))
    assert np.max(np.abs(out_2xN)) <= 1.0

    # Test (N, 2)
    stereo_Nx2 = np.stack([noisy_ch1, noisy_ch2], axis=1)
    out_Nx2, res_Nx2 = enhancer.enhance(stereo_Nx2, sample_rate=sr)
    assert out_Nx2.shape == stereo_Nx2.shape
    assert res_Nx2.enhancement_applied is True


def test_very_short_and_empty_audio():
    """Test handling of empty, sub-frame, and shorter-than-FFT audio signals without crash."""
    enhancer = NoiseEnhancer(n_fft=1024)
    sr = 16000

    # Empty array
    empty = np.array([], dtype=np.float32)
    out_empty, res_empty = enhancer.enhance(empty, sample_rate=sr)
    assert len(out_empty) == 0
    assert res_empty.enhancement_applied is False

    # Short audio (e.g. 50 samples, much shorter than n_fft 1024)
    short = np.array([0.1, -0.2, 0.3, -0.1] * 10, dtype=np.float32)
    out_short, res_short = enhancer.enhance(short, sample_rate=sr)
    assert len(out_short) == len(short)
    assert np.all(np.isfinite(out_short))
    assert np.max(np.abs(out_short)) <= 1.0


def test_silent_and_dc_audio():
    """Test silent audio (all zeros) and pure DC offset audio."""
    enhancer = NoiseEnhancer()
    sr = 16000

    zeros = np.zeros(16000, dtype=np.float32)
    out_zeros, res_zeros = enhancer.enhance(zeros, sample_rate=sr)
    assert res_zeros.noise_detected is False
    assert res_zeros.enhancement_applied is False
    assert np.allclose(out_zeros, zeros)

    # Constant DC
    dc = np.ones(16000, dtype=np.float32) * 0.2
    out_dc, res_dc = enhancer.enhance(dc, sample_rate=sr)
    assert np.all(np.isfinite(out_dc))
    assert np.max(np.abs(out_dc)) <= 1.0


def test_nan_and_inf_handling():
    """Test that inputs containing NaN or Inf are sanitized safely."""
    enhancer = NoiseEnhancer()
    sr = 16000
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    corrupt = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    corrupt[100] = np.nan
    corrupt[200] = np.inf
    corrupt[300] = -np.inf

    out, res = enhancer.enhance(corrupt, sample_rate=sr)
    assert np.all(np.isfinite(out))
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.max(np.abs(out)) <= 1.0


def test_clipping_prevention_and_bounds():
    """Test that outputs remain strictly within [-1.0, 1.0] even for extreme amplitudes."""
    enhancer = NoiseEnhancer()
    sr = 16000
    # Over-amplitude signal exceeding 1.0
    over_amp = np.random.normal(0, 2.5, size=16000).astype(np.float32)

    out, res = enhancer.enhance(over_amp, sample_rate=sr, force_process=True)
    assert np.max(out) <= 1.0
    assert np.min(out) >= -1.0
    assert np.all(np.isfinite(out))


def test_multiple_sample_rates():
    """Test enhancement stability across different standard audio sample rates (8k, 16k, 44.1k, 48k)."""
    rates = [8000, 16000, 44100, 48000]
    enhancer = NoiseEnhancer()

    for sr in rates:
        t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
        sig = (0.5 * np.sin(2 * np.pi * 400 * t) + np.random.normal(0, 0.1, size=len(t))).astype(np.float32)
        out, res = enhancer.enhance(sig, sample_rate=sr)

        assert res.sample_rate == sr
        assert len(out) == len(sig)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.0


def test_determinism(synthetic_audio_signals):
    """Test that enhancement produces identical, deterministic output across consecutive runs."""
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    enhancer = NoiseEnhancer()
    out1, res1 = enhancer.enhance(noisy, sample_rate=sr)
    out2, res2 = enhancer.enhance(noisy, sample_rate=sr)

    assert np.array_equal(out1, out2)
    assert res1.estimated_snr_before == res2.estimated_snr_before
    assert res1.estimated_snr_after == res2.estimated_snr_after
    assert res1.enhancement_applied == res2.enhancement_applied


def test_speech_formant_protection(synthetic_audio_signals):
    """Test that protect_speech_bands maintains higher fidelity on speech frequencies."""
    noisy = synthetic_audio_signals["noisy"]
    clean = synthetic_audio_signals["clean"]
    sr = synthetic_audio_signals["sr"]

    enh_protected = NoiseEnhancer(protect_speech_bands=True)
    enh_unprotected = NoiseEnhancer(protect_speech_bands=False)

    out_prot, _ = enh_protected.enhance(noisy, sample_rate=sr)
    out_unprot, _ = enh_unprotected.enhance(noisy, sample_rate=sr)

    # Both should be finite and bounded
    assert np.all(np.isfinite(out_prot))
    assert np.all(np.isfinite(out_unprot))
    # Speech protection should preserve positive correlation with clean voice and exceed noisy baseline
    corr_noisy = np.corrcoef(clean, noisy)[0, 1]
    corr_prot = np.corrcoef(clean, out_prot)[0, 1]
    assert corr_prot > corr_noisy
    assert corr_prot >= 0.78


def test_speech_band_energy_preservation(synthetic_audio_signals):
    """Verify that speech band (300-3400 Hz) energy is not catastrophically attenuated."""
    clean = synthetic_audio_signals["clean"]
    noisy = synthetic_audio_signals["noisy"]
    sr = synthetic_audio_signals["sr"]

    enhancer = NoiseEnhancer(protect_speech_bands=True)
    enhanced, _ = enhancer.enhance(noisy, sample_rate=sr)

    # Compute FFT energy in speech band (300 - 3400 Hz)
    freqs = np.fft.rfftfreq(len(clean), 1.0 / sr)
    speech_band = (freqs >= 300) & (freqs <= 3400)

    clean_speech_energy = np.sum(np.abs(np.fft.rfft(clean)[speech_band]) ** 2)
    enh_speech_energy = np.sum(np.abs(np.fft.rfft(enhanced)[speech_band]) ** 2)

    # Energy ratio should remain reasonable (> 40% of original clean speech energy retained)
    energy_ratio = enh_speech_energy / (clean_speech_energy + 1e-10)
    assert energy_ratio > 0.40


def test_four_channel_multichannel_handling():
    """Test 4-channel surround audio input (4, N) and (N, 4) layout preservation."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    channels_data = [
        (0.4 * np.sin(2 * np.pi * (200 + 100 * ch) * t) + np.random.normal(0, 0.15, len(t))).astype(np.float32)
        for ch in range(4)
    ]

    enhancer = NoiseEnhancer()

    # (4, N) layout
    audio_4xN = np.stack(channels_data, axis=0)
    out_4xN, res_4xN = enhancer.enhance(audio_4xN, sample_rate=sr)
    assert out_4xN.shape == (4, len(t))
    assert res_4xN.enhancement_applied is True
    assert np.all(np.isfinite(out_4xN))
    assert np.max(np.abs(out_4xN)) <= 1.0

    # (N, 4) layout
    audio_Nx4 = np.stack(channels_data, axis=1)
    out_Nx4, res_Nx4 = enhancer.enhance(audio_Nx4, sample_rate=sr)
    assert out_Nx4.shape == (len(t), 4)
    assert res_Nx4.enhancement_applied is True


def test_amplitude_extremes_and_unusual_lengths():
    """Test very low amplitude (whisper level), high amplitude, and prime/odd sample counts."""
    enhancer = NoiseEnhancer()
    sr = 16000

    # Very low amplitude signal (1e-5 RMS)
    low_amp = (1e-5 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, 8000))).astype(np.float32)
    out_low, res_low = enhancer.enhance(low_amp, sample_rate=sr)
    assert np.all(np.isfinite(out_low))
    assert np.max(np.abs(out_low)) <= 1.0

    # Unusual/prime length audio (e.g. 7919 samples)
    odd_len = (0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.4949, 7919))).astype(np.float32)
    out_odd, res_odd = enhancer.enhance(odd_len, sample_rate=sr)
    assert len(out_odd) == 7919
    assert np.all(np.isfinite(out_odd))


