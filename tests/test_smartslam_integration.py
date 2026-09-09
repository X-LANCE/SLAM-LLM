from pathlib import Path

from omegaconf import OmegaConf

from slam_llm.enhancement.noise_enhancer import NoiseEnhancer
from slam_llm.models.smartslam_router import SmartSLAMRouter
from slam_llm.pipeline.smartslam_dispatcher import SmartSLAMDispatcher


SPEECH_AUDIO = Path("examples/s2s/audio_prompt/en/prompt_1.wav")
NON_SPEECH_AUDIO = Path("examples/s2s/audio_prompt/en/prompt_6.wav")


def make_base_configs():
    model_config = OmegaConf.create({
        "file": None,
        "encoder_name": None,
        "encoder_dim": 1280,
        "encoder_projector": "linear",
        "encoder_projector_ds_rate": 5,
    })

    dataset_config = OmegaConf.create({
        "input_type": "raw",
        "mel_size": 80,
    })

    return model_config, dataset_config


def test_speech_audio_end_to_end_dispatch():
    enhancer = NoiseEnhancer()
    router = SmartSLAMRouter()

    enhanced_audio, enhancement_result = enhancer.enhance(
        str(SPEECH_AUDIO)
    )

    import torch
    routing = router.route(torch.from_numpy(enhanced_audio))

    assert routing["modality"] == "speech"
    assert routing["encoder"] == "whisper"
    assert enhanced_audio.dtype.name == "float32"
    assert enhancement_result.sample_rate > 0

    model_config, dataset_config = make_base_configs()
    dispatcher = SmartSLAMDispatcher(routing)
    result = dispatcher.dispatch(model_config, dataset_config)

    assert result["encoder"] == "whisper"
    assert result["model_config"].encoder_name == "whisper"
    assert result["model_config"].encoder_dim == 1280
    assert result["model_config"].encoder_projector == "linear"
    assert result["model_config"].encoder_projector_ds_rate == 5
    assert result["dataset_config"].input_type == "mel"
    assert result["dataset_config"].mel_size == 128


def test_non_speech_audio_end_to_end_dispatch():
    enhancer = NoiseEnhancer()
    router = SmartSLAMRouter()

    enhanced_audio, enhancement_result = enhancer.enhance(
        str(NON_SPEECH_AUDIO)
    )

    import torch
    routing = router.route(torch.from_numpy(enhanced_audio))

    assert routing["modality"] == "non_speech"
    assert routing["encoder"] == "eat"
    assert enhanced_audio.dtype.name == "float32"
    assert enhancement_result.sample_rate > 0

    model_config, dataset_config = make_base_configs()
    dispatcher = SmartSLAMDispatcher(routing)
    result = dispatcher.dispatch(model_config, dataset_config)

    assert result["encoder"] == "eat"
    assert result["model_config"].encoder_name == "eat"
    assert result["model_config"].encoder_dim == 768
    assert result["model_config"].encoder_projector == "linear"
    assert result["model_config"].encoder_projector_ds_rate == 5
    assert result["dataset_config"].input_type == "mel"
    assert result["dataset_config"].mel_size == 80
    assert result["dataset_config"].model_name == "eat"
    assert result["dataset_config"].fbank_mean == -4.268
    assert result["dataset_config"].fbank_std == 4.569
    assert result["dataset_config"].target_length == 1024


def test_smartslam_pipeline_prepare():
    from slam_llm.pipeline.smartslam_pipeline import SmartSLAMPipeline

    model_config, dataset_config = make_base_configs()
    pipeline = SmartSLAMPipeline(rag_directory="rag_data")

    result = pipeline.prepare(
        SPEECH_AUDIO,
        "How does RAG help SmartSLAM?",
        model_config,
        dataset_config,
        top_k=3,
    )

    try:
        assert Path(result["input_wav"]).resolve() == SPEECH_AUDIO.resolve()
        assert Path(result["enhanced_wav"]).is_file()

        assert result["enhancement"].sample_rate > 0

        assert result["routing"]["modality"] == "speech"
        assert result["routing"]["encoder"] == "whisper"
        assert result["routing"]["confidence"] > 0

        assert result["dispatch"]["encoder"] == "whisper"
        assert result["dispatch"]["model_config"].encoder_name == "whisper"
        assert result["dispatch"]["model_config"].encoder_dim == 1280

        assert len(result["retrieved_results"]) > 0
        assert result["rag_confidence"] > 0
        assert isinstance(result["rag_explanation"], str)
        assert len(result["rag_explanation"]) > 0
        assert isinstance(result["augmented_prompt"], str)
        assert len(result["augmented_prompt"]) > 0
    finally:
        pipeline.cleanup(result)

    assert not Path(result["enhanced_wav"]).exists()


def test_smartslam_pipeline_inference_bridge(monkeypatch):
    from slam_llm.pipeline import smartslam_pipeline
    from slam_llm.pipeline.smartslam_pipeline import SmartSLAMPipeline

    calls = {}

    class FakeTokenizer:
        def batch_decode(self, outputs, **kwargs):
            calls["decode_kwargs"] = kwargs
            return ["fake SmartSLAM response"]

    class FakeModel:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def inference(self, wav_path=None, prompt=None,
                      dataset_config=None, device=None):
            calls["inference_wav"] = str(wav_path)
            calls["inference_prompt"] = prompt
            calls["inference_dataset_config"] = dataset_config
            calls["inference_device"] = device
            return ["fake_tokens"]

    def fake_factory(train_config, model_config, **kwargs):
        calls["factory_model_config"] = model_config
        calls["factory_kwargs"] = kwargs
        return FakeModel(), FakeTokenizer()

    import builtins
    import types

    fake_model_utils = types.ModuleType("slam_llm.utils.model_utils")
    fake_model_utils.get_custom_model_factory = (
        lambda model_config, logger: fake_factory
    )

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "slam_llm.utils.model_utils":
            return fake_model_utils
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model_config, dataset_config = make_base_configs()
    pipeline = SmartSLAMPipeline(rag_directory="rag_data")

    result = pipeline.inference(
        wav_path=SPEECH_AUDIO,
        prompt="How does RAG help SmartSLAM?",
        train_config=OmegaConf.create({
            "enable_fsdp": False,
            "enable_ddp": False,
        }),
        model_config=model_config,
        dataset_config=dataset_config,
        top_k=3,
        device="cpu",
    )

    try:
        assert result["routing"]["encoder"] == "whisper"
        assert result["dispatch"]["model_config"].encoder_name == "whisper"

        assert calls["factory_model_config"].encoder_name == "whisper"
        assert calls["factory_model_config"].encoder_dim == 1280

        assert Path(calls["inference_wav"]).is_file()
        assert "How does RAG help SmartSLAM?" in calls["inference_prompt"]

        assert calls["inference_dataset_config"].input_type == "mel"
        assert calls["inference_dataset_config"].mel_size == 128
        assert calls["inference_device"] == "cpu"

        assert calls["device"] == "cpu"
        assert calls["eval"] is True

        assert result["model_outputs"] == ["fake_tokens"]
        assert result["output_text"] == ["fake SmartSLAM response"]

        assert calls["decode_kwargs"]["add_special_tokens"] is False
        assert calls["decode_kwargs"]["skip_special_tokens"] is True
    finally:
        pipeline.cleanup(result)

    assert not Path(result["enhanced_wav"]).exists()
