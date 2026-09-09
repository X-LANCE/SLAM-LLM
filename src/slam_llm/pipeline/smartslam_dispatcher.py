from copy import deepcopy
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class SmartSLAMDispatcher:
    """Dispatch SmartSLAM routing decisions to existing SLAM-LLM recipes."""

    FACTORY_MAP = {
        "whisper": "examples/asr_librispeech/model/slam_model_asr.py:model_factory",
        "eat": "examples/aac_audiocaps/model/slam_model_aac.py:model_factory",
    }

    def __init__(self, routing_result):
        if not isinstance(routing_result, dict):
            raise TypeError("routing_result must be a dictionary")

        encoder = routing_result.get("encoder")
        profile = routing_result.get("slam_llm_profile")

        if encoder not in self.FACTORY_MAP:
            raise ValueError(f"Unsupported SmartSLAM encoder: {encoder}")

        if not isinstance(profile, dict):
            raise ValueError("Routing result is missing slam_llm_profile")

        self.encoder = encoder
        self.profile = deepcopy(profile)

    def get_factory_path(self):
        """Return the existing SLAM-LLM factory selected by the router."""
        return self.FACTORY_MAP[self.encoder]

    def build_model_config(self, model_config):
        """Create a model config using the routed encoder profile."""
        config = OmegaConf.create(OmegaConf.to_container(model_config, resolve=False))

        config.file = self.get_factory_path()

        for key in (
            "encoder_name",
            "encoder_dim",
            "encoder_projector",
            "encoder_projector_ds_rate",
        ):
            if key in self.profile:
                config[key] = self.profile[key]

        return config

    def build_dataset_config(self, dataset_config):
        """Create a dataset config using routed preprocessing settings."""
        config = OmegaConf.create(
            OmegaConf.to_container(dataset_config, resolve=False)
        )

        for key in (
            "input_type",
            "mel_size",
            "fbank_mean",
            "fbank_std",
            "target_length",
        ):
            if key in self.profile:
                config[key] = self.profile[key]

        if self.encoder == "eat":
            config.model_name = "eat"

        return config

    def dispatch(self, model_config, dataset_config):
        """Return routed model and dataset configurations."""
        return {
            "encoder": self.encoder,
            "factory": self.get_factory_path(),
            "model_config": self.build_model_config(model_config),
            "dataset_config": self.build_dataset_config(dataset_config),
        }
