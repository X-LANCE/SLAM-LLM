from generate.generate_s2s_batch_stream_mini_omni import main as batch_inference_stream_mini_omni
from generate.generate_s2s_online_stream_mini_omni import main as inference_online_stream_mini_omni
from generate.generate_s2s_batch import main as batch_inference
from generate.generate_s2s_online import main as inference_online # single-round inference
from generate.generate_s2s_online_multi_round import main as inference_online_multi_round # multi-round inference
from generate.generate_s2s_batch_multi_round import main as batch_inference_multi_round


import hydra
import logging
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional
from s2s_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig, DecodeConfig


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    decode_config: DecodeConfig = field(default_factory=DecodeConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: str = field(
        default="output/model.pt", metadata={"help": "The path to model backbone checkpoint"}
    )
    peft_ckpt_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to peft checkpoint",
        },
    )
    output_text_only: bool = field(
        default=False, metadata={"help": "Decode text only"}
    )
    inference_online: bool = field(
        default=False, metadata={"help": "Inference online"}
    )
    inference_streaming: bool = field(
        default=False, metadata={"help": "Inference stream"}
    )
    speech_sample_rate: int = field(
        default=24000, metadata={"help": "The sample rate for speech"}
    )
    audio_prompt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to audio prompt"}
    )
    multi_round: bool = field(
        default=False, metadata={"help": "Multi-round inference"}
    )
    mini_omni_modeling: bool = field(
        default=False, metadata={"help": "Mini Omni modeling"}
    )        
    batch_input_jsonl: Optional[str] = field(
        default=None, metadata={"help": "The path to batch input jsonl"}
    )


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    # kwargs = to_plain_list(cfg)
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if cfg.get("debug", False):
        import pdb

        pdb.set_trace()

    if cfg.mini_omni_modeling:
        if cfg.inference_online:
            if cfg.inference_streaming:
                inference_online_stream_mini_omni(cfg)
            else:
                inference_online(cfg)
        else:
            if cfg.inference_streaming:
                batch_inference_stream_mini_omni(cfg)
            else:
                batch_inference(cfg)

    else:
        if cfg.inference_online:
            if cfg.multi_round:
                inference_online_multi_round(cfg)
            else:
                inference_online(cfg)
        else:
            if cfg.multi_round:
                batch_inference_multi_round(cfg)
            else:
                batch_inference(cfg)

if __name__ == "__main__":
    main_hydra()
