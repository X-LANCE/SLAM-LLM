import hydra
import logging
from typing import Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf

from seld_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig, PeftConfig
from slam_llm.pipeline.finetune import main as train

@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    peft_config: PeftConfig = field(default_factory=PeftConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to projector checkpoint"}
    )

@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    cfg.train_config.peft_config = cfg.peft_config

    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())
    logging.basicConfig(level=log_level)
        
    train(cfg)


if __name__ == "__main__":
    main_hydra()