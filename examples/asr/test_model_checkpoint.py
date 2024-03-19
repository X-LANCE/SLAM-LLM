from slam_llm.pipeline.model_factory import model_factory
from slam_llm.model_checkpointing import save_model_checkpoint_peft

import hydra
import os
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig

@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: str = field(
        default="output/model.pt", metadata={"help": "The path to projector checkpoint"}
    )
    peft_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to peft checkpoint, should be a directory including adapter_config.json"
        },
    )

@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    
    # kwargs = to_plain_list(cfg)
    kwargs = cfg
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
    
    logging.basicConfig(level=log_level)
    
    if kwargs.get("debug", False):
        import pdb;
        pdb.set_trace()
        
    train_config, fsdp_config, model_config, log_config, dataset_config = kwargs.train_config, \
                                                                          kwargs.fsdp_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config
    del kwargs.train_config
    del kwargs.fsdp_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    save_model_checkpoint_peft( model, None, 0, train_config, checkpoint_name="new_checkpoint")
    state_dict = torch.load(os.path.join(train_config.output_dir, "new_checkpoint", "model.pt"))
    state_dict = model.state_dict()

    kwargs.peft_ckpt = None
    kwargs.ckpt_path = os.path.join(train_config.output_dir, "new_checkpoint", "model.pt")
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    new_state_dict = model.state_dict()
    for item in state_dict:
        print(item)
        if not (state_dict[item]==new_state_dict[item]).all():
            print("not equal")
            raise
    





if __name__ == "__main__":
    main_hydra()