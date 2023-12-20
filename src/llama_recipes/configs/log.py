# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class log_config:
    use_wandb: bool = False
    wandb_dir: str = "/root/test_wandb"
    wandb_entity_name : str = "project_name"
    wandb_project_name : str = "project_name"
    wandb_exp_name : str = "exp_name"
    log_file: str="/root/test.log"
    log_interval: int = 5