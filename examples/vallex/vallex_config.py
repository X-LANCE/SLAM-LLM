from dataclasses import dataclass, field
from typing import Optional, List

from torch.distributed.fsdp import ShardingStrategy


@dataclass
class ModelConfig:
    llm_name: str = "vallex"
    llm_type: str = "decoder_only"
    ar_at_dict: str = ""
    ar_st_dict: str = ""
    nar_at_dict: str = ""
    nar_st_dict: str = ""
    only_ar: bool = False
    only_nar: bool = False
    encoder_name: str = ""
    llm_path: str = ""
    encoder_projector: str = "null"
    
@dataclass
class TrainConfig:
    model_name:str = "PATH/to/LLAMA/7B"
    enable_ddp:bool = False
    enable_deepspeed:bool = False
    enable_fsdp:bool = False
    low_cpu_fsdp:bool = False
    run_validation:bool = True
    batch_size_training:int = 4
    batching_strategy:str = field(default="custom", metadata={
        "help":"alternative: padding"
    }) #
    context_length:int = 4096
    gradient_accumulation_steps:int = 1
    num_epochs:int = 3
    num_workers_dataloader:int = 1
    warmup_steps:int = 1000
    total_steps:int = 100000
    validation_interval:int = 1000
    lr:float = 1e-4
    weight_decay:float = 0.0
    gamma:float = 0.85
    seed:int = 42
    use_fp16:bool = False
    mixed_precision:bool = True
    val_batch_size:int = 1

    use_peft:bool = False
    output_dir:str = "PATH/to/save/PEFT/model"
    freeze_layers:bool = False
    num_freeze_layers:int = 1
    quantization:bool = False
    one_gpu:bool = False
    save_model:bool = True
    dist_checkpoint_root_folder:str = "PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder:str = "fine-tuned" # will be used if using FSDP
    save_optimizer:bool = False # will be used if using FSDP
    run_test_during_validation:bool = False
    freeze_llm: bool = False

@dataclass
class DataConfig:
    dataset: str = "vallex_dataset"
    file: str = "src/slam_llm/datasets/vallex_dataset.py:get_vallex_dataset"
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    train_split: str = "train"
    test_split: str = "valid"
    
@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    use_fp16: bool = False
    # sharding_strategy = "FULL_SHARD" #ShardingStrategy = ShardingStrategy.FULL_SHARD
    sharding_strategy: ShardingStrategy = "NO_SHARD" #ShardingStrategy.NO_SHARD #MZY: set NO_SHARD when use DDP
    checkpoint_type: str = "SHARDED_STATE_DICT"  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"

@dataclass
class LogConfig:
    use_wandb: bool = False
    wandb_dir: str = "/root/test_wandb"
    wandb_entity_name: str = "project_name"
    wandb_project_name: str = "project_name"
    wandb_exp_name: str = "exp_name"
    log_file: str = "/root/test.log"
    log_interval: int = 5
