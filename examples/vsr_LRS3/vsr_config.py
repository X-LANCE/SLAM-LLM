from dataclasses import dataclass, field
from typing import Optional, List
@dataclass
class ModelConfig:
    file: str = "examples/vsr_LRS3/model/slam_model_vsr.py:model_factory"
    llm_name: str = "vicuna-7b-v1.5"
    llm_path: str = "PATH/to/Vicuna/7B"
    llm_type: str = "decoder_only"
    llm_dim: int = 4096
    encoder_name: Optional[str] = "av_hubert"
    encoder_path: Optional[str] = "PATH/to/self_large_vox_433h.pt"
    encoder_dim: int = 1024
    encoder_projector: str = "cov1d-linear"
    encoder_projector_ds_rate: int = 5
    
@dataclass
class PeftConfig:
    peft_method: str = "lora" # None , llama_adapter, prefix
    r: int = 8
    lora_alpha: int = 32
    target_modules: List = field(default_factory=lambda: [ "q_proj", "v_proj" ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class TrainConfig:
    model_name:str = "av_hubert"
    enable_ddp:bool = False
    enable_deepspeed:bool = False
    enable_fsdp:bool = False
    low_cpu_fsdp:bool = False
    run_validation:bool = True
    batch_size_training:int = 4
    batching_strategy:str = field(default="packing", metadata={
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
    peft_config:PeftConfig = field(default_factory=PeftConfig)
    output_dir:str = "PATH/to/save/PEFT/model"
    freeze_layers:bool = False
    num_freeze_layers:int = 1
    quantization:bool = False
    one_gpu:bool = False
    save_model:bool = True
    dist_checkpoint_root_folder:str = "PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder:str = "fine-tuned" # will be used if using FSDP
    save_optimizer:bool = False # will be used if using FSDP
    use_fast_kernels:bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    run_test_during_validation:bool = False
    run_test_during_validation_file:str = "test.wav"
    run_test_during_validation_prompt:str = "<|ASR|>"
    freeze_llm:bool = field(default=False, metadata={
        "help": "whether to freeze llm when finetuning, should be True when use peft finetuning"
    })
    freeze_encoder:bool = False

@dataclass
class DataConfig:
    fix_length_audio: int = -1
    inference_mode: bool = False
    dataset: str = "avhubert_dataset"
    file: str = "src/slam_llm/datasets/avhubert_dataset.py:get_audio_dataset"
    data: str = "/nfs/yangguanrou.ygr/LRS_new/433h_data/"
    train_split: str = "train"
    test_split: str = "val"
    labels: List = field(default_factory=lambda:["wrd"])
    label_dir: str = "/nfs/yangguanrou.ygr/LRS_new/433h_data"
    label_rate: int = -1
    is_s2s: bool = True
    noise_wav: Optional[str] = None
    noise_snr: str = "0.0"
    noise_num: int = 1
    sample_rate: int = 16000
    normalize: bool = True
    enable_padding: bool = False
    max_sample_size: int = 500
    min_sample_size: int = 0
    max_trim_sample_size: int = 500
    single_target: bool = True
    random_crop: bool = False
    pad_audio: bool = True
    pdb: bool = False
    stack_order_audio: int = 1
    skip_verify: bool = False
    # image_aug: True
    image_crop_size: int = 88
    image_mean: float = 0.421
    image_std: float = 0.165
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    fine_tuning: bool = True
    modal: str = "VO"
    modalities: List = field(default_factory=lambda: ['video'])
    shuffle: bool = True
    prompt: str = "Transcribe the silent speech in this video to text by lip-reading the speaker's clear and visible lip movements."

@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    use_fp16: bool = False
    # sharding_strategy = "FULL_SHARD" #ShardingStrategy = ShardingStrategy.FULL_SHARD
    sharding_strategy: str = "NO_SHARD" #ShardingStrategy.NO_SHARD #MZY: set NO_SHARD when use DDP
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
