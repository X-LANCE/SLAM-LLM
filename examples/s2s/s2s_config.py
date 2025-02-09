from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class VocabConfig:
    text_vocabsize: int = 151936
    text_specialtokens: int = 64
    audio_vocabsize: int = 4096
    audio_specialtokens: int = 64
    code_layer: int = 7

    padded_text_vocabsize: int = field(init=False)
    padded_audio_vocabsize: int = field(init=False)
    total_audio_vocabsize: int = field(init=False)
    total_vocabsize: int = field(init=False)

    eot: int = field(init=False)   # end of text token
    pad_t: int = field(init=False) # padding text token
    input_t: int = field(init=False) # input text token
    answer_t: int = field(init=False) # answer text token
    asr: int = field(init=False)   # ASR token

    eoa: int = field(init=False)   # end of audio token
    pad_a: int = field(init=False) # padding audio token
    input_a: int = field(init=False) # input audio token
    answer_a: int = field(init=False) # answer audio token
    split: int = field(init=False) # split token

    def __post_init__(self):
        self.padded_text_vocabsize = self.text_vocabsize + self.text_specialtokens
        self.padded_audio_vocabsize = self.audio_vocabsize + self.audio_specialtokens
        self.total_audio_vocabsize = self.padded_audio_vocabsize * self.code_layer
        self.total_vocabsize = self.padded_text_vocabsize + self.total_audio_vocabsize

        self.eot = self.text_vocabsize
        self.pad_t = self.text_vocabsize + 1
        self.input_t = self.text_vocabsize + 2
        self.answer_t = self.text_vocabsize + 3
        self.asr = self.text_vocabsize + 4

        self.eoa = self.audio_vocabsize
        self.pad_a = self.audio_vocabsize + 1
        self.input_a = self.audio_vocabsize + 2
        self.answer_a = self.audio_vocabsize + 3
        self.split = self.audio_vocabsize + 4

@dataclass
class TTSAdapterConfig:
    add_qkv_bias: Optional[bool] = True
    bias: bool = False
    gelu_approximate: Optional[str] = None
    head_size: Optional[int] = 64
    intermediate_size: Optional[int] = 4864
    lm_head_bias: bool = False
    mlp_class_name: str = "GptNeoxMLP"
    n_layer: int = 6
    n_head: int = 14
    n_embd: int = 896
    n_query_groups: Optional[int] = 2
    norm_class_name: str = "RMSNorm"
    norm_eps: float = 1e-6
    parallel_residual: bool = False
    rotary_percentage: float = 1
    shared_attention_norm: bool = False

    def __post_init__(self):
        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

@dataclass
class ModelConfig:
    file: str = "examples/s2s/model/slam_model_s2s.py:model_factory"
    llm_name: str = "vicuna-13b-v1.5"
    llm_path: str = "PATH/to/LLAMA/7B"
    llm_type: str = "decoder_only"
    llm_dim: int = 4096
    encoder_name: Optional[str] = None
    encoder_ds_rate: int = 2
    encoder_path: Optional[str] = None
    encoder_dim: int = 1280
    encoder_projector: str = "linear"
    encoder_projector_ds_rate: int = 5
    modal: str = "audio"
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })
    encoder_type: str = field(default="finetune", metadata={
        "help": "whether model is only pretrained or finetuned, used for models such as hubert"
    })
    vocab_config: VocabConfig = field(default_factory=VocabConfig)
    codec_decode: bool = False
    codec_decoder_type: str = "SNAC"
    codec_decoder_path: Optional[str] = None
    tts_adapter: bool = False
    tts_adapter_config: TTSAdapterConfig = field(default_factory=TTSAdapterConfig)
    encoder_path_hf: Optional[str] = None
    code_type: str = "SNAC" 
    group_decode: bool = False
    group_decode_adapter_type: str = "linear"
    whisper_decode: bool = False


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
    model_name:str = "PATH/to/LLAMA/7B"
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
    run_test_during_validation_prompt:str = "<|S2S|>"
    freeze_llm:bool = field(default=False, metadata={
        "help": "whether to freeze llm when finetuning, should be true when use peft finetuning"
    })
    freeze_encoder:bool = False
    train_embed_only:bool = False
    train_audio_embed_only:bool = False
    train_embed:bool = False
    task_type:str = "s2s"
    freeze_encoder_projector:bool = False
    freeze_group_decode_adapter:bool = False



@dataclass
class DataConfig:
    dataset: str = "speech_dataset_s2s"
    file: str = "examples/s2s/speech_dataset_s2s.py:get_speech_dataset"
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    train_split: str = "train"
    test_split:str = "validation"
    prompt: Optional[str] = None
    data_path: Optional[str] = None
    max_words: Optional[int] = None
    max_mel: Optional[float] = None
    fix_length_audio: int = -1
    inference_mode:bool = False
    input_type: str = field(default="raw", metadata={
                                "help":"Use raw when input is wav, mel when for whisper"
                            })
    mel_size: int = field(default=80, metadata={
        "help": "80 for whisper large v1 and v2, 128 for v3"
    })
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })
    seed: int = 42
    manifest_format: str = field(default="parquet", metadata={ "help": "alternative: jsonl" })
    split_size: float = 0.1

    vocab_config: VocabConfig = field(default_factory=VocabConfig)
    load_from_cache_file: bool = False
    task_type: str = "s2s"
    upsample_text_tokens: bool = False
    upsampling_factor: int = 1
    upsample_method: str = "repeat"
    code_type: str = "SNAC" 
    num_latency_tokens: int = 0
    do_layershift: bool = True

@dataclass
class DecodeConfig:
    do_sample: bool = False
    max_new_tokens: int = 256
    min_length: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    num_samples: int = 1
    max_time: float = 0.0
    text_repetition_penalty: float = 1.0
    audio_repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    no_repeat_ngram_size: int = 0
    bad_words_ids: List = field(default_factory=list)
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    task_type: str = "s2s"
    decode_text_only: bool = False
    streaming: bool = False
    stream_stride: int = 4
    upsampling_factor: int = 1
    input_text: bool = False
    do_layershift: bool = True
    num_latency_tokens: int = 0

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
    wandb_dir: str = "/valleblob/v-wenxichen/exp/wandb_log"
    wandb_entity_name: str = "project_name"
    wandb_project_name: str = "project_name"
    wandb_exp_name: str = "exp_name"
    log_file: str = "/valleblob/v-wenxichen/exp/log/test.log"
    log_interval: int = 10
    online_output_dir: Optional[str] = None

