# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
# from dataclasses import asdict

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from peft import (
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq

# from llama_recipes.configs import datasets, lora_config, llama_adapter_config, prefix_config, train_config
from slam_llm.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler

from omegaconf import OmegaConf

import logging
logger = logging.getLogger(__name__)

# def update_config(config, **kwargs):
#     if isinstance(config, (tuple, list)):
#         for c in config:
#             update_config(c, **kwargs)
#     else:
#         for k, v in kwargs.items():
#             if hasattr(config, k):
#                 setattr(config, k, v)
#             elif "." in k:
#                 # allow --some_config.some_param=True
#                 config_name, param_name = k.split(".")
#                 if type(config).__name__ == config_name:
#                     if hasattr(config, param_name):
#                         setattr(config, param_name, v)
#                     else:
#                         # In case of specialized config we can warm user
#                         logger.warning(f"Warning: {config_name} does not accept parameter: {k}")
#             elif isinstance(config, train_config):
#                 logger.warning(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config):
    # configs = (lora_config, llama_adapter_config, prefix_config)
    # peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    peft_configs = {"lora": LoraConfig,
                    "llama_adapter": AdaptionPromptConfig,
                    "prefix": PrefixTuningConfig
                    }
    # names = tuple(c.__name__.rstrip("_config") for c in configs)
    #
    # assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"
    #
    # config = configs[names.index(train_config.peft_method)]()
    config = train_config.peft_config

    params = OmegaConf.to_container(config, resolve=True)
    # peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    params.pop("peft_method", None) #(FIX:MZY): remove peft_method from params to avoid error
    peft_config = peft_configs[config.get("peft_method", "lora")](**params)

    return peft_config


def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
        kwargs = {}
        batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
        if train_config.batching_strategy == "padding":
            if train_config.enable_fsdp or train_config.enable_ddp or train_config.enable_deepspeed:
                kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                    shuffle=mode=="train",
                )
            else:
                kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
            kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        elif train_config.batching_strategy == "packing":
            if train_config.enable_fsdp or train_config.enable_ddp or train_config.enable_deepspeed:
                kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = default_data_collator
        else:
            # raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")
            if train_config.enable_fsdp or train_config.enable_ddp or train_config.enable_deepspeed:
                kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode=="train",
            )
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = dataset.collator
            logger.info(f"Using batching strategy: {train_config.batching_strategy}")

        return kwargs
