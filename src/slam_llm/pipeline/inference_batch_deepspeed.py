# os
import os
import fire
import deepspeed
import random
import importlib
from tqdm import tqdm
# nn
import torch
import torch_npu
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.utils.data import DistributedSampler
# opt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from slam_llm.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
import torch.distributed as dist
# config
# from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
# from llama_recipes.configs import model_config as MODEL_CONFIG
# from llama_recipes.configs import log_config as LOG_CONFIG
from slam_llm.data.concatenator import ConcatDataset

# util
from slam_llm.utils import fsdp_auto_wrap_policy
from slam_llm.utils.config_utils import get_dataloader_kwargs

from slam_llm.utils.dataset_utils import get_preprocessed_dataset, load_module_from_py_file
from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.deepspeed_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
)

import sys
import logging
import wandb

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path

@hydra.main(config_name=None, version_base=None)  # strict=False 允许忽略未知参数)
def main_hydra(cfg: DictConfig):
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
        
    main(kwargs)


def main(kwargs: DictConfig):
    # Update the configuration for the training and sharding process
    # train_config, fsdp_config, model_config, log_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG(), LOG_CONFIG()
    # update_config((train_config, fsdp_config, model_config, log_config), **kwargs)

    train_config, model_config, log_config, dataset_config = kwargs.train_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config
    del kwargs.train_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config
    
    # Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 

    logger.addHandler(file_handler)


    # Set the seeds for reproducibility
    torch_npu.npu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    deepspeed.init_distributed(
        dist_backend='nccl',    # 使用NCCL后端（GPU场景）
    )

    if rank == 0:
        logger.info("train_config: {}".format(train_config))
        logger.info("model_config: {}".format(model_config))
        logger.info("log_config: {}".format(log_config))

    # Set wandb
    if rank == 0:
        if log_config.use_wandb:
            if not os.path.exists(log_config.wandb_dir):
                os.makedirs(log_config.wandb_dir, exist_ok=True)
            wandb_config={"train_config": train_config, "model_config": model_config, "log_config": log_config}
            wandb.init(dir=log_config.wandb_dir, entity=log_config.wandb_entity_name, project=log_config.wandb_project_name,name=log_config.wandb_exp_name ,config=wandb_config)


    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    device = torch.device(f"npu:{local_rank}" if torch.npu.is_available() else "cpu") # FIX(MZY): put the whole model to device.
    model.to(device)
    model.eval()
    logger.info("dataset_config: {}".format(dataset_config))
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    # sampler = DistributedSampler(
    #             dataset_test,
    #             rank=dist.get_rank(),
    #             num_replicas=dist.get_world_size(),
    #         )
    test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            shuffle=False,
            batch_size=train_config.val_batch_size,
            drop_last=False,
            collate_fn=dataset_test.collator,
            # sampler=sampler
            # multiprocessing_context=mp.get_context("spawn")
        )

    logger.info("=====================================")
    pred_path = kwargs.get('decode_log') + f"_pred"
    gt_path = kwargs.get('decode_log') + f"_gt"
    pred_result = ""
    gt_result = ""
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_dataloader)):
            for key in batch.keys():
                batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
            model_outputs = model.generate(**batch)
            if hasattr(model, 'tokenizer'):
                output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
            else:
                output_text = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
            for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
                pred_result += key + " " + text.strip() + "\n"
                gt_result += key + " " + target + "\n"
    with open(pred_path, "a+") as pred, open(gt_path, "a+") as gt:
        pred.write(pred_result)
        gt.write(gt_result)
if __name__ == "__main__":
    main_hydra()