# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
from pathlib import Path
from datetime import datetime
import torch
import time
from collections import OrderedDict

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist


import logging
logger = logging.getLogger(__name__)


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    logger.info(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(model, rank, cfg):
    # torch.manual_seed(103)
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            logger.info(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
         logger.info(f"loading model from model path: {load_dir} ")
    reader = FileSystemReader(load_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            logger.info(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            logger.info(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            logger.info(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        logger.info(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg,optim=None):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        logger.info(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        logger.info(f"Sharded state checkpoint saved to {save_dir}")
        logger.info(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )
def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()

        logger.info(f"saving process: rank {rank}  done w model state_dict\n")
   

    if rank == 0:
        logger.info(f"--> saving model ...")
        # create save path
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        
        logger.info(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")

def save_model_checkpoint_deepspeed(model, cfg, checkpoint_name="checkpoint"):
    logger.info(f"--> saving model ...")
    save_dir = os.path.join(cfg.output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    # save_full_path = os.path.join(save_dir, "model.pt")
    save_full_path = save_dir
    model.save_checkpoint(save_dir=save_full_path, exclude_frozen_parameters=True)
    logger.info(f"encoder saved at {save_full_path}")
      
def save_model_checkpoint_peft(model, optimizer, rank, cfg, checkpoint_name="checkpoint", save_trainable_only=True):
    logger.info(f"--> saving model ...")
    save_dir = os.path.join(cfg.output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    save_full_path = os.path.join(save_dir, "model.pt")
    if cfg.enable_ddp:
        model = model.module
    cpu_state = model.state_dict()
    if save_trainable_only:
        state_dict = OrderedDict()
        for name, para in model.named_parameters():
            if para.requires_grad:
                state_dict[name] = cpu_state[name]
    else:
        state_dict = cpu_state
    torch.save(state_dict, save_full_path)
    logger.info(f"encoder saved at {save_full_path}")
    
def save_model_checkpoint_peft_full_shard(model, optimizer, rank, cfg, epoch=0):
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()
        logger.info(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        logger.info(f"--> saving model ...")
        save_dir = os.path.join(cfg.output_dir, cfg.model_name, str(epoch+1))
        os.makedirs(save_dir, exist_ok=True)

        if not cfg.freeze_llm:
            llm_dict = {}
            for key in cpu_state.keys():
                if key.startswith("llm."):
                    llm_dict[key] = cpu_state[key]
            model.llm.save_pretrained(save_directory=save_dir, state_dict=llm_dict)
            logger.info(f"llm saved at {save_dir}")

        save_full_path = os.path.join(save_dir, "model.pt")
        encoder_dict = {}
        if not cfg.freeze_encoder:
            for key in cpu_state.keys():
                if key.startswith("encoder."):
                    encoder_dict[key] = cpu_state[key]
        for key in cpu_state.keys():
            if key.startswith("encoder_projector."):
                encoder_dict[key] = cpu_state[key]
        torch.save(encoder_dict, save_full_path)
        logger.info(f"encoder saved at {save_full_path}")

        logger.info(f"model checkpoint saved for epoch {epoch+1}\n")
        
    dist.barrier()

def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    if not full_state_dict_model_path.is_file():
        logger.info(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return


    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    
    logger.info(f"model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

   
    logger.info(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    
    logger.info(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        logger.info(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        logger.info(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """


    if not optimizer_checkpoint_path.is_file():
        logger.info(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    logger.info(f"optimizer shard loaded on rank {rank}")

def load_sharded_model_single_gpu(model,model_path):
    
    reader = FileSystemReader(model_path)
    
    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader= FileSystemReader(model_path),
                no_dist=True,
            )
    
    model.load_state_dict(state_dict["model"])
    
    logger.info(f"Sharded state checkpoint loaded from {model_path}")
    return model
