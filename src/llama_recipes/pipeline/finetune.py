# os
import os
import fire
import random
import importlib

# nn
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# opt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

# config
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import model_config as MODEL_CONFIG
from llama_recipes.data.concatenator import ConcatDataset

# util
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    get_policies
)

from model_factory import model_factory
import sys
import logging
import wandb

def main(**kwargs):

    # Update the configuration for the training and sharding process
    train_config, fsdp_config, model_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG()
    update_config((train_config, fsdp_config, model_config), **kwargs)

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()  
    #logger = logging.getLogger("finetune")  #不知道为啥那么建完是warning
    logger.setLevel(logging.INFO)

    # 修改默认的console handler
    # logger.handlers logger.handlers[0]
    file_handler = logging.FileHandler(filename=train_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # console_handler.setFormatter(console_formatter)    

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 

    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    # logger.propagate=False

    # effective_level = logger.getEffectiveLevel()  # 打印有效日志级别
    # print("Effective log level:", effective_level)

    wandb_config={"train_config":vars(train_config), "fsdp_config":vars(fsdp_config), "model_config":vars(model_config)}
    wandb.init(project="project_name",name="exp_name",config=wandb_config) #记录参数
    
    logger.info("train_config: {}".format(train_config))
    logger.info("fsdp_config: {}".format(fsdp_config))
    logger.info("model_config: {}".format(model_config))




    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:  #x
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    if torch.distributed.is_initialized(): #x
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.  #device(type='cuda')
    model.to(device)

    
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:  #x
        model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy, #(FIX:MZY): Using my_auto_wrapping_policy whether peft or not. This will avoid model shard type check error of requires_grad mismatching.
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:  #
        model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)
    logger.info("dataset_config: {}".format(dataset_config))
    # wandb.config.update( {"dataset_config": vars(dataset_config)} )  数据集里有notimplemented
    
    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    if not train_config.enable_fsdp or rank == 0: #
        logger.info(f"--> Training Set Length = {len(dataset_train)}")  #60
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="val",
    )
    if not train_config.enable_fsdp or rank == 0:
        logger.info(f"--> Validation Set Length = {len(dataset_val)}")  #20
    if train_config.batching_strategy == "packing":  #x   是custom
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    # for i, batch in enumerate(train_dataloader):
    #     if batch["inputBatch0"].shape[1]<10000:
    #         print(batch["inputBatch0"].shape)
    #         print(batch["inputBatch2"].shape)
    #         print(batch["targetLenBatch"])
    #         print(i)
    #     if i%1000==0:
    #         print("step:%d"%i)

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing": #x
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else: #
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,  #1
        train_config,
        fsdp_config if train_config.enable_fsdp else None,  #false
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [logger.info(f'Key: {k}, Value: {v}') for k, v in results.items()]

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)