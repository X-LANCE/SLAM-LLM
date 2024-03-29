import torch
from llama_recipes.models.slam_model import setup_model, setup_tokenizer
from llama_recipes.utils.train_utils import print_model_size
import os

import logging
logger = logging.getLogger(__name__)


def model_factory(train_config, model_config, **kwargs):

    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    model = setup_model(tokenizer, train_config, model_config, **kwargs)

    ckpt_path = kwargs.get("ckpt_path", None) #FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None and model_config.encoder_name == 'eat':
        logger.info("loading other parts from: {} and there will be a bug when using q-former or lora~".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        updated_ckpt_dict = {key.replace('encoder_projector.', ''): value for key, value in ckpt_dict.items()}
        model.encoder_projector.load_state_dict(updated_ckpt_dict, strict=False)   # fixme: 暂时只 load 线性层
    
    elif ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)
            

    print_model_size(model, train_config, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model, tokenizer
