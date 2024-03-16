import torch
from slam_llm.utils.train_utils import print_model_size
import os

import logging
logger = logging.getLogger(__name__)

def model_factory(train_config, model_config, **kwargs):
    llm_type=model_config.get("llm_type", "decoder_only")
    if llm_type == "decoder_only":
        from slam_llm.models.slam_model import setup_model, setup_tokenizer
    else:
        from slam_llm.models.slam_model_t5 import setup_model, setup_tokenizer

    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    model = setup_model(tokenizer, train_config, model_config, **kwargs)

    ckpt_path = kwargs.get("ckpt_path", None) #FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
            logger.info("loading other parts from: {}".format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(model, train_config, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model, tokenizer
