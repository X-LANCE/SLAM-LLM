import torch
from llama_recipes.models.slam_model import setup_model, setup_tokenizer
from llama_recipes.utils.train_utils import print_model_size
import os

def model_factory(train_config, model_config, **kwargs):

    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    model = setup_model(tokenizer, train_config, model_config, **kwargs)
    ckpt_path = kwargs.get("ckpt_path", None) #FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
            print("loading other parts from: ", ckpt_path)
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(model, train_config, int(os.environ["RANK"]) if train_config.enable_fsdp else 0)
    return model, tokenizer
