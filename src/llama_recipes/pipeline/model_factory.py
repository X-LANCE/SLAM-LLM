import torch
from llama_recipes.models.slam_model import setup_model, setup_tokenizer

def model_factory(train_config, model_config, **kwargs):

    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    model = setup_model(tokenizer, train_config, model_config, **kwargs)
    ckpt_path = kwargs.get("ckpt_path", None) #FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
            print("loading ckpt from: ", ckpt_path)
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt_dict, strict=False)

    return model, tokenizer
