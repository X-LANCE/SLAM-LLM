import logging

import torch


def count_model_parameters(model):
    if not isinstance(model, torch.nn.Module):
        return 0, 0
    name = f"{model.__class__.__name__} {model.__class__}"
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size = num * 4.0 / 1024.0 / 1024.0 # float32, MB
    logging.info(f"#param of {name} is {num} = {size:.1f} MB (float32)")
    return num, size
