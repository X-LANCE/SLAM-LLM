import torch
import torch.nn as nn
from slam_llm.utils.train_utils import print_module_size
import os

class Linear_GroupDecodeAdapter(nn.Module):
    def __init__(self, audio_vocab_size, code_layer):
        super(Linear_GroupDecodeAdapter, self).__init__()
        self.audio_vocab_size = audio_vocab_size
        self.code_layer = code_layer
        self.linear = nn.Linear(audio_vocab_size, code_layer * audio_vocab_size)

    def forward(self, logits):
        logits = self.linear(logits)
        return logits


def setup_group_decode_adapter(model_config, train_config, **kwargs):
    audio_vocab_size = model_config.vocab_config.total_audio_vocabsize
    code_layer = model_config.vocab_config.code_layer
    
    if model_config.group_decode_adapter_type == "linear":
        group_decode_adapter = Linear_GroupDecodeAdapter(audio_vocab_size, code_layer)
    else:
        raise NotImplementedError

    group_decode_adapter_name = "GroupDecodeAdapter_" + model_config.group_decode_adapter_type
    print_module_size(group_decode_adapter, group_decode_adapter_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return group_decode_adapter
