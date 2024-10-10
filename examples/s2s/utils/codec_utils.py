from snac import SNAC
from slam_llm.utils.train_utils import print_module_size, print_model_size
import os

def setup_codec(train_config, model_config, **kwargs):
    if model_config.codec_decoder_type == "SNAC":
        codec_decoder = SNAC.from_pretrained(model_config.codec_decoder_path).eval()
    else:
        raise NotImplementedError
    print_module_size(codec_decoder, model_config.codec_decoder_type, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    
    return codec_decoder