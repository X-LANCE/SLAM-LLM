import fire
import random
import torch
# import argparse
from llama_recipes.models.slam_model import slam_model
# config
from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import model_config as MODEL_CONFIG
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.pipeline.model_factory import model_factory

def main(**kwargs):

	# Update the configuration for the training and sharding process
	train_config, fsdp_config, model_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG()
	update_config((train_config, fsdp_config, model_config), **kwargs)
	
	# Set the seeds for reproducibility
	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	model.to(kwargs.get("device", "cuda"))
	model.eval()
	
	print("=====================================")
	# wav_path = input("Your Wav Path:\n")
	# prompt = input("Your Prompt:\n")
	wav_path = "/nfs/zhifu.gzf/data/IEMOCAP_full_release/Session1/sentences/wav/Ses01M_impro01/Ses01M_impro01_M001.wav"
	print(model.generate(wav_path))



if __name__ == "__main__":
	fire.Fire(main)