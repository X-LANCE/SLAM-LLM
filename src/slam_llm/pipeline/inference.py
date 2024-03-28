# import fire
import logging
import random
import torch
# import argparse
from slam_llm.models.slam_model import slam_model
# config
# from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
# from llama_recipes.configs import model_config as MODEL_CONFIG

from slam_llm.utils.model_utils import get_custom_model_factory

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
	def to_plain_list(cfg_item):
		if isinstance(cfg_item, ListConfig):
			return OmegaConf.to_container(cfg_item, resolve=True)
		elif isinstance(cfg_item, DictConfig):
			return {k: to_plain_list(v) for k, v in cfg_item.items()}
		else:
			return cfg_item
	
	# kwargs = to_plain_list(cfg)
	kwargs = cfg
	log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
	
	logging.basicConfig(level=log_level)
	
	if kwargs.get("debug", False):
		import pdb;
		pdb.set_trace()
	
	main(kwargs)

def main(kwargs: DictConfig):

	# Update the configuration for the training and sharding process
	# train_config, fsdp_config, model_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG()
	# update_config((train_config, fsdp_config, model_config), **kwargs)
	train_config, fsdp_config, model_config, log_config, dataset_config = kwargs.train_config, \
	                                                                      kwargs.fsdp_config, \
	                                                                      kwargs.model_config, \
	                                                                      kwargs.log_config, \
	                                                                      kwargs.dataset_config
	
	del kwargs.train_config
	del kwargs.fsdp_config
	del kwargs.model_config
	del kwargs.log_config
	del kwargs.dataset_config
	
	# Set the seeds for reproducibility
	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	
	model_factory = get_custom_model_factory(model_config, logger)
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.
	model.to(device)
	model.eval()
	
	while True:
		print("=====================================")
		wav_path = input("Your Wav Path:\n")
		prompt = input("Your Prompt:\n")
		# wav_path = kwargs.get('wav_path')
		# prompt = kwargs.get('prompt')
		try:
			model_outputs = model.inference(wav_path, prompt)
			output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
			print(output_text)
		except:
			continue



if __name__ == "__main__":
	main_hydra()