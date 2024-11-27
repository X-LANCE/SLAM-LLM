
import hydra
import logging
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional
from asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig
# import fire
import random
import torch
import logging
import sacrebleu
# import argparse
import itertools
import json
import time
from slam_llm.models.slam_model import slam_model





# config
# from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
# from llama_recipes.configs import model_config as MODEL_CONFIG
# from llama_recipes.configs import log_config as LOG_CONFIG
from slam_llm.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    get_policies
)
from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
import os
import logging
from tqdm import tqdm
from model.slam_model_st import model_factory
from model.slm_model import CustomSLM
from transformers import  AutoTokenizer,AutoConfig,AutoModel

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf



class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def Inference(kwargs: DictConfig):

	# Update the configuration for the training and sharding process
	train_config, fsdp_config, model_config, log_config, dataset_config,ckpt_path = kwargs.train_config, \
	                                                                      kwargs.fsdp_config, \
	                                                                      kwargs.model_config, \
	                                                                      kwargs.log_config, \
	                                                                      kwargs.dataset_config, \
                                                                          kwargs.ckpt_path 

	OmegaConf.set_struct(kwargs,False)
	del kwargs["train_config"]
	del kwargs["fsdp_config"]
	del kwargs["model_config"]
	del kwargs["log_config"]
	del kwargs["dataset_config"]
	OmegaConf.set_struct(kwargs,True)



	# Set the seeds for reproducibility
	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)




	if train_config.enable_fsdp or train_config.enable_ddp:
		setup()
		local_rank = int(os.environ["LOCAL_RANK"])
		rank = int(os.environ["RANK"])
		world_size = int(os.environ["WORLD_SIZE"])


	if torch.distributed.is_initialized():
		torch.cuda.set_device(local_rank)
		clear_gpu_cache(local_rank)
		setup_environ_flags(rank)

	config = AutoConfig.from_pretrained("Qwen/Qwen2-7B")  # 加载 Qwen2-7B 的配置
	tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
	model = CustomSLM(config,ckpt_path=ckpt_path)     
	# model = AutoModel.from_pretrained("/home/yxdu/hit/SLAM-LLM/examples/st_covost2/output/step_10/test") 
			

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.
	model.to(torch.bfloat16)
	dataset_config["bf16"]=True
	model.to(device)
	model.eval()
	tokenizer.padding_side = 'right'




	dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

	test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
			sampler=InferenceSampler(len(dataset_test)),
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
			shuffle=False,
            batch_size=train_config.val_batch_size,
			drop_last=False,
			prefetch_factor=1000,
            persistent_workers=True,
			collate_fn=dataset_test.collator
        )
	

	gts = []
	sources = []
	rets = []

	source = dataset_config.get("source", None)
	
	for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
		for key in batch.keys():
			batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
		
		model_outputs = model.generate(**batch)
		output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)

		for key, text, target in zip(batch["keys"], output_text, batch["targets"]):	
			print("Prediction:  ",key,text)
			print("Ground Truth:",key,target)

			rets.append(text)
			gts.append(target)
			sources.append(source)
			
	torch.distributed.barrier()


	
	merged_gts = [None for _ in range(world_size)]
	merged_sources = [None for _ in range(world_size)]
	merged_responses = [None for _ in range(world_size)]
	torch.distributed.all_gather_object(merged_gts, gts)
	torch.distributed.all_gather_object(merged_sources, sources)
	torch.distributed.all_gather_object(merged_responses, rets)

	merged_gts = [_ for _ in itertools.chain.from_iterable(merged_gts)]
	merged_sources = [_ for _ in itertools.chain.from_iterable(merged_sources)]
	merged_responses = [_ for _ in itertools.chain.from_iterable(merged_responses)]

	if torch.distributed.get_rank() == 0:

		results_file = log_config.decode_log
		with open(results_file, 'w') as f:
			for gt, response, source in zip(merged_gts, merged_responses, merged_sources):
				result = {
					'gt': gt,
					'response': response,
					'source': source,
				}
				f.write(json.dumps(result,ensure_ascii=False) + '\n')

	torch.distributed.barrier()


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: str = field(
        default="output/model.pt", metadata={"help": "The path to projector checkpoint"}
    )
    peft_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to peft checkpoint, should be a directory including adapter_config.json"
        },
    )


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    # kwargs = to_plain_list(cfg)
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if cfg.get("debug", False):
        import pdb

        pdb.set_trace()

    Inference(cfg)


if __name__ == "__main__":
    main_hydra()
