# import fire
import random
import torch
import logging

from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from utils.snac_utils import reconscruct_snac, reconstruct_tensors
import numpy as np
import os
import logging
import soundfile as sf
import time

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

	train_config, fsdp_config, model_config, log_config, dataset_config, decode_config = kwargs.train_config, \
	                                                                      				kwargs.fsdp_config, \
																						kwargs.model_config, \
																						kwargs.log_config, \
																						kwargs.dataset_config, \
																						kwargs.decode_config

	OmegaConf.set_struct(kwargs,False)
	del kwargs["train_config"]
	del kwargs["fsdp_config"]
	del kwargs["model_config"]
	del kwargs["log_config"]
	del kwargs["dataset_config"]
	del kwargs["decode_config"]
	OmegaConf.set_struct(kwargs,True)

	# Set log
	if not os.path.exists(os.path.dirname(log_config.log_file)):
		os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
	logging.basicConfig(
		level=logging.INFO, 
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		filemode='w'
	)

	logger = logging.getLogger()  
	logger.setLevel(logging.INFO)

	file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
	file_handler.setLevel(logging.INFO)
	file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	file_handler.setFormatter(file_formatter)

	logger.handlers[0].setLevel(logging.INFO)
	console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	logger.handlers[0].setFormatter(console_formatter) 

	logger.addHandler(file_handler)
    
	logger.info("train_config: {}".format(train_config))
	logger.info("fsdp_config: {}".format(fsdp_config))
	logger.info("model_config: {}".format(model_config))

	
	# Set the seeds for reproducibility
	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	
	model_factory = get_custom_model_factory(model_config, logger)
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	codec_decoder = model.codec_decoder
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.
	model.to(device)
	model.eval()

	# dataset_config = generate_dataset_config(train_config, kwargs)
	logger.info("dataset_config: {}".format(dataset_config))
	dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
	if not (train_config.enable_fsdp or train_config.enable_ddp):
		logger.info(f"--> Training Set Length = {len(dataset_test)}")

	test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
			shuffle=False,
            batch_size=train_config.val_batch_size,
			drop_last=False,
			collate_fn=dataset_test.collator
        )

	task_type = decode_config.task_type
	logger.info("decode_config: {}".format(decode_config))	
	if decode_config.do_sample:
		logger.info("Decode Strategy: Sampling")
	else:
		logger.info("Decode Strategy: Greedy")
	if decode_config.decode_text_only:
		logger.info("Decode Text Only")
	else:
		logger.info("Decode Text & Audio")
	logger.info("============== Start {task_type} Inference (Streaming Version) ==============".format(task_type=task_type))

	decode_log_dir = kwargs.get('decode_log')
	output_text_only = kwargs.get('output_text_only', False)

	if not os.path.exists(decode_log_dir):
		os.makedirs(decode_log_dir)

	pred_path = os.path.join(decode_log_dir, "pred_text_stream")
	gt_path = os.path.join(decode_log_dir, "gt_text")
	question_path = os.path.join(decode_log_dir, "question_text")
	generate_audio_dir = os.path.join(decode_log_dir, "pred_audio_stream")

	if not os.path.exists(generate_audio_dir) and not (output_text_only or decode_config.decode_text_only):
		os.makedirs(generate_audio_dir)

	with open(pred_path, "w") as pred, open(gt_path, "w") as gt, open(question_path, "w") as q:
		for step, batch in enumerate(test_dataloader):
			for key in batch.keys():
				batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
			
			audio_text_generator = model.stream_generate(**batch, **decode_config)
			output_text = ""

			if output_text_only or decode_config.decode_text_only:
				for result in audio_text_generator:
					text_tokens = result.get('text_stream')
					if text_tokens is not None:
						output_text += model.tokenizer.decode(torch.tensor(text_tokens))
				for key, source_text, target_text, generated_text in zip(batch["keys"], batch["source_texts"], batch["target_texts"], [output_text]):
					q.write(key + "\t" + source_text + "\n")
					gt.write(key + "\t" + target_text + "\n")
					pred.write(key + "\t" + generated_text + "\n")

				if task_type == "s2s":
					logger.info(f"Question: {source_text}")
				elif task_type == "tts":
					logger.info(f"Target Text: {target_text}")

				logger.info(f"Generated Text: {output_text}")
				continue			
			else:
				key = batch["keys"][0]
				audio_key = key[:-4] if key[-4:] == ".wav" else key
				start_time = time.time()
				first_chunk_time = None

				with sf.SoundFile(f"{generate_audio_dir}/{audio_key}.wav", mode='w', samplerate=24000, channels=1, subtype='PCM_16') as f:
					for result in audio_text_generator:
						if first_chunk_time is None:
							first_chunk_time = time.time()
							delay = first_chunk_time - start_time

						text_tokens = result.get('text_stream')
						if text_tokens is not None:
							output_text += model.tokenizer.decode(torch.tensor(text_tokens))

						audio_bytes = result.get('audio_stream')
						if audio_bytes is not None:
							audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
							f.write(audio_chunk)
				
				for key, source_text, target_text, generated_text in zip(batch["keys"], batch["source_texts"], batch["target_texts"], [output_text]):
					q.write(key + "\t" + source_text + "\n")
					gt.write(key + "\t" + target_text + "\n")
					pred.write(key + "\t" + generated_text + "\n")

				if task_type == "s2s":
					logger.info(f"Question: {source_text}")
				elif task_type == "tts":
					logger.info(f"Target Text: {target_text}")
				
				logger.info(f"Generated Text: {output_text}")
				logger.info(f"Generated Audio: {audio_key}.wav")
				logger.info(f"First Chunk Delay: {delay:.2f} seconds")

	logger.info("============== Inference (Streaming Version) Finished ==============")

if __name__ == "__main__":
	main_hydra()