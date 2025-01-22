import os
import json
import torch
import soundfile as sf
import logging
import random
from omegaconf import OmegaConf, DictConfig
from slam_llm.utils.model_utils import get_custom_model_factory
from utils.snac_utils import layershift, simple_shift
from whisper import DecodingOptions, decode
from tqdm import tqdm
from generate.generate_s2s_online_multi_round import generate_from_wav, generate_from_text

def main(kwargs: DictConfig):
	train_config, fsdp_config, model_config, log_config, dataset_config, decode_config = kwargs.train_config, \
																				kwargs.fsdp_config, \
																				kwargs.model_config, \
																				kwargs.log_config, \
																				kwargs.dataset_config, \
																				kwargs.decode_config

	OmegaConf.set_struct(kwargs, False)
	del kwargs["train_config"]
	del kwargs["fsdp_config"]
	del kwargs["model_config"]
	del kwargs["log_config"]
	del kwargs["dataset_config"]
	del kwargs["decode_config"]
	OmegaConf.set_struct(kwargs, True)

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

	torch.cuda.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)

	model_factory = get_custom_model_factory(model_config, logger)
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	task_type = decode_config.task_type
	code_layer = model_config.vocab_config.code_layer
	code_type = model_config.code_type
	do_layershift = dataset_config.do_layershift

	output_text_only = kwargs.get('output_text_only', False)
	speech_sample_rate = kwargs.get('speech_sample_rate', 24000)
	audio_prompt_path = kwargs.get('audio_prompt_path', None)
	jsonl_path = kwargs.get('batch_input_jsonl', None)

	output_dir = log_config.online_output_dir
	logger.info("output_dir: {}".format(output_dir))

	if audio_prompt_path is None or not os.path.exists(audio_prompt_path):
		tone_dir = "default_tone"
	else:
		tone_dir = os.path.basename(audio_prompt_path).split('.')[0]
	tone_audio_dir = os.path.join(output_dir, tone_dir)

	if not os.path.exists(tone_audio_dir) and not (decode_config.decode_text_only or output_text_only):
		os.makedirs(tone_audio_dir)

	layer_shift = layershift if do_layershift else simple_shift

	task_type = decode_config.task_type
	logger.info("decode_config: {}".format(decode_config))

	if decode_config.do_sample:
		logger.info("Decode Strategy: Sampling")
	else:
		logger.info("Decode Strategy: Greedy")

	if decode_config.input_text:
		logger.info("Input Text")
	else:
		logger.info("Input Audio")

	if decode_config.decode_text_only:
		logger.info("Decode Text Only")
	else:
		logger.info("Decode Text & Audio")
		
	logger.info("Decode Code Type: {}".format(code_type))
	logger.info("Decode Code Layer: {}".format(code_layer))
	logger.info("Tone for Audio Generation: {}".format(tone_dir))

	output_jsonl_path = os.path.join(output_dir, "output_with_text.jsonl")

	with open(jsonl_path, 'r', encoding='utf-8') as f, open(output_jsonl_path, 'w', encoding='utf-8') as out_f:
		lines = f.readlines()

		for line in tqdm(lines, desc="Batch Inference"):
			data = json.loads(line.strip())
			conv_id = data["id"]
			num_round = data["num_round"]
			dialogue = data["dialogue"]

			conversation_dir = os.path.join(tone_audio_dir, f"{conv_id}")
			os.makedirs(conversation_dir, exist_ok=True)

			history = ""
			for round_item in dialogue:
				source_wav = round_item.get("source_wav", None)
				source_text = round_item.get("source_text", "")

				if source_wav and os.path.exists(source_wav) and not decode_config.input_text:
					output_wav, output_text, history_assistant, transcribed_text = generate_from_wav(
						source_wav, model, dataset_config, decode_config, logger, device, model_config, 
						tone_dir, audio_prompt_path, output_text_only, history, layer_shift
					)

					history_user = "USER: " + transcribed_text.strip() + " "
					history = history + history_user + history_assistant.strip() + " "

					logger.info(f"[ID {conv_id}, Round {round_item['round']}] Transcribed Text: {transcribed_text}")
					
					if output_wav is not None and not (decode_config.decode_text_only or output_text_only):
						round_idx = round_item["round"]
						output_wav_path = os.path.join(conversation_dir, f"chat_{round_idx}.wav")
						sf.write(output_wav_path, output_wav.squeeze().cpu().numpy(), speech_sample_rate)
						logger.info(f"[ID {conv_id}, Round {round_idx}] Generated Audio saved at: {output_wav_path}")

					logger.info(f"[ID {conv_id}, Round {round_item['round']}] Generated Text: {output_text}")
					round_item["output_text"] = output_text

				else:
					audio_hat, output_text, history = generate_from_text(
						source_text, model, dataset_config, decode_config, logger, device, model_config,
						tone_dir, audio_prompt_path, output_text_only, history, layer_shift
					)

					if audio_hat is not None and not (decode_config.decode_text_only or output_text_only):
						round_idx = round_item["round"]
						output_wav_path = os.path.join(conversation_dir, f"chat_{round_idx}.wav")
						sf.write(output_wav_path, audio_hat.squeeze().cpu().numpy(), speech_sample_rate)
						logger.info(f"[ID {conv_id}, Round {round_idx}] Generated Audio saved at: {output_wav_path}")

					logger.info(f"[ID {conv_id}, Round {round_item['round']}] Generated Text: {output_text}")
					round_item["output_text"] = output_text

			out_f.write(json.dumps({
				"id": conv_id,
				"num_round": num_round,
				"dialogue": dialogue
			}, ensure_ascii=False) + "\n")

	logger.info("============== Batch Inference Finished ==============")
	logger.info(f"Updated dialogues with output_text have been saved to: {output_jsonl_path}")

