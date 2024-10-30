import random
import torch
import logging
import os
import soundfile as sf
from slam_llm.utils.model_utils import get_custom_model_factory
from utils.snac_utils import reconscruct_snac, reconstruct_tensors, layershift, get_snac_answer_token
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import whisper
import numpy as np
import time


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


def extract_audio_feature(audio_path, mel_size):
	audio_raw = whisper.load_audio(audio_path)
	audio_raw = whisper.pad_or_trim(audio_raw)
	audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size).permute(1, 0)
	audio_length = (audio_mel.shape[0] + 1) // 2
	audio_length = audio_length // 5
	audio_res = audio_mel
	 
	return audio_res, audio_length


def get_input_ids(length, special_token_a, special_token_t, vocab_config):
	input_ids = []
	for i in range(vocab_config.code_layer):
		input_ids_item = []
		input_ids_item.append(layershift(vocab_config.input_a, i))
		input_ids_item += [layershift(vocab_config.pad_a, i)] * length
		input_ids_item += [(layershift(vocab_config.eoa, i)), layershift(special_token_a, i)]
		input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
	input_id_T = torch.tensor([vocab_config.input_t] + [vocab_config.pad_t] * length + [vocab_config.eot, special_token_t])
	input_ids.append(input_id_T.unsqueeze(0))
	return input_ids

def get_padded_input(text_input_idx, text_index_length, code_layer, _pad_a):
	padded_input = []
	for i in range(code_layer):
		padded_input_item = [layershift(_pad_a, i)] * text_index_length
		padded_input.append(torch.tensor(padded_input_item).unsqueeze(0))
	
	final_layer_input = torch.tensor(text_input_idx)
	padded_input.append(final_layer_input.unsqueeze(0))
	return padded_input


def generate_from_wav_stream(wav_path, model, codec_decoder, dataset_config, decode_config, logger, device):
	mel_size = dataset_config.mel_size
	prompt = dataset_config.prompt
	prompt_template = "USER: {}\n ASSISTANT: "
	vocab_config = dataset_config.vocab_config
	special_token_a = vocab_config.answer_a
	special_token_t = vocab_config.answer_t
	_input_t = vocab_config.input_t
	_eot = vocab_config.eot
	code_layer = vocab_config.code_layer
	task_type = dataset_config.task_type

	audio_mel, audio_length = extract_audio_feature(wav_path, mel_size)

	prompt = prompt_template.format(prompt)
	prompt_ids = model.tokenizer.encode(prompt)
	prompt_ids = [_input_t] + prompt_ids + [_eot]
	prompt_length = len(prompt_ids)
	prompt_ids = get_padded_input(prompt_ids, prompt_length, code_layer, vocab_config.pad_a)

	example_ids = get_input_ids(audio_length, special_token_a, special_token_t, vocab_config)
	example_ids = [torch.cat((prompt_ids[i], example_ids[i]), dim = 1) for i in range(code_layer + 1)]

	input_length = audio_length
	example_mask = example_ids[0][0].ge(-1)
	example_ids = torch.stack(example_ids).squeeze()

	input_ids = example_ids.unsqueeze(0).to(device)
	attention_mask = example_mask.unsqueeze(0).to(device)
	audio_mel = audio_mel.unsqueeze(0).to(device)
	input_length = torch.tensor([input_length]).to(device)
	audio_length = torch.tensor([audio_length]).to(device)
	task_type = [task_type]

	modality_mask = torch.zeros_like(attention_mask)
	padding_left = prompt_length + 1 # +1 for <bos>
	modality_mask[0, padding_left:padding_left+audio_length] = True

	batch = {
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"audio_mel": audio_mel,
		"input_length": input_length,
		"audio_length": audio_length,
		"modality_mask": modality_mask,
		"task_types": task_type,
	}

	audio_text_generator = model.stream_generate(**batch, **decode_config)
	  
	return audio_text_generator


def generate_from_text_stream(text_input, model, codec_decoder, dataset_config, decode_config, logger, device):
	prompt = dataset_config.prompt
	prompt_template = "USER: {}\n ASSISTANT: "
	vocab_config = dataset_config.vocab_config
	special_token_a = vocab_config.answer_a
	special_token_t = vocab_config.answer_t
	_input_t = vocab_config.input_t
	_eot = vocab_config.eot
	code_layer = vocab_config.code_layer
	task_type = dataset_config.task_type

	prompt = prompt_template.format(prompt)
	prompt_ids = model.tokenizer.encode(prompt)
	prompt_ids = [_input_t] + prompt_ids + [_eot]
	prompt_length = len(prompt_ids)
	prompt_ids = get_padded_input(prompt_ids, prompt_length, code_layer, vocab_config.pad_a)

	text_input_ids = model.tokenizer.encode(text_input)
	text_input_length = len(text_input_ids)
	text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
	example_ids = get_input_ids(text_input_length, special_token_a, special_token_t, vocab_config)
	text_layer = example_ids[code_layer]
	text_layer = torch.cat((text_layer[:,:1], text_input_ids.unsqueeze(0), text_layer[:,-2:]), dim=1)
	example_ids[code_layer] = text_layer
	example_ids = [torch.cat((prompt_ids[i], example_ids[i]), dim = 1) for i in range(code_layer + 1)]

	input_length = text_input_length
	example_mask = example_ids[0][0].ge(-1)
	example_ids = torch.stack(example_ids).squeeze()

	input_ids = example_ids.unsqueeze(0).to(device)
	attention_mask = example_mask.unsqueeze(0).to(device)
	input_length = torch.tensor([input_length]).to(device)
	task_type = [task_type]

	modality_mask = torch.zeros_like(attention_mask)

	batch = {
		"input_ids": input_ids,
		"attention_mask": attention_mask,
		"audio_mel": None,
		"input_length": input_length,
		"audio_length": None,
		"modality_mask": modality_mask,
		"task_types": task_type,
	}

	audio_text_generator = model.stream_generate(**batch, **decode_config)
	  
	return audio_text_generator


def save_streamed_audio(generator, output_wav_path, model, logger, sample_rate=24000):
	generated_text = ""
	start_time = time.time()
	first_chunk_time = None

	with sf.SoundFile(output_wav_path, mode='w', samplerate=sample_rate, channels=1, subtype='PCM_16') as f:
		for result in generator:
			if first_chunk_time is None:
				first_chunk_time = time.time()
				delay = first_chunk_time - start_time
				
			text_tokens = result.get('text_stream')
			if text_tokens is not None:
				generated_text += model.tokenizer.decode(torch.tensor(text_tokens))

			audio_bytes = result.get('audio_stream')
			if audio_bytes is not None:
				audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
				f.write(audio_chunk)
				
	logger.info(f"First Chunk Delay: {delay:.2f} seconds")
	logger.info(f"Final Generated Text: {generated_text}")


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
	codec_decoder = model.codec_decoder
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	output_dir = log_config.online_output_dir
	logger.info("output_dir: {}".format(output_dir))

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


	if decode_config.input_text:
		logger.info("============== Ready for t2s Online Inference (Streaming Version) ==============")
		while True:
			text_input = input("Please provide the text input (or type 'q' to quit): ")
			if text_input.lower() == 'q':
				break
			
			if output_dir is not None:
				os.makedirs(output_dir, exist_ok=True)
				output_wav_path = os.path.join(output_dir, f"generated_{text_input.replace(' ', '_')[:20]}.wav")
			else:
				output_wav_path = f"generated_{text_input.replace(' ', '_')}.wav"

			audio_generator = generate_from_text_stream(
                text_input, model, codec_decoder, dataset_config, decode_config, logger, device
            )

			save_streamed_audio(audio_generator, output_wav_path, model, logger)
			logger.info(f"Generated Audio saved at: {output_wav_path}")
	
	else:
		logger.info("============== Ready for {task_type} Online Inference (Streaming Version) ==============".format(task_type=task_type))
		while True:
			wav_path = input("Please provide the path to a WAV file (or type 'q' to quit): ")
			if wav_path.lower() == 'q':
				break

			if not os.path.exists(wav_path):
				logger.warning(f"File {wav_path} does not exist. Please try again.")
				continue
			
			if output_dir is not None:
				os.makedirs(output_dir, exist_ok=True)
				output_wav_path = os.path.join(output_dir, f"generated_{os.path.basename(wav_path)}")
			else:
				output_wav_path = f"generated_{os.path.basename(wav_path)}"

			if not output_wav_path.lower().endswith('.wav'):
				output_wav_path = os.path.splitext(output_wav_path)[0] + '.wav'

			audio_generator = generate_from_wav_stream(
				wav_path, model, codec_decoder, dataset_config, decode_config, logger, device
			)
			
			save_streamed_audio(audio_generator, output_wav_path, model, logger)
			logger.info(f"Generated Audio saved at: {output_wav_path}")
	
	logger.info("============== Online Inference (Streaming Version) Finished ==============")

if __name__ == "__main__":
	main_hydra()
