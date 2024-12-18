import random
import torch
import logging
import os
import soundfile as sf
from slam_llm.utils.model_utils import get_custom_model_factory
from utils.snac_utils import reconscruct_snac, reconstruct_tensors, layershift, simple_shift
from utils.codec_utils import audio_decode_cosyvoice
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
import whisper


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


def get_input_ids(length, special_token_a, special_token_t, vocab_config, layer_shift=layershift):
	input_ids = []
	for i in range(vocab_config.code_layer):
		input_ids_item = []
		input_ids_item.append(layer_shift(vocab_config.input_a, i))
		input_ids_item += [layer_shift(vocab_config.pad_a, i)] * length
		input_ids_item += [(layer_shift(vocab_config.eoa, i)), layer_shift(special_token_a, i)]
		input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
	input_id_T = torch.tensor([vocab_config.input_t] + [vocab_config.pad_t] * length + [vocab_config.eot, special_token_t])
	input_ids.append(input_id_T.unsqueeze(0))
	return input_ids

def get_padded_input(text_input_idx, text_index_length, code_layer, _pad_a, layer_shift=layershift):
	padded_input = []
	for i in range(code_layer):
		padded_input_item = [layer_shift(_pad_a, i)] * text_index_length
		padded_input.append(torch.tensor(padded_input_item).unsqueeze(0))
	
	final_layer_input = torch.tensor(text_input_idx)
	padded_input.append(final_layer_input.unsqueeze(0))
	return padded_input


def generate_from_wav(wav_path, model, codec_decoder, dataset_config, decode_config, logger, device, model_config, tone_dir, audio_prompt_path=None, output_text_only=False, layer_shift=layershift, inference_streaming=False):
	mel_size = dataset_config.mel_size
	prompt = dataset_config.prompt
	prompt_template = "<SYSTEM>: {}\n "
	vocab_config = dataset_config.vocab_config
	special_token_a = vocab_config.answer_a
	special_token_t = vocab_config.answer_t
	_input_t = vocab_config.input_t
	_eot = vocab_config.eot
	code_layer = vocab_config.code_layer
	task_type = dataset_config.task_type
	code_type = model_config.code_type
	num_latency_tokens = dataset_config.num_latency_tokens

	audio_mel, audio_length = extract_audio_feature(wav_path, mel_size)

	prompt = prompt_template.format(prompt)
	prompt_ids = model.tokenizer.encode(prompt)
	prompt_ids = [_input_t] + prompt_ids + [_eot]
	prompt_length = len(prompt_ids)
	prompt_ids = get_padded_input(prompt_ids, prompt_length, code_layer, vocab_config.pad_a, layer_shift)

	example_ids = get_input_ids(audio_length, special_token_a, special_token_t, vocab_config, layer_shift)
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
		"mini_omni_modeling": False,
	}

	if inference_streaming:
		return model.stream_generate(**batch, **decode_config)

	model_outputs = model.generate(**batch, **decode_config)
	text_outputs = model_outputs[code_layer]
	audio_outputs = model_outputs[:code_layer]	
	output_text = model.tokenizer.decode(text_outputs, add_special_tokens=False, skip_special_tokens=True)
	
	if decode_config.decode_text_only or output_text_only:
		return None, output_text

	if audio_outputs[0].shape[0] == decode_config.max_new_tokens:
		logger.warning(f"Audio token is too long, skip. You can try to increase the max_new_tokens in the decode_config.")
		return None, output_text
	
	audio_tokens = [audio_outputs[layer] for layer in range(code_layer)]

	if code_type == "SNAC":
		audiolist = reconscruct_snac(audio_tokens)
		audio = reconstruct_tensors(audiolist)
		with torch.inference_mode():
			audio_hat = codec_decoder.decode(audio)
	elif code_type == "CosyVoice":
		audio_hat = audio_decode_cosyvoice(audio_tokens, model_config, codec_decoder, tone_dir, audio_prompt_path, code_layer, num_latency_tokens, speed=1.0)
	else:
		raise NotImplementedError

	return audio_hat, output_text


def generate_from_text(text_input, model, codec_decoder, dataset_config, decode_config, logger, device, model_config, tone_dir, audio_prompt_path=None, output_text_only=False, layer_shift=layershift, inference_streaming=False):
	prompt = dataset_config.prompt
	prompt_template = "<SYSTEM>: {}\n "
	vocab_config = dataset_config.vocab_config
	special_token_a = vocab_config.answer_a
	special_token_t = vocab_config.answer_t
	_input_t = vocab_config.input_t
	_eot = vocab_config.eot
	code_layer = vocab_config.code_layer
	task_type = dataset_config.task_type
	code_type = model_config.code_type
	num_latency_tokens = dataset_config.num_latency_tokens

	prompt = prompt_template.format(prompt)
	prompt_ids = model.tokenizer.encode(prompt)
	prompt_ids = [_input_t] + prompt_ids + [_eot]
	prompt_length = len(prompt_ids)
	prompt_ids = get_padded_input(prompt_ids, prompt_length, code_layer, vocab_config.pad_a, layer_shift)

	text_input_ids = model.tokenizer.encode(text_input)
	text_input_length = len(text_input_ids)
	text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
	example_ids = get_input_ids(text_input_length, special_token_a, special_token_t, vocab_config, layer_shift)
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
		"mini_omni_modeling": False,
	}

	if inference_streaming:
		return model.stream_generate(**batch, **decode_config)

	model_outputs = model.generate(**batch, **decode_config)
	text_outputs = model_outputs[code_layer]
	audio_outputs = model_outputs[:code_layer]	
	output_text = model.tokenizer.decode(text_outputs, add_special_tokens=False, skip_special_tokens=True)
	
	if decode_config.decode_text_only or output_text_only:
		return None, output_text

	if audio_outputs[0].shape[0] == decode_config.max_new_tokens:
		logger.warning(f"Audio token is too long, skip. You can try to increase the max_new_tokens in the decode_config.")
		return None, output_text
	
	audio_tokens = [audio_outputs[layer] for layer in range(code_layer)]

	if code_type == "SNAC":
		audiolist = reconscruct_snac(audio_tokens)
		audio = reconstruct_tensors(audiolist)
		with torch.inference_mode():
			audio_hat = codec_decoder.decode(audio)
	elif code_type == "CosyVoice":
		audio_hat = audio_decode_cosyvoice(audio_tokens, model_config, codec_decoder, tone_dir, audio_prompt_path, code_layer, num_latency_tokens, speed=1.0)
	else:
		raise NotImplementedError
	
	return audio_hat, output_text


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

	task_type = decode_config.task_type
	code_layer = model_config.vocab_config.code_layer
	code_type = model_config.code_type
	do_layershift = dataset_config.do_layershift

	output_text_only = kwargs.get('output_text_only', False)
	speech_sample_rate = kwargs.get('speech_sample_rate', 24000)
	audio_prompt_path = kwargs.get('audio_prompt_path', None)
	inference_streaming = kwargs.get('inference_streaming', False)

	output_dir = log_config.online_output_dir
	logger.info("output_dir: {}".format(output_dir))

	if audio_prompt_path is None or not os.path.exists(audio_prompt_path):
		tone_dir = "default_tone"
	else:
		tone_dir = os.path.basename(audio_prompt_path).split('.')[0]
	tone_audio_dir = os.path.join(output_dir, tone_dir)

	if not os.path.exists(tone_audio_dir) and not (output_text_only or decode_config.decode_text_only):
		os.makedirs(tone_audio_dir)

	layer_shift = None
	if do_layershift:
		layer_shift = layershift
	else:
		layer_shift = simple_shift

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

	if not inference_streaming:
		if decode_config.input_text:
			logger.info("============== Ready for t2s Online Inference (Non-Streaming) ==============")
			while True:
				text_input = input("Please provide the text input (or type 'q' to quit): ")
				if text_input.lower() == 'q':
					break

				audio_hat, output_text = generate_from_text(text_input, model, codec_decoder, dataset_config, decode_config, logger, device, model_config, tone_dir, audio_prompt_path, output_text_only, layer_shift, inference_streaming)
				logger.info(f"Generated Text: {output_text}")

				if tone_audio_dir is not None:
					os.makedirs(tone_audio_dir, exist_ok=True)
					output_wav_path = os.path.join(tone_audio_dir, f"generated_{text_input.replace(' ', '_')[:20]}.wav")
				else:
					output_wav_path = f"generated_{text_input.replace(' ', '_')}.wav"
				sf.write(output_wav_path, audio_hat.squeeze().cpu().numpy(), speech_sample_rate)
				logger.info(f"Generated Audio saved at: {output_wav_path}")
		else:
			logger.info("============== Ready for s2s Online Inference (Non-Streaming) ==============")
			while True:
				wav_path = input("Please provide the path to a WAV file (or type 'q' to quit): ")
				if wav_path.lower() == 'q':
					break

				if not os.path.exists(wav_path):
					logger.warning(f"File {wav_path} does not exist. Please try again.")
					continue

				output_wav, output_text = generate_from_wav(wav_path, model, codec_decoder, dataset_config, decode_config, logger, device, model_config, tone_dir, audio_prompt_path, output_text_only, layer_shift, inference_streaming)
				logger.info(f"Generated Text: {output_text}")

				if output_wav is None:
					if not output_text_only or not decode_config.decode_text_only:
						logger.warning(f"Generated Audio is None. Please try again.")
					continue
				
				if tone_audio_dir is not None:
					os.makedirs(tone_audio_dir, exist_ok=True)
					output_wav_path = os.path.join(tone_audio_dir, f"generated_{os.path.basename(wav_path)}")
				else:
					output_wav_path = f"generated_{os.path.basename(wav_path)}"

				if not output_wav_path.lower().endswith('.wav'):
					output_wav_path = os.path.splitext(output_wav_path)[0] + '.wav'

				sf.write(output_wav_path, output_wav.squeeze().cpu().numpy(), speech_sample_rate)		
				logger.info(f"Generated Audio saved at: {output_wav_path}")
	else:
		if decode_config.input_text:
			logger.info("============== Ready for t2s Online Inference (Streaming) ==============")
			while True:
				text_input = input("Please provide the text input (or type 'q' to quit): ")
				if text_input.lower() == 'q':
					break

				audio_hat, output_text = generate_from_text(text_input, model, codec_decoder, dataset_config, decode_config, logger, device, model_config, tone_dir, audio_prompt_path, output_text_only, layer_shift, inference_streaming)
				logger.info(f"Generated Text: {output_text}")

				if tone_audio_dir is not None:
					os.makedirs(tone_audio_dir, exist_ok=True)
					output_wav_path = os.path.join(tone_audio_dir, f"generated_{text_input.replace(' ', '_')[:20]}.wav")
				else:
					output_wav_path = f"generated_{text_input.replace(' ', '_')}.wav"
				sf.write(output_wav_path, audio_hat.squeeze().cpu().numpy(), speech_sample_rate)
				logger.info(f"Generated Audio saved at: {output_wav_path}")
		else:
			logger.info("============== Ready for s2s Online Inference (Streaming) ==============")
			while True:
				wav_path = input("Please provide the path to a WAV file (or type 'q' to quit): ")
				if wav_path.lower() == 'q':
					break

				if not os.path.exists(wav_path):
					logger.warning(f"File {wav_path} does not exist. Please try again.")
					continue

				# output_wav, output_text = generate_from_wav(wav_path, model, codec_decoder, dataset_config, decode_config, logger, device, model_config, tone_dir, audio_prompt_path, output_text_only, layer_shift, inference_streaming)

				output_generator = generate_from_wav(wav_path, model, codec_decoder, dataset_config, decode_config, logger, device, model_config, tone_dir, audio_prompt_path, output_text_only, layer_shift, inference_streaming)
					
				audio_tokens_list = []
				text_tokens_list = []

				for result in output_generator:
					audio_tokens_list.extend(result["audio_tokens"])
					text_tokens_list.append(result["text_token"])

				output_text = model.tokenizer.decode(text_tokens_list, add_special_tokens=False, skip_special_tokens=True)
				logger.info(f"Generated Text: {output_text}")

				if output_wav is None:
					logger.warning(f"Generated Audio is None. Please try again.")
					continue
				
				if tone_audio_dir is not None:
					os.makedirs(tone_audio_dir, exist_ok=True)
					output_wav_path = os.path.join(tone_audio_dir, f"generated_{os.path.basename(wav_path)}")
				else:
					output_wav_path = f"generated_{os.path.basename(wav_path)}"

				if not output_wav_path.lower().endswith('.wav'):
					output_wav_path = os.path.splitext(output_wav_path)[0] + '.wav'

				sf.write(output_wav_path, output_wav.squeeze().cpu().numpy(), speech_sample_rate)		
				logger.info(f"Generated Audio saved at: {output_wav_path}")
		
	logger.info("============== Online Inference Finished ==============")

if __name__ == "__main__":
	main_hydra()
