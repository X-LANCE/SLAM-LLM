import random
import torch
import logging
import os
import soundfile as sf
from slam_llm.utils.model_utils import get_custom_model_factory
from utils.snac_utils import reconscruct_snac, reconstruct_tensors
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


def generate_from_wav(wav_path, model, tokenizer, codec_decoder, decode_config, logger, device):
    audio_raw, _ = sf.read(wav_path)
    
    audio_input = torch.tensor(audio_raw).to(device).unsqueeze(0)  # 添加batch维度

    batch = {
        "audio": audio_input,
    }

    model_outputs = model.generate(**batch, **decode_config)
    text_outputs = model_outputs[7]
    audio_outputs = model_outputs[:7]

    output_text = tokenizer.decode(text_outputs, add_special_tokens=False, skip_special_tokens=True)
    logger.info(f"Generated Text: {output_text}")

    if decode_config.decode_text_only:
        return output_text

    audio_tokens = [audio_outputs[layer] for layer in range(7)]
    audiolist = reconscruct_snac(audio_tokens)
    audio = reconstruct_tensors(audiolist)
    with torch.inference_mode():
        audio_hat = codec_decoder.decode(audio)

    output_wav_path = f"generated_{os.path.basename(wav_path)}"
    sf.write(output_wav_path, audio_hat.squeeze().cpu().numpy(), 24000)
    logger.info(f"Generated Audio saved at: {output_wav_path}")
    
    return output_wav_path


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
    logger.info("decode_config: {}".format(decode_config))    
    if decode_config.do_sample:
        logger.info("Decode Strategy: Sampling")
    else:
        logger.info("Decode Strategy: Greedy")
    if decode_config.decode_text_only:
        logger.info("Decode Text Only")
    else:
        logger.info("Decode Text & Audio")

    logger.info("============== Ready for {task_type} Online Inference ==============".format(task_type=task_type))

    while True:
        wav_path = input("Please provide the path to a WAV file (or type 'exit' to quit): ")
        if wav_path.lower() == 'exit':
            break

        if not os.path.exists(wav_path):
            print(f"File {wav_path} does not exist. Please try again.")
            continue

        output_wav = generate_from_wav(wav_path, model, tokenizer, codec_decoder, decode_config, logger, device)
        print(f"Generated WAV file saved at: {output_wav}")

if __name__ == "__main__":
    main_hydra()
