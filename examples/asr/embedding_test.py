# from slam_llm.pipeline.inference_batch import main as inference

import hydra
import logging
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional, List
from asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig
from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from slam_llm.models.slam_model import setup_encoder, setup_encoder_projector, setup_llm
from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.metric import compute_accuracy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
)

logger = logging.getLogger(__name__)


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


class slam_model(nn.Module):
    def __init__(self, tokenizer, train_config, model_config, **kwargs):
        super().__init__()
        # modality encoder
        self.encoder = setup_encoder(train_config, model_config, **kwargs)

        # llm
        self.llm = setup_llm(train_config, model_config, **kwargs)

        self.llm_embed = self.llm.model.embed_tokens

        # projector
        self.encoder_projector = setup_encoder_projector(
            train_config, model_config, **kwargs
        )

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get(
            "audio_mel_post_mask", None
        )  # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        vis_len = kwargs.get("vis_len", None)
        maskw2v = kwargs.get(
            "maskw2v", False
        )  # (FIX:MZY) False for supervised learning and inference

        # for text encoder
        instruct_ids = kwargs.get("instruct_ids", None)
        instruct_mask = kwargs.get("instruct_mask", None)

        modality_mask = kwargs.get("modality_mask", None)

        encoder_outs = None
        if audio_mel is not None or audio is not None:
            if self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(
                    audio_mel.permute(0, 2, 1)
                )  # bs*seq*dim
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(
                    audio_mel, audio_mel_mask
                )  # bs*seq*dim
            if self.model_config.encoder_name == "wavlm":
                encoder_outs = self.encoder.extract_features(
                    audio, 1 - audio_mask
                )  # (FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
            if self.model_config.encoder_name == "moco_wav2vec2":
                encoder_outs, inputLenBatch, audio_mel_post_mask = self.encoder(
                    (audio, audio_mask, visual, vis_len), maskw2v
                )  # bs*seq*dim
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        if instruct_ids is not None:
            if self.encoder is not None:
                encoder_outs = self.encoder(
                    input_ids=instruct_ids, attention_mask=instruct_mask
                ).last_hidden_state

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, instruct_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        batch_size, length, dims = encoder_outs.shape
        token_num, dims = self.llm_embed.weight.shape

        mat_mul_value = torch.zeros((batch_size, length, token_num), device=encoder_outs.device, dtype=encoder_outs.dtype)
        step = math.ceil(token_num/8)
        llm_embed_weight_T = self.llm_embed.weight.transpose(0, 1)
        for index in range(0, token_num, step):
            mat_mul_value[:,:,index: index+step] = torch.matmul(
                encoder_outs, llm_embed_weight_T[:, index:index+step] 
            )
        cosine_similarity = (
            mat_mul_value
            / torch.sqrt(torch.sum(torch.square(encoder_outs), dim=-1)).unsqueeze(-1)
            / torch.sqrt(torch.sum(torch.square(self.llm_embed.weight), dim=-1))
            .unsqueeze(0)
            .unsqueeze(1)
        )
        quantize_index = cosine_similarity.argmax(dim=-1)
        encoder_outs = self.llm_embed(quantize_index)
        # c_mask = c.max(-1)[0]<0.095
        # c_argmax = c.argmax(-1)
        # c_argmax[c_mask]=0
        # c_t_mask = model.tokenizer.batch_decode(
        #     c_argmax, add_special_tokens=False, skip_special_tokens=True
        # )
        # return (
        #     encoder_outs,
        #     torch.matmul(encoder_outs, self.llm_embed.weight.transpose(0, 1)),
        #     torch.matmul(encoder_outs, self.llm_embed.weight.transpose(0, 1)).argmax(dim=-1),
        # )

        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        if modality_mask is not None:
            batch_size, token_num, dims = inputs_embeds.shape
            _, l, _ = encoder_outs.shape
            encoder_outs_pad = F.pad(
                encoder_outs, (0, 0, 0, token_num - l, 0, 0), value=0.0
            )
            inputs_embeds = encoder_outs_pad * modality_mask[
                :, :, None
            ] + inputs_embeds * (~modality_mask[:, :, None])

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        # model_outputs = self.llm(
        #     inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        # )

        # acc = -1
        # if self.metric:
        #     with torch.no_grad():
        #         preds = torch.argmax(model_outputs.logits, -1)
        #         acc = compute_accuracy(
        #             preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100
        #         )

        return quantize_index

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return model_outputs

    @torch.no_grad()
    def inference(
        self,
        wav_path=None,
        prompt=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        **kwargs,
    ):

        device = kwargs.get("device", "cuda")
        if os.path.exists(wav_path):  # Audio-Text QA
            import whisper

            audio_raw = whisper.load_audio(wav_path)
            audio_raw = whisper.pad_or_trim(audio_raw)

            mel_size = getattr(
                self.dataset_config, "mel_size", 80
            )  # 80 for large v1 and v2, 128 for large v3
            audio_mel = (
                whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size)
                .permute(1, 0)[None, :, :]
                .to(device)
            )

            encoder_outs = self.encoder.extract_variable_length_features(
                audio_mel.permute(0, 2, 1)
            )

            if self.model_config.encoder_projector == "q-former":
                audio_mel_post_mask = torch.ones(
                    encoder_outs.size()[:-1], dtype=torch.long
                ).to(encoder_outs.device)
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
        else:  # Text QA
            encoder_outs = torch.empty(
                1, 0, self.llm.model.embed_tokens.embedding_dim
            ).to(device)

        prompt = "USER: {}\n ASSISTANT:".format(prompt)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(device)

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (encoder_outs, inputs_embeds[None, :, :]), dim=1
        )  # [audio,prompt]

        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            inputs_embeds.device
        )

        # generate
        model_outputs = self.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        return model_outputs


def model_factory(train_config, model_config, **kwargs):
    from slam_llm.models.slam_model import setup_tokenizer

    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    model = slam_model(tokenizer, train_config, model_config, **kwargs)

    ckpt_path = kwargs.get(
        "ckpt_path", None
    )  # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    return model, tokenizer


def main(kwargs: DictConfig):

    # Update the configuration for the training and sharding process
    # train_config, fsdp_config, model_config, log_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG(), LOG_CONFIG()
    # update_config((train_config, fsdp_config, model_config, log_config), **kwargs)
    train_config, fsdp_config, model_config, log_config, dataset_config = (
        kwargs.train_config,
        kwargs.fsdp_config,
        kwargs.model_config,
        kwargs.log_config,
        kwargs.dataset_config,
    )

    del kwargs.train_config
    del kwargs.fsdp_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config

    # Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.handlers[0].setFormatter(console_formatter)

    logger.addHandler(file_handler)

    logger.info("train_config: {}".format(train_config))
    logger.info("fsdp_config: {}".format(fsdp_config))
    logger.info("model_config: {}".format(model_config))

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # FIX(MZY): put the whole model to device.
    model.to(device)
    model.eval()

    # dataset_config = generate_dataset_config(train_config, kwargs)
    logger.info("dataset_config: {}".format(dataset_config))
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        logger.info(f"--> Training Set Length = {len(dataset_test)}")

    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        shuffle=False,
        batch_size=train_config.val_batch_size,
        drop_last=False,
        collate_fn=dataset_test.collator,
    )

    logger.info("=====================================")
    pred_path = kwargs.get("decode_log") + "_pred"
    gt_path = kwargs.get("decode_log") + "_gt"
    with open(pred_path, "w") as pred, open(gt_path, "w") as gt:
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            for key in batch.keys():
                batch[key] = (
                    batch[key].to(device)
                    if isinstance(batch[key], torch.Tensor)
                    else batch[key]
                )

            # index = model(**batch)
            # output_text = model.tokenizer.batch_decode(
            #     index, add_special_tokens=False, skip_special_tokens=True
            # )

            model_outputs = model.generate(**batch)
            output_text = model.tokenizer.batch_decode(
                model_outputs, add_special_tokens=False, skip_special_tokens=True
            )

            for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
                pred.write(key + "\t" + text.replace("\n", " ") + "\n")
                gt.write(key + "\t" + target + "\n")


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

    main(cfg)


if __name__ == "__main__":
    main_hydra()
