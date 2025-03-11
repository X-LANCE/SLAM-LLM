import torch
import os
import logging
import torch.nn.functional as F
import torch.nn as nn
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_encoder,
    setup_encoder_projector,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from typing import List, Optional, Generator
from slam_llm.utils.metric import compute_accuracy
from transformers import T5ForConditionalGeneration
from tqdm import tqdm
from utils.tts_adapter_utils import setup_tts_adapter
from utils.codec_utils import setup_codec
from utils.trick_utils import partial_freeze_weights, train_embedding_layer_only
from utils.snac_utils import get_snac, generate_audio_data, simple_shift
from utils.snac_utils import layershift as layer_shift
from utils.projector_utils import setup_group_decode_adapter
from slam_llm.utils.config_utils import generate_peft_config
from peft import get_peft_model

logger = logging.getLogger(__name__)
import pdb

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    whisper_model = None
    if train_config.task_type == "s2s" or train_config.task_type == "asr":
        if not model_config.whisper_decode:
            encoder = setup_encoder(train_config, model_config, **kwargs)
        else:
            whisper_model = setup_encoder(train_config, model_config, **kwargs)
            encoder = whisper_model.encoder
    elif train_config.task_type == "tts":
        encoder = None
    else:
        raise NotImplementedError

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    if encoder is not None:
        encoder_projector = setup_encoder_projector(
            train_config, model_config, **kwargs
        )
        if train_config.freeze_encoder_projector:
            for name, param in encoder_projector.named_parameters():
                param.requires_grad = False
            encoder_projector.eval()
    else:
        encoder_projector = None

    codec_decoder = None
    if model_config.codec_decode:
        codec_decoder = setup_codec(train_config, model_config, **kwargs)

    tts_adapter = None
    if model_config.tts_adapter:
        adapter_config = model_config.tts_adapter_config
        tts_adapter = setup_tts_adapter(adapter_config, model_config, **kwargs)

    group_decode_adapter = None
    if model_config.group_decode:
        group_decode_adapter = setup_group_decode_adapter(model_config, train_config, **kwargs)
        if train_config.freeze_group_decode_adapter:
            for name, param in group_decode_adapter.named_parameters():
                param.requires_grad = False
            group_decode_adapter.eval()

    model = slam_model_s2s_dpo(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        tts_adapter,
        codec_decoder,
        group_decode_adapter,
        whisper_model,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get(
        "ckpt_path", None
    )  # FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None: #x
        logger.info("loading other parts from: {}\n".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    if train_config.train_audio_embed_only:
        partial_freeze_weights(model, model_config.vocab_config.padded_text_vocabsize, model_config.vocab_config.total_vocabsize)

    if train_config.train_embed_only:
        train_embedding_layer_only(model)
    # pdb.set_trace()
    # fixme: here has a bug -> during inference, we need load the ckpt again since the ckpt above is for FFT stage
    if train_config.use_peft:
        logger.info("setup peft for llm")
        peft_config = generate_peft_config(train_config)
        model.llm = get_peft_model(model.llm, peft_config)
        if int(os.environ.get("RANK", "0")) == 0:
            model.llm.print_trainable_parameters()

    if kwargs.get("peft_ckpt", None):
        logger.info("loading peft-stage ckpt from: {}\n".format(kwargs.get("peft_ckpt")))
        ckpt_dict = torch.load(kwargs.get("peft_ckpt"), map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)
        # msg= model.load_state_dict(ckpt_dict, strict=False)logger.info(msg)
    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class slam_model_s2s_dpo(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        tts_adapter,
        codec_decoder,
        group_decode_adapter,
        whisper_model,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )

        # resize llm embedding layer
        self.original_vocabsize = self.llm.lm_head.weight.size(0)
        if self.model_config.vocab_config.total_vocabsize != self.original_vocabsize:
            self.llm.resize_token_embeddings(self.model_config.vocab_config.total_vocabsize)  #!!!!!
            # embed_tokens = self.llm.model.embed_tokens
            # save_path = '/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/belle/tts/tn_remake/embed_tokens.pth'
            # save_path = '/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/belle/tts/tn_remake/embed_tokens_infer.pth'
            # torch.save(embed_tokens, save_path)
            # self.llm.model.embed_tokens = torch.load('/nfs/yangguanrou.ygr/codes/SLAM-LLM/examples/s2s/scripts/ygr/belle/tts/tn_remake/embed_tokens.pth') #重要

            if int(os.environ.get("RANK", "0")) == 0:
                logger.info("Resize llm embedding layer's vocab size to {}\n".format(self.model_config.vocab_config.total_vocabsize))

        self.codec_decoder = codec_decoder
        self.whisper_model = whisper_model
        self.tts_adapter = tts_adapter
        self.code_layer = self.model_config.vocab_config.code_layer
        self.group_decode_adapter = group_decode_adapter


    def forward(self, chosen_dict, reject_dict, **kwargs): #chosen_dict可以算一下acc
        chosen_model_outputs, chosen_text_acc, chosen_audio_acc, chosen_loss_recorder = self.single_forward(**chosen_dict, is_chosen=True)
        reject_model_outputs, reject_text_acc, reject_audio_acc, reject_loss_recorder = self.single_forward(**reject_dict, is_chosen=False)

        chosen_logps = chosen_model_outputs.loss
        reject_logps = reject_model_outputs.loss
        # pdb.set_trace()
        return {
            "chosen_logps": chosen_logps,
            "reject_logps": reject_logps,
            "chosen_text_acc": chosen_text_acc,
            "chosen_audio_acc": chosen_audio_acc,
            "chosen_loss_recorder":chosen_loss_recorder,
            
            # "chosen_model_outputs_logits": chosen_model_outputs.logits,
        }

    def single_forward(self,
                input_ids: torch.LongTensor = None, #torch.Size([6, 4, 178])
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                is_chosen: Optional[bool] = None,
                **kwargs,
                ):
        # pdb.set_trace()
        audio_mel = kwargs.get("audio_mel", None)
        audio_embedding = kwargs.get("audio_embedding", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)

        modality_mask = kwargs.get("modality_mask", None)

        encoder_outs = None
        if audio_mel is not None or audio is not None:
            if audio_embedding is None:
                if self.train_config.freeze_encoder: # freeze encoder
                    self.encoder.eval()

                if self.model_config.encoder_name == "whisper":
                    encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim
                if self.model_config.encoder_name == "wavlm":
                    encoder_outs = self.encoder.extract_features(audio, 1 - audio_mask) #(FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
                if self.model_config.encoder_name == "hubert":
                    results = self.encoder(source = audio, padding_mask = 1-audio_mask)
                    if self.model_config.encoder_type == "pretrain":
                        encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
                    if self.model_config.encoder_type == "finetune":
                        encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                        encoder_outs = encoder_outs.transpose(0, 1)
                if self.encoder is None:
                    encoder_outs = audio_mel if audio_mel is not None else audio
            else:
                encoder_outs = audio_embedding

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
            if self.model_config.encoder_projector == "cov1d-linear": 
                encoder_outs = self.encoder_projector(encoder_outs)

        if input_ids is not None: #torch.Size([4, 8, 439]) 前7个全身padding ; torch.Size([4, 2, 220])
            input_ids[input_ids == -1] = 0  # [btz, code_layer + 1, seq_length]

            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)  # [btz, code_layer + 1, seq_length, emb_dim] #torch.Size([4, 8, 439, 896]);torch.Size([4, 2, 220, 896])
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        if modality_mask is not None and encoder_outs is not None: #x
            if self.train_config.modeling_paradigm == "parallel":
                modality_mask = modality_mask.unsqueeze(1).repeat(1, self.code_layer, 1)  # [btz, code_layer, seq_length]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=2)
                modality_lengths = torch.clamp(modality_mask.sum(dim=2), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    for j in range(self.code_layer):
                        start_idx = modality_mask_start_indices[i, j].item()
                        length = modality_lengths[i][j]
                        encoder_outs_pad[i, j, start_idx:start_idx+length] = encoder_outs[i, :length]
                
                inputs_embeds[:, :self.code_layer, :, :] = encoder_outs_pad[:, :self.code_layer, :, :] + inputs_embeds[:, :self.code_layer, :, :] * (~modality_mask[:, :, :, None])
        
                inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers #torch.Size([4, 439, 896]) ; torch.Size([4, 220, 896]) 这一步之后形状会有变化

            elif self.train_config.modeling_paradigm == "interleaved":
                inputs_embeds = inputs_embeds.squeeze(1)  # [btz, seq_length, emb_dim]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
                modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    encoder_outs_pad[
                        i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                    ] = encoder_outs[i][:modality_lengths[i]]
                
                inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
            
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers #torch.Size([4, 439, 896]) ; torch.Size([4, 220, 896]) 这一步之后形状会有变化

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        text_labels = labels[:,self.code_layer] if labels is not None else None #torch.Size([4, 448])
        audio_labels = labels[:, :self.code_layer] if labels is not None else None #torch.Size([4, 7, 448])
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=text_labels)    # here we use the text token layer as the target label

        # parrallel / interleaved
        # TODO: add tts adapter forward
        if self.train_config.modeling_paradigm == "parallel":
            # pdb.set_trace()
            x_ori = model_outputs.logits #torch.Size([4, 448, 181120]). torch.Size([6, 178, 156160])
            text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
            audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
            xt = x_ori[..., :text_vocab_size] #torch.Size([4, 448, 152000]);torch.Size([4, 220, 152000])
            xa = []

            if self.group_decode_adapter is not None:
                x_audio_ori = x_ori[..., text_vocab_size:] #torch.Size([6, 178, 4160])
                x_audio = self.group_decode_adapter(x_audio_ori) #torch.Size([6, 178, 12480])
                for i in range(self.code_layer):
                    xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])
            else:
                for i in range(self.code_layer):
                    xa.append(x_ori[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)]) #xa[0].shape:torch.Size([4, 220, 4160])

            loss_recorder = []
            total_loss, loss_recorder = self.compute_parallel_loss(xt, text_labels, xa, audio_labels) #torch.Size([4, 220, 152000]), [torch.Size([4, 220])], torch.Size([4, 220, 4160]), torch.Size([4, 1, 220])
            model_outputs.loss = total_loss
        elif self.train_config.modeling_paradigm == "interleaved":
            x_ori = model_outputs.logits
        else:
            raise NotImplementedError

        text_acc = -1
        audio_acc = [-1 for _ in range(self.code_layer)] if self.code_layer > 0 else -1
        if self.metric and is_chosen:
            with torch.no_grad():
                if self.train_config.modeling_paradigm == "parallel":
                    preds = torch.argmax(xt, -1)
                    text_acc = compute_accuracy(preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                    preds_audio = [torch.argmax(xa[i], -1) for i in range(self.code_layer)]
                    audio_acc = [compute_accuracy(preds_audio[i].detach()[:, :-1], audio_labels[:, i, 1:], ignore_label=-100) for i in range(self.code_layer)]
                elif self.train_config.modeling_paradigm == "interleaved":
                    # pdb.set_trace()
                    preds_start_idx = (text_labels != -100).float().argmax(dim=1)  #text_labels 张量中每一行第一个不等于 -100 的元素的索引。 tensor([131, 131], device='cuda:0')
                    preds = torch.argmax(x_ori, -1) #torch.Size([2, 2531]) 预测的id
                    # new_preds, new_labels = self.extract_interleaved_tokens(preds, text_labels, preds_start_idx) #new_preds:torch.Size([2, 600]) 
                    text_preds, text_labels, audio_preds, audio_labels = self.extract_interleaved_tokens(preds, text_labels, preds_start_idx) #new_preds:torch.Size([2, 600]) 

                    # padding token is not counted in text_acc
                    text_pad_token = self.model_config.vocab_config.pad_t
                    text_labels[text_labels == text_pad_token] = -100
                    
                    text_acc = compute_accuracy(text_preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                    audio_pad_token = self.model_config.vocab_config.pad_a
                    audio_labels[audio_labels == audio_pad_token] = -100
                    
                    audio_acc = [ compute_accuracy(audio_preds.detach()[:, :-1], audio_labels.detach()[:, 1:], ignore_label=-100)]

                    loss_recorder = None
                else:
                    raise NotImplementedError

        # metrics = {"text_acc": text_acc, "audio_acc": audio_acc, "layer_loss": loss_recorder}
        return model_outputs, text_acc, audio_acc, loss_recorder #model_outputs 各层取平均，loss_recorder记录每一层的loss值  如果是interleave,rest-> [tensor(0.4856, device='cuda:0'), -1, None] (train_utils.py)



    def compute_parallel_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute the parallel loss for text and audio layers.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        layer_loss = [0 for _ in range(self.code_layer+1) ] #[0, 0, 0, 0]
        
        if text_labels is not None: #mean 是在batch上mean
            # text_loss = F.cross_entropy(xt.reshape(-1, text_vocab_size), text_labels.reshape(-1), ignore_index=-100)
            text_loss = F.cross_entropy(xt[:, :-1, :].reshape(-1, text_vocab_size), text_labels[:, 1:].reshape(-1), ignore_index=-100) #torch.Size([6, 178]) torch.Size([1062])
            layer_loss[self.code_layer] = text_loss
        else:
            text_loss = 0

        total_audio_loss = 0
        single_audio_loss = 0
        for i in range(self.code_layer):
            if audio_labels[:,i] is not None:
                # audio_loss += F.cross_entropy(xa[i].reshape(-1, audio_vocab_size), audio_labels[:,i].reshape(-1), ignore_index=-100)
                single_audio_loss = F.cross_entropy(xa[i][:, :-1, :].reshape(-1, audio_vocab_size), audio_labels[:, i, 1:].reshape(-1), ignore_index=-100)
                layer_loss[i] = single_audio_loss
                total_audio_loss += single_audio_loss

        total_loss = (text_loss + total_audio_loss) / (self.code_layer+1)
        return total_loss, layer_loss #10.63 [13.54,7.71]    #就要这个就可以


    @torch.no_grad()
    def generate(self,
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
        ) #torch.Size([1, 32, 896]) ?作用是

        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        num_latency_tokens = kwargs.get("num_latency_tokens", 0)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False     # Track whether text generation has ended
        audio_end = False    # Track whether audio generation has ended

        if self.train_config.modeling_paradigm == "interleaved":
            model_outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                num_beams=kwargs.get("num_beams", 4),
                do_sample=kwargs.get("do_sample", False),
                min_length=kwargs.get("min_length", 1),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=text_repetition_penalty,
                length_penalty=kwargs.get("length_penalty", 1.0),
                temperature=kwargs.get("temperature", 1.0),
                attention_mask=attention_mask,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=layershift(eoa, 0),
                pad_token_id=self.tokenizer.pad_token_id
            )
            model_outputs = self.process_interleaved_output(model_outputs)
            return model_outputs

        # NOTE: currently, we only support greedy decoding and sampling for parallel generation, no beam search  这是个问题？
        for step in tqdm(range(max_new_tokens), desc="Generating"):
            # pdb.set_trace()
            if current_input_text is not None: #x
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens, current_input_text.unsqueeze(1)], dim=1)
                if self.train_config.use_peft:
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids) #
                else:
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids) #正常情况
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1) #!!!
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]  #torch.Size([1, 33, 896])
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]                      # batch size is 1  torch.Size([33, 156160])
            past_key_values = outputs.past_key_values       # Update past_key_values for the next step

            # Split logits into text and audio layers based on vocab size
            xt_logits = logits[..., :text_vocab_size] #torch.Size([33, 152000])
            if self.group_decode_adapter is not None: #
                xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:]) #输入: torch.Size([33, 4160]) -> torch.Size([33, 12480]) 原来是这么搞的  (linear): Linear(in_features=4160, out_features=12480, bias=True)
                xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)] #torch.Size([33, 4160]) 里头每一个是
            else:
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

            # Apply repetition penalty to the logits
            xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)  #已经生成的text token 的列表
            for i in range(self.code_layer):
                xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)  #要加这一步吗

            if not text_end:
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
            else:
                next_token_text = torch.tensor([pad_t], device=input_ids.device)
            # logger.info(next_token_text.item())
            next_tokens_audio = []
            for i in range(self.code_layer):
                if not audio_end and not decode_text_only and num_latency_tokens <= step:
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                else:
                    next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                next_tokens_audio.append(next_token_audio)

            if eoa in next_tokens_audio or decode_text_only:
                audio_end = True
            if next_token_text == eot:
                text_end = True
            
            # Update input_ids for the next step
            current_input_text = next_token_text
            for i in range(self.code_layer):
                current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            # Append generated tokens to the tensor
            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]  # Audio layers
            generated_ids[self.code_layer][step] = next_token_text  # Text layer


            if self.model_config.use_text_stream:
                if audio_end and text_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break       
            else:
                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break     

        # Concatenate the generated tokens to form the complete sequence
        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][0]] if eot in text_tokens else text_tokens

        if eoa in generated_ids[self.code_layer - 1] and do_layershift: #
            end_ids = (generated_ids[self.code_layer - 1] == eoa).nonzero(as_tuple=True)[0][0]
            for i in range(self.code_layer):
                audio_tokens = generated_ids[i]
                generated_ids[i] = audio_tokens[:end_ids] #没有4096

        if upsampling_factor > 1:
            generated_ids[self.code_layer] = generated_ids[self.code_layer][::upsampling_factor]
            
        return generated_ids


    @torch.no_grad()
    def stream_generate(
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
    ) -> Generator:
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

        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False
        audio_end = False
        begin_generate = False
        text_stream_end = False

        stream_stride = kwargs.get("stream_stride", 4)
        current_index = 0
        index = 0
        last_text_index = 0

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if current_input_text is not None:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens, current_input_text.unsqueeze(1)], dim=1)
                inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]
            past_key_values = outputs.past_key_values

            xt_logits = logits[..., :text_vocab_size]
            if self.group_decode_adapter is not None:
                xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
            else:
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

            xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
            for i in range(self.code_layer):
                xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)

            if not text_end:
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
            else:
                next_token_text = torch.tensor([pad_t], device=input_ids.device)

            next_tokens_audio = []
            for i in range(self.code_layer):
                if not audio_end and not decode_text_only:
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                else:
                    next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                next_tokens_audio.append(next_token_audio)

            if eoa in next_tokens_audio or decode_text_only:
                audio_end = True
            if next_token_text == eot:
                text_end = True
            
            current_input_text = next_token_text
            for i in range(self.code_layer):
                current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]
            generated_ids[self.code_layer][step] = next_token_text

            if audio_end and text_end:
                for i in range(self.code_layer):
                    generated_ids[i] = generated_ids[i][:step+1]
                break

            if index == self.code_layer:
                begin_generate = True

            if begin_generate and not decode_text_only:
                current_index += 1
                if current_index == stream_stride:
                    current_index = 0
                    snac = get_snac(generated_ids, index, stream_stride)
                    audio_stream = generate_audio_data(snac, self.codec_decoder, input_ids.device)
                    text_stream = generated_ids[self.code_layer][last_text_index:index] if not text_stream_end else None

                    if text_stream is not None and eot in text_stream:
                        text_stream_end = True
                        text_stream = text_stream[:text_stream.index(eot)]

                    last_text_index = index
                    yield {
                        "audio_stream": audio_stream,
                        "text_stream": text_stream,
                    }

            if decode_text_only:
                yield {
                    "audio_stream": None,
                    "text_stream": next_token_text,
                }
            
            index += 1

        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][0]] if eot in text_tokens else text_tokens

        if eoa in generated_ids[self.code_layer - 1] and do_layershift:
            end_ids = (generated_ids[self.code_layer - 1] == eoa).nonzero(as_tuple=True)[0][0]
            for i in range(self.code_layer):
                audio_tokens = generated_ids[i]
                generated_ids[i] = audio_tokens[:end_ids]

        if upsampling_factor > 1:
            generated_ids[self.code_layer] = generated_ids[self.code_layer][::upsampling_factor]
            
        return generated_ids


    @torch.no_grad()
    def sample_next_token(self, logits, **kwargs):
        """
        Generate the next token based on the model output logits.
        Supports both greedy decoding, top-k sampling, and top-p (nucleus) sampling.
        """
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 0)
        top_p = kwargs.get("top_p", 1.0)
        num_samples = kwargs.get("num_samples", 1)

        # Adjust logits with temperature
        logits = logits.squeeze(0)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Make sure top_k is within the vocab size
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[..., [-1]]] = -float('Inf')  # Filter tokens not in top_k

        # Top-p filtering (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            # Perform sampling
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples)
        else:
            # Greedy decoding (argmax)
            return torch.argmax(logits, dim=-1, keepdim=True)


    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

        # Scatter the updated scores back into logits
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits

    # def extract_interleaved_tokens(self, preds, text_labels, preds_start_idx):
    #     """
    #     Extract predictions and labels in interleaved mode.
    #     """
    #     interleaved_text_num = self.train_config.interleaved_text_token_num
    #     interleaved_audio_num = self.train_config.interleaved_audio_token_num
        
    #     new_preds_batch = []
    #     new_labels_batch = []

    #     for i in range(preds.size(0)):
    #         new_preds = []
    #         new_labels = []
    #         start_idx = preds_start_idx[i].item()
    #         total_length = preds.size(1)
    #         idx = start_idx
            
    #         while idx < total_length:
    #             if idx + interleaved_text_num <= total_length:
    #                 new_preds.append(preds[i, idx:idx + interleaved_text_num].unsqueeze(0))
    #                 new_labels.append(text_labels[i, idx:idx + interleaved_text_num].unsqueeze(0))
    #             idx += interleaved_text_num
                
    #             idx += interleaved_audio_num
            
    #         new_preds_batch.append(torch.cat(new_preds, dim=1))
    #         new_labels_batch.append(torch.cat(new_labels, dim=1))
        
    #     new_preds = torch.cat(new_preds_batch, dim=0) if new_preds else torch.empty(0, interleaved_text_num)
    #     new_labels = torch.cat(new_labels_batch, dim=0) if new_labels else torch.empty(0, interleaved_text_num)
        
    #     return new_preds, new_labels

    def extract_interleaved_tokens(self, preds, text_labels, preds_start_idx):
        """
        Extract predictions and labels in interleaved mode.
        """
        interleaved_text_num = self.train_config.interleaved_text_token_num
        interleaved_audio_num = self.train_config.interleaved_audio_token_num
        
        text_preds_batch = []
        text_labels_batch = []
        audio_preds_batch = []
        audio_labels_batch = []

        for i in range(preds.size(0)):
            text_preds = []
            text_labels_list = []
            audio_preds = []
            audio_labels = []

            start_idx = preds_start_idx[i].item()
            total_length = preds.size(1)
            idx = start_idx
            
            while idx < total_length:
                # 提取文本部分
                if idx + interleaved_text_num <= total_length:
                    text_preds.append(preds[i, idx:idx + interleaved_text_num].unsqueeze(0))
                    text_labels_list.append(text_labels[i, idx:idx + interleaved_text_num].unsqueeze(0))
                idx += interleaved_text_num
                
                # 提取音频部分
                if idx + interleaved_audio_num <= total_length:
                    audio_preds.append(preds[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                    audio_labels.append(text_labels[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                idx += interleaved_audio_num
            
            # 将每个样本的结果合并
            # if text_preds:
            text_preds_batch.append(torch.cat(text_preds, dim=1))
            # if text_labels_list:
            text_labels_batch.append(torch.cat(text_labels_list, dim=1))
            # if audio_preds:
            audio_preds_batch.append(torch.cat(audio_preds, dim=1))
            # if audio_labels:
            audio_labels_batch.append(torch.cat(audio_labels, dim=1))

        # 合并所有批次的结果
        text_preds = torch.cat(text_preds_batch, dim=0) if text_preds_batch else torch.empty(0, interleaved_text_num)
        text_labels = torch.cat(text_labels_batch, dim=0) if text_labels_batch else torch.empty(0, interleaved_text_num)
        audio_preds = torch.cat(audio_preds_batch, dim=0) if audio_preds_batch else torch.empty(0, interleaved_audio_num)
        audio_labels = torch.cat(audio_labels_batch, dim=0) if audio_labels_batch else torch.empty(0, interleaved_audio_num)
        
        return text_preds, text_labels, audio_preds, audio_labels
        
    def process_interleaved_output(self, model_outputs):
        """
        Parse the interleaved generation results and separate tokens into audio and text.
        """
        batch_size, seq_len = model_outputs.shape
        interleaved_audio_token_num = self.train_config.interleaved_audio_token_num
        interleaved_text_token_num = self.train_config.interleaved_text_token_num
        audio_shift = self.model_config.vocab_config.padded_text_vocabsize

        audio_tokens, text_tokens = [], []

        for i in range(batch_size):
            current_audio, current_text = [], []
            sequence = model_outputs[i]

            idx = 0
            while idx < seq_len:
                text_chunk = sequence[idx: idx + interleaved_text_token_num]
                current_text.append(text_chunk)
                idx += interleaved_text_token_num

                if idx < seq_len:
                    audio_chunk = sequence[idx: idx + interleaved_audio_token_num] - audio_shift
                    current_audio.append(audio_chunk)
                    idx += interleaved_audio_token_num

            audio_tokens.append(torch.cat(current_audio) if current_audio else torch.tensor([], device=model_outputs.device))
            text_tokens.append(torch.cat(current_text) if current_text else torch.tensor([], device=model_outputs.device))

        audio_tokens = torch.stack(audio_tokens) if audio_tokens else torch.empty((batch_size, 0), device=model_outputs.device)
        text_tokens = torch.stack(text_tokens) if text_tokens else torch.empty((batch_size, 0), device=model_outputs.device)

        return {
            "audio": audio_tokens,
            "text": text_tokens.squeeze(0),
        }