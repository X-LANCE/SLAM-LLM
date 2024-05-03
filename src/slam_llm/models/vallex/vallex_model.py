import random
from typing import Dict, Iterator, List, Tuple, Union
from fairseq import utils
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from fairseq.data import Dictionary
from src.slam_llm.models.vallex.transformers import (
    LayerNorm,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from src.slam_llm.models.vallex.vallex_config import VallexConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
from dataclasses import dataclass

@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor
    acc: torch.Tensor

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, scale=1, prob_mask=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if prob_mask is not None:
        lprobs = lprobs.masked_fill(prob_mask, 0.0)
        n_class = (1-prob_mask.float()).sum()
    else:
        n_class = lprobs.size(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target) 
    # nll_loss = nll_loss * scale
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True) * scale 
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index) 
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        pad_mask_float = (1 - pad_mask.to(torch.float)).sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (n_class - 1) 
    loss = (1.0 - epsilon - eps_i) * nll_loss + \
        eps_i * smooth_loss 
    return loss / pad_mask_float, nll_loss / pad_mask_float


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx = None
    ):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state = None,
        timestep = None,
        positions = None,
    ):
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


class Transpose(nn.Identity):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


class VALLF(PreTrainedModel):
    config_class = VallexConfig
    
    def __init__(
        self,
        config: VallexConfig
    ):
        super().__init__(config)
        
        self.ar_at_dict = Dictionary.load(self.config.ar_at_dict)
        self.ar_st_dict = Dictionary.load(self.config.ar_st_dict)
        self.nar_at_dict = Dictionary.load(self.config.nar_at_dict)
        self.nar_st_dict = Dictionary.load(self.config.nar_st_dict)
        
        self.ar_at_dict.tts_flag = self.ar_at_dict.add_symbol("<TTS>")
        self.ar_st_dict.asr_flag = self.ar_st_dict.add_symbol("<ASR>")
        self.ar_st_dict.mt_flag = self.ar_st_dict.add_symbol("<MT>")
        
        self.padding_idx = self.ar_at_dict.pad()
        self.config = config
        d_model = self.config.n_dim
        nar_scale_factor = self.config.nar_scale_factor
        prepend_bos = self.config.prepend_bos
        
        norm_first = self.config.norm_first
        num_layers = self.config.n_layer
        self.NUM_AUDIO_TOKENS = self.ar_at_dict.eos()
        
        nar_d_model = int(d_model * nar_scale_factor)

        self.ar_text_embedding = nn.Embedding(len(self.ar_st_dict), d_model, self.ar_st_dict.pad())  # W_x
        if config.only_ar:
            pass
        else:
            self.nar_text_embedding = nn.Embedding(len(self.nar_st_dict), d_model, self.nar_st_dict.pad())

        # ID self.NUM_AUDIO_TOKENS     -> PAD
        # ID self.NUM_AUDIO_TOKENS + 1 -> BOS
        self.ar_audio_prepend_bos = prepend_bos
        self.ar_audio_embedding = EncodecDecoderLstm(
            dictionary=self.ar_at_dict, emb_dim=d_model
        )

        self.ar_text_prenet = nn.Identity()
        self.ar_audio_prenet = nn.Identity()

        self.ar_text_position = SinusoidalPositionalEmbedding(
            d_model,
            padding_idx=self.ar_at_dict.pad(),
            init_size=1024+self.ar_at_dict.pad()+1
        )
        self.ar_audio_position = SinusoidalPositionalEmbedding(
            d_model,
            padding_idx=self.ar_at_dict.pad(),
            init_size=1024+self.ar_at_dict.pad()+1
        )

        self.ar_decoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                self.config.n_head,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(
            d_model, len(self.ar_at_dict), bias=False
        )

        self.rng = random.Random(0)
        self.num_heads = self.config.n_head
        self.prefix_mode = self.config.prefix_mode
        self.num_quantizers = self.config.num_quantizers

        assert self.num_quantizers >= 1
        if config.only_ar:
            pass
        else:
            if self.num_quantizers > 1:
                self.nar_audio_embeddings = NATEncodecDecoderLstm(
                    codecs=[0, 1, 2, 3, 4, 5, 6, 7], dictionary=self.nar_at_dict, emb_dim=d_model
                )  # W_a

                self.nar_text_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()

                self.nar_text_position = SinusoidalPositionalEmbedding(
                    d_model,
                    padding_idx=self.nar_at_dict.pad(),
                    init_size=1024+self.nar_at_dict.pad()+1
                )
                self.nar_audio_position = SinusoidalPositionalEmbedding(
                    d_model,
                    padding_idx=self.nar_at_dict.pad(),
                    init_size=1024+self.nar_at_dict.pad()+1
                )

                self.nar_decoder = TransformerEncoder(
                    TransformerEncoderLayer(
                        nar_d_model,
                        int(self.num_heads * nar_scale_factor),
                        dim_feedforward=nar_d_model * 4,
                        dropout=0.1,
                        batch_first=True,
                        norm_first=norm_first,
                        adaptive_layer_norm=True,
                    ),
                    num_layers=int(num_layers * nar_scale_factor),
                    norm=nn.LayerNorm(nar_d_model)
                    if norm_first
                    else None,
                )
                self.nar_predict_layers = nn.ModuleList(
                    [
                        nn.Linear(nar_d_model, len(self.nar_at_dict), bias=False)
                        for i in range(self.num_quantizers)
                    ]
                )
                self.nar_stage_embeddings = None

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if name.startswith("ar_"):
                    print(f" AR parameter: {name}")
                    yield param

        if stage == 2:
            for name, param in self.named_parameters():
                if name.startswith("nar_"):
                    print(f"NAR parameter: {name}")
                    yield param

    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_"):
                    yield pair

        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_"):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # inputs, targets
        if self.ar_audio_prepend_bos:
            return (
                F.pad(targets[:, :-1], (1, 0), value=self.NUM_AUDIO_TOKENS + 1),
                targets,
            )

        return targets[:, :-1], targets[:, 1:]

class VALLE(VALLF):
    config_class = VallexConfig
    
    def __init__(
        self,
        config: VallexConfig,
        **kwargs,
    ):
        super(VALLE, self).__init__(
            config,
            **kwargs,
        )
        print(config)
        self.config = config
        d_model = self.config.n_dim
        self.eps = config.eps
        
        self.language_ID = {
            'en': 0,
            'zh': 1,
        }
        self.ar_language_embedding = nn.Embedding(3, d_model, padding_idx=2) 
        self.nar_language_embedding = nn.Embedding(3, d_model, padding_idx=2) 
        self.embed_scale = 32.0
        
    def forward(
        self,
        zh,
        en
    ):
        """
        "zh": {
            "st_tokens": zh_st,
            "at_tokens_wbos": zh_prev_at,
            "at_tokens_tgt": zh_tgt_at,
            "self_atten_mask": zh_self_atten_mask,
            "padding_mask": zh_padding_mask,
            "langid": zh_id.long()
        },
        "en": {
            "st_tokens": en_st,
            "at_tokens_wbos": en_prev_at,
            "at_tokens_tgt": en_tgt_at,
            "self_atten_mask": en_self_atten_mask,
            "padding_mask": en_padding_mask,
            "langid": en_id.long()
        }
        """
        flag = (np.random.randint(low=0, high=1000) % 2 == 0) # zh or en
        if flag:
            data = zh
        else:
            data = en
        
        st_tokens = data["st_tokens"]
        at_tokens_wbos = data["at_tokens_wbos"]
        at_tokens_tgt = data["at_tokens_tgt"]
        self_atten_mask = data["self_atten_mask"]
        padding_mask = data["padding_mask"]
        langid = data["langid"]
        
        st_len = st_tokens.size(1)
        st_emb = self.embed_scale * self.ar_text_embedding(st_tokens)
        src_lang_emb = self.embed_scale * self.ar_language_embedding(langid)
        st_emb += src_lang_emb
        st_pos = self.ar_text_position(st_tokens)
        st_emb += st_pos
        
        at_emb, _ = self.ar_audio_embedding(at_tokens_wbos, None)
        at_emb = self.embed_scale * at_emb
        tgt_lang_emb = self.embed_scale * self.ar_language_embedding(langid)
        at_emb += tgt_lang_emb
        at_pos = self.ar_audio_position(at_tokens_wbos)
        at_emb += at_pos
        
        x = torch.concat([st_emb, at_emb], dim=1)
        
        x = self.ar_decoder(
            x,
            mask=self_atten_mask,
            src_key_padding_mask=padding_mask
        )
        x = self.ar_predict_layer(x)
        x = x[:, st_len:, :]
        loss, nll_loss, lprob, right_rate = self.calculate_loss(
            x, at_tokens_tgt
        )
        return ModelOutput(logits=lprob, loss=loss, acc=right_rate), right_rate

    def calculate_loss(self, encoder_out, target, reduce=True, scale=1.0, prob_mask=None, acc=True):
        lprob = self.get_normalized_probs(encoder_out, log_probs=True)
        with torch.no_grad():
            mask = target.ne(self.padding_idx)
            n_correct = torch.sum(
                lprob.argmax(-1).masked_select(mask).eq(target.masked_select(mask))
            )
            total = torch.sum(mask)
            right_rate = n_correct * 100.0 / total
        
        lprob, target = lprob.view(-1, lprob.size(-1)), target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprob,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
            scale=scale,
            prob_mask=prob_mask
        )
        
        return loss, nll_loss, lprob, right_rate
    
    def get_normalized_probs(self, encoder_out, log_probs, sample=None):
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
            
    
    def inference_24L(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
        prompt_language: str = None,
        text_language: str = None,
        best_of: int = 1,
        length_penalty: float = 1.0,
        return_worst: bool = False,
        at_eos: int = -1
    ) -> torch.Tensor:
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)
        self.NUM_AUDIO_TOKENS = at_eos
        text = x
        x = self.embed_scale * self.ar_text_embedding(text)
        prompt_language_id = prompt_language.to(x.device)
        text_language_id = text_language.to(x.device)
        src_lang_emb = self.embed_scale * self.ar_language_embedding(prompt_language_id)
        tgt_lang_emb = self.embed_scale * self.ar_language_embedding(text_language_id)
        x[:, :enroll_x_lens, :] += src_lang_emb
        x[:, enroll_x_lens:, :] += tgt_lang_emb
        x = self.ar_text_prenet(x)
        x_pos = self.ar_text_position(text)

        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=self.ar_at_dict.tts_flag)

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        kv_cache = None
        use_kv_caching = True

        sum_logprobs = torch.zeros(best_of, device=y.device)  # implement batch decoding here
        x = x.repeat(best_of, 1, 1)
        y = y.repeat(best_of, 1)
        lstm_h = None
        first_ar = True
        while True:
            if first_ar:
                y_emb, lstm_h = self.ar_audio_embedding(y, lstm_h)
                y_emb = y_emb * self.embed_scale
                y_emb = self.ar_audio_prenet(y_emb)
                y_pos = self.ar_audio_position(y)
                y_emb[:, :prefix_len] = y_emb[:, :prefix_len] + src_lang_emb
                y_emb[:, prefix_len:] = y_emb[:, prefix_len:] + tgt_lang_emb
                xy_pos_token = torch.concat([x_pos+x, y_pos+y_emb], dim=1)
                first_ar = False
            else:
                y_emb_cur, lstm_h = self.ar_audio_embedding(y[:, -1:], lstm_h)
                y_emb_cur = y_emb_cur * self.embed_scale
                y_emb_cur = self.ar_audio_prenet(y_emb_cur)
                y_pos_cur = self.ar_audio_position(y)[:, -1:]
                y_emb_cur = y_emb_cur + src_lang_emb
                y_emb_cur = y_emb_cur + tgt_lang_emb
                xy_pos_token = torch.concat([xy_pos_token, y_pos_cur+y_emb_cur], dim=1)
            # print(xy_pos_token.size())

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)


            if use_kv_caching and kv_cache is not None:
                xy_pos = xy_pos_token[:, [-1]]
                xy_attn_mask = xy_attn_mask[:, [-1]]
            else:
                xy_pos = xy_pos_token

            xy_dec, kv_cache = self.ar_decoder.infer(
                xy_pos,
                mask=xy_attn_mask,
                past_kv=kv_cache,
                use_cache=use_kv_caching,
            )

            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples, current_logprobs = topk_sampling(
                logits, top_k=top_k, top_p=1, temperature=temperature
            )
            sum_logprobs += current_logprobs * (y[:, -1] != self.NUM_AUDIO_TOKENS)
            samples[y[:, -1] == self.NUM_AUDIO_TOKENS] = self.NUM_AUDIO_TOKENS
            completed = (samples[:, -1] == self.NUM_AUDIO_TOKENS).all()
            if (
                completed
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 32
            ):  
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )
                lengths = torch.sum(y != self.NUM_AUDIO_TOKENS, dim=1)
                avg_logprobs = sum_logprobs / lengths ** length_penalty
                # choose the best beam according to sum_logprobs
                best_beam = y[torch.argmax(avg_logprobs), :]
                worst_beam = y[torch.argmin(avg_logprobs), :]
                # strip all eos tokens
                best_beam = best_beam[best_beam != self.NUM_AUDIO_TOKENS]
                worst_beam = worst_beam[worst_beam != self.NUM_AUDIO_TOKENS]
                if return_worst:
                    y = worst_beam.unsqueeze(0)
                else:
                    y = best_beam.unsqueeze(0)
                print(f"VALL-E EOS [{prompts.shape[1]} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        if self.prefix_mode in [2, 4]:  # Exclude enrolled_phonemes
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1

        x = self.embed_scale * self.nar_text_embedding(text)
        # Add language embedding
        prompt_language_id = prompt_language.to(x.device)
        text_language_id = text_language.to(x.device)
        src_lang_emb = self.embed_scale * self.nar_language_embedding(prompt_language_id)
        tgt_lang_emb = self.embed_scale * self.nar_language_embedding(text_language_id)
        x[:, :enroll_x_lens, :] += src_lang_emb
        x[:, enroll_x_lens:, :] += tgt_lang_emb
        x = self.nar_text_prenet(x)
        x_pos = self.nar_text_position(text)

        if self.prefix_mode == 0:
            for i, predict_layer in enumerate(
                self.nar_predict_layers
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += self.embed_scale * self.nar_audio_embeddings(
                        prompts[..., i + 1]
                    )[0]
                    y_emb[:, prefix_len:] += self.embed_scale * self.nar_audio_embeddings(samples)[0]
        else:
            y_pos = self.nar_audio_position(y[:, int(self.ar_audio_prepend_bos):])
            
            ref_at_emb = self.embed_scale * self.nar_audio_embeddings(prompts)[0] + src_lang_emb
            est_at = y[:, prefix_len+int(self.ar_audio_prepend_bos):].unsqueeze(-1)
            # 
            for i in range(1, 8):
                y_emb, _ = self.nar_audio_embeddings(est_at)
                y_emb = self.embed_scale * y_emb + tgt_lang_emb
                
                y_emb = torch.concat([ref_at_emb, y_emb], dim=1)
                xy_pos = torch.concat([x+x_pos, y_emb+y_pos], dim=1)

                xy_dec = self.nar_decoder(
                    xy_pos
                )
                logits = self.nar_predict_layers[i-1](xy_dec[:, text_len + prefix_len :])
                # print(logits.size(), xy_pos.size(), xy_dec.size())
                samples = torch.argmax(logits, dim=-1)
                est_at = torch.concat([est_at, samples.unsqueeze(-1)], dim=-1)
                codes.append(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)
            
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    current_logprobs = logprobs[torch.arange(logprobs.shape[0]), token.squeeze(1)]
    return token, current_logprobs

class SLSTM(nn.Module):
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional=False):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)            
        if bidirectional:
            self.out_fc = nn.Linear(dimension*2, dimension)
        else:
            self.out_fc = None

    def forward(self, x, hidden=None):
        x = x.permute(2, 0, 1)
        y, hidden = self.lstm(x, hidden)
        if self.out_fc is not None:
            y = self.out_fc(y)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y, hidden
    
class EncodecDecoderLstm(nn.Module):
    def __init__(self, dictionary, emb_dim, 
                 out_dim=None,
                 num_layers=3, lstm_skip=True, lstm_bidire=False,
                 activation_param={'alpha': 1.0}, **kwargs):
        super().__init__()
        
        # Identity()
        if out_dim is None:
            out_dim = emb_dim
        self.slstm = SLSTM(dimension=out_dim, num_layers=num_layers, skip=lstm_skip, bidirectional=lstm_bidire)
        self.elu = nn.ELU(**activation_param)
        self.embedding_dim = emb_dim
        self.padding_idx = dictionary.pad()
        self.emb = nn.Embedding(len(dictionary), emb_dim, dictionary.pad_index)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x (_type_): B,T,D
        """
        # print(x.size())
        quantized_out = self.emb(x)
        out, hidden = self.slstm(quantized_out.permute(0,2,1), hidden)
        out = self.elu(out)
        return out.permute(0,2,1), hidden

class NATEncodecDecoderLstm(nn.Module):
    def __init__(self, codecs, dictionary, emb_dim, 
                 out_dim=None,
                 num_layers=3, lstm_skip=True, lstm_bidire=False,
                 activation_param={'alpha': 1.0}, **kwargs):
        super().__init__()
        
        # Identity()
        if out_dim is None:
            out_dim = emb_dim
        self.slstm = SLSTM(dimension=out_dim, num_layers=num_layers, skip=lstm_skip, bidirectional=lstm_bidire)
        self.elu = nn.ELU(**activation_param)
        self.codecs = codecs
        self.embedding_dim = emb_dim
        self.padding_idx = dictionary.pad()
        self.emb_list = nn.ModuleList(
            [nn.Embedding(len(dictionary), emb_dim, dictionary.pad_index) for i in range(len(self.codecs))]
        )
    
    def forward(self, x, hidden=None):
        """
        Args:
            x (_type_): B,T,D
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(-1)
        
        if x.size(2) != len(self.codecs) and x.size(1) == len(self.codecs):
            x = x.permute(0, 2, 1)
        
        quantized_out = 0
        for i in range(x.size(2)):
            quantized = self.emb_list[i](x[: , :, i])
            quantized_out = quantized_out + quantized
        # quantized_out = quantized_out / len(self.codecs)
        
        out, hidden = self.slstm(quantized_out.permute(0,2,1), hidden)
        out = self.elu(out)
        return out.permute(0,2,1), hidden

AutoModel.register(VallexConfig, VALLE)