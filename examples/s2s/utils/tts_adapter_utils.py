# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

def setup_tts_adapter(adapter_config, model_config, **kwargs):
    return nn.ModuleDict(
        dict(
            post_adapter=nn.ModuleList(
                Block(adapter_config) for _ in range(adapter_config.n_layer)
            ),
            post_adapter_audio_ln=adapter_config.norm_class(
                model_config.llm_dim, eps=adapter_config.norm_eps
            ),
            post_adapter_audio_lm_head=nn.Linear(
                model_config.llm_dim, model_config.vocab_config.total_audio_vocabsize, bias=adapter_config.lm_head_bias
            ),
        )
    )

class Block(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        if config.norm_class_name == "RMSNorm":
            self.norm_class = RMSNorm

        self.norm_1 = self.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = (
            None
            if config.shared_attention_norm
            else self.norm_class(config.n_embd, eps=config.norm_eps)
        )

        if config.mlp_class_name == "GptNeoxMLP":
            self.mlp_class = GptNeoxMLP

        self.mlp = self.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓             ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───►  norm_2
           │  ↓                     │  ↓             ↓
           │  attn                  │  attn          mlp
           │  ↓                     │  ↓             │
        ┌─ └► +                     └► + ◄───────────┘
        │     norm_2
        │     ↓
        │     mlp
        │     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(x_normed, cos, sin, mask, input_pos)

        if self.config.parallel_residual:
            x_normed = x_normed if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.mlp(self.norm_2(x)) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.add_qkv_bias)
        # output projection
        # if `head_size` is explicitly specified in the config, `n_emd` might not be equal to `head_size * n_head`
        self.proj = nn.Linear(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (
            input_pos is None or self.config.n_query_groups != 1
        ):
            k = k.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )
            v = v.expand(
                B, self.config.n_query_groups, q_per_kv, T, self.config.head_size
            )

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(
            B, T, self.config.head_size * self.config.n_head
        )  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"
                )
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)




def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)



class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(
        self, size: int, dim: int = -1, eps: float = 1e-6, add_unit_offset: bool = False
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        x_normed = x_normed.to(dtype=dtype)
        if self.add_unit_offset:
            # Gemma model requires a unit offset
            # https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L176
            return x_normed * (1 + self.weight)
        return x_normed * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)


class GptNeoxMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)