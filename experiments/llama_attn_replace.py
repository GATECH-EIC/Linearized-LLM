# Modified based on https://github.com/lm-sys/FastChat

import warnings
from typing import Optional, Tuple
import os
import torch
from torch import nn
import transformers
from einops import rearrange

from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func
)

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half
from flash_attn.bert_padding import unpad_input, pad_input
import math
split = int(os.environ.get("SPLIT", 4))
group_size_ratio = 1/split
def forward_flashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if not self.training:
        raise ValueError("This function is only for training. For inference, please use forward_flashattn_inference.")

    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask.repeat(2, 1)
    nheads = qkv.shape[-2]
    # shift

    group_size = int(q_len * group_size_ratio)
    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d." % (q_len, group_size))

    qkv = qkv.reshape(bsz, q_len, 3, 2, self.num_heads // 2, self.head_dim).permute(0, 3, 1, 2, 4, 5).reshape(bsz * 2,
                                                                                                              q_len, 3,
                                                                                                              self.num_heads // 2,
                                                                                                              self.head_dim)
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    cu_q_len_tmp = torch.arange(0, max_s, group_size, device=key_padding_mask.device, dtype=cu_q_lens.dtype)
    cu_q_len_tmp = torch.stack([cu_q_len_tmp, cu_q_len_tmp + group_size // 2]).repeat(bsz, 1) + cu_q_lens[:-1].unsqueeze(-1)
    cu_q_lens = torch.cat([cu_q_len_tmp, cu_q_lens[1:].unsqueeze(-1)], dim=-1).view(-1)

    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads // 2
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, group_size, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz * 2, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads // 2,
    )
    output = output.reshape(bsz, 2, q_len, nheads // 2, self.head_dim).transpose(1, 2).reshape(bsz, q_len, nheads,
                                                                                               self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value

def forward_flashattn_full(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack(
        [query_states, key_states, value_states], dim=2
    )  # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
    x_unpad = rearrange(
        x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
    )
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    output = output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, past_key_value


def forward_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)

    if q_len % group_size > 0:
        raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    num_group = q_len // group_size

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # shift
    def shift(qkv, bsz, q_len, group_size, num_heads, head_dim):
        if not NOSHIFT:
            qkv[:, num_heads // 2:] = qkv[:, num_heads // 2:].roll(-group_size // 2, dims=2)
        qkv = qkv.transpose(1, 2).reshape(bsz * (q_len // group_size), group_size, num_heads, head_dim).transpose(1, 2)
        return qkv
    if ADD_CONV:
        value_states0 = value_states
    query_states = shift(query_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    key_states = shift(key_states, bsz, q_len, group_size, self.num_heads, self.head_dim)
    value_states = shift(value_states, bsz, q_len, group_size, self.num_heads, self.head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz * num_group, self.num_heads, group_size, group_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_group, self.num_heads, group_size, group_size)}, but is"
            f" {attn_weights.size()}"
        )

    attention_mask = attention_mask[:, :, :group_size, :group_size].repeat(num_group, 1, 1, 1)
    if attention_mask is not None:
        if attention_mask.size() != (bsz * num_group, 1, group_size, group_size):
            raise ValueError(
                f"Attention mask should be of size {(bsz * num_group, 1, group_size, group_size)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz * num_group, self.num_heads, group_size, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz * num_group, self.num_heads, group_size, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)

    # shift back
    attn_output[:, :, self.num_heads//2:] = attn_output[:, :, self.num_heads//2:].roll(group_size//2, dims=1)
    if ADD_CONV:
        
        self.dwconv.weight.data *= self.conv_mask
        conv_out = self.dwconv(value_states0) 
        #print("conv: ", value_states0.size(), conv_out.size(), attn_output.size())
        attn_output += conv_out.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
import os 
USE_LOCAL = int(os.environ.get("USE_LOCAL", 0))
USE_GLOBAL = int(os.environ.get("USE_GLOBAL", 0))
GLOBAL_FACTOR = int(os.environ.get("GLOBAL_FACTOR", 0))
LOCAL_FACTOR = int(os.environ.get("LOCAL_FACTOR", 0))
USE_CASTLING = int(os.environ.get("USE_CASTLING", 0))
ADD_CONV = int(os.environ.get("ADD_CONV", 0))
ZERO_CONV = int(os.environ.get("ZERO_CONV", 0))
USE_YOSO = int(os.environ.get("USE_YOSO", 0))
HASHCODE_LEN = int(os.environ.get("HASHCODE_LEN", 8))
USE_LINEAR = int(os.environ.get("USE_LINEAR", 0))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", 64))
NORM_LOCAL = int(os.environ.get("NORM_LOCAL", 0))
NORM_GLOBAL = int(os.environ.get("NORM_GLOBAL", 0))
NORM_TOGETHER = int(os.environ.get("NORM_TOGETHER", 0))
NOSHIFT = int(os.environ.get("NOSHIFT", 0))

@torch.compile
def yoso_pointwise(attn_weights):
    return torch.pow((1 - torch.acos(0.98 * attn_weights) / math.pi), HASHCODE_LEN)

def _local_quadratic_attn_p( q, k, v, causal_mask=None):
    qk = torch.einsum('bhgns,bhgms->bhgnm', q, k)
    if USE_YOSO:
        a = yoso_pointwise(qk)
    else:          
        a = torch.nn.functional.relu(qk) ** 2
    if causal_mask is not None:
        a = causal_mask.unsqueeze(1) * a
    #a = causal_mask(a) if causal else a
    return torch.einsum('bhgnm,bhgme->bhgne', a, v)

def _local_quadratic_attn_p_g1( q, k, v, causal_mask=None):
    #print("g1 size: ", q.size(), k.size(), v.size())
    qk = torch.einsum('bhns,bhms->bhnm', q, k)
    if USE_YOSO:
        a = yoso_pointwise(qk)
    else:          
        a = torch.nn.functional.relu(qk) ** 2
    if causal_mask is not None:
        a = causal_mask.unsqueeze(1) * a
    #a = causal_mask(a) if causal else a
    return torch.einsum('bhnm,bhme->bhne', a, v)

def _global_linear_attn_p( q, k, v, causal):
    assert causal
    if causal:
        kv = torch.einsum('bhgcs,bhgce->bhgse', k, v)
        kv = torch.cumsum(kv, dim=2) - kv
        return torch.einsum('bhgcs,bhgse->bhgce', q, kv)
    else:
        kv = torch.einsum('bhgcs,bhgce->bhse', k, v)
        return torch.einsum('bhgcs,bhse->bhgce', q, kv)
        
import einops
def _global_linear_attn_causal(q, k, v, q_res, k_res, v_res):
    #print("size: ", q.size(), k.size(), v.size(), q_res.size(), k_res.size(), v_res.size())
    kv = torch.einsum('bhgcs,bhgce->bhgse', k, v)
    kv1 = torch.cumsum(kv, dim=2) 
    if kv1.size(2) > 0: 
        qkv_res = torch.einsum('bhcs,bhse->bhce', q_res, kv1[:,:,-1,:,:])
        if q.size(2) > 0:
            qkv0 = torch.einsum('bhgcs,bhgse->bhgce', q, kv1 - kv)
        else:
            qkv0 = torch.zeros((q_res.size()[0], q_res.size()[1], 0, q_res.size()[3], kv1.size()[-1]), device=q_res.device, dtype=q_res.dtype)
    else:
        qkv_res = torch.zeros((q_res.size()[0], q_res.size()[1], q_res.size()[2], kv1.size()[-1]), device=q_res.device, dtype=q_res.dtype) #torch.einsum('bhcs,bhse->bhce', q_res, kv1[:,:,0,:,:])
        if q.size(2) > 0:
            qkv0 = torch.zeros((q_res.size()[0], q_res.size()[1], 0, q_res.size()[3], kv1.size()[-1]), device=q_res.device, dtype=q_res.dtype)
        else:
            qkv0 = torch.einsum('bhgcs,bhgse->bhgce', q, kv1 - kv)
    
    #print("ss: ", qkv0.size(), qkv_res.size())
    ret = torch.cat((einops.rearrange(qkv0, "b h g c d -> b h (g c) d"), qkv_res), dim=2)
    #print(ret.size(), qkv0.size(), qkv_res.size())
    return ret

def linear_forward_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    group_size = int(q_len * group_size_ratio)

    #if q_len % group_size > 0:
    #    raise ValueError("q_len %d should be divisible by group size %d."%(q_len, group_size))
    #num_group = q_len // group_size

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    #print("at0: ", attention_mask.size())
    if attention_mask is not None:
        attention_mask0 = torch.where(attention_mask < 0, 0, 1).squeeze(0).squeeze(0)
        

    import einops

    GROUP_SIZE_Q = min(GROUP_SIZE, query_states.size(2) % GROUP_SIZE)
    GROUP_SIZE_K = min(GROUP_SIZE, key_states.size(2) % GROUP_SIZE)

    query_t = query_states
    key_t = key_states 
    value_t = value_states
    query_t = query_t / query_t.norm(dim=-1, keepdim=True)
    key_t = key_t / key_t.norm(dim=-1, keepdim=True)

    res = query_t.size(2) % GROUP_SIZE
    res_k = key_t.size(2) % GROUP_SIZE


    if not res:
        query_t = einops.rearrange(query_t, "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        key_t = einops.rearrange(key_t, "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        value_t = einops.rearrange(value_t, "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        attention_mask0 = einops.rearrange(attention_mask0, "(g c) (g1 c1) -> c c1 g g1", c = GROUP_SIZE, c1 = GROUP_SIZE)
        attention_mask0 = torch.diagonal(attention_mask0, dim1=-1, dim2=-2)
        attention_mask0 = attention_mask0.unsqueeze(0)
        attention_mask0 = einops.rearrange(attention_mask0, "b c c1 dia-> b dia c c1")

        local_out0 = _local_quadratic_attn_p(query_t, key_t, value_t, causal_mask=attention_mask0)
        local_out = einops.rearrange(local_out0, "b h g c d -> b h (g c) d")
        attn_output = local_out
    else:
        query_t_pre = einops.rearrange(query_t[:,:,:-res,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        query_t_res = query_t[:,:,-res:,:]
        
        if res_k == 0:
            res_k = key_t.size(2)

        key_t_pre = einops.rearrange(key_t[:,:,:-res_k,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        key_t_res = key_t[:,:,-res_k:,:]

        value_t_pre = einops.rearrange(value_t[:,:,:-res_k,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        value_t_res = value_t[:,:,-res_k:,:]
        attention_mask_pre = einops.rearrange(attention_mask0[:-res,:-res_k], "(g c) (g1 c1) -> c c1 g g1", c = GROUP_SIZE, c1 = GROUP_SIZE)
        attention_mask_pre = torch.diagonal(attention_mask_pre, dim1=-1, dim2=-2)
        attention_mask_pre = attention_mask_pre.unsqueeze(0)
        attention_mask_pre = einops.rearrange(attention_mask_pre, "b c c1 dia-> b dia c c1")

        attention_mask_res = attention_mask0[-res:, -res_k:]
        attention_mask_res = attention_mask_res.unsqueeze(0)
        '''
        else:
            key_t_pre = einops.rearrange(key_t[:,:,:,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
            key_t_res = key_t[:,:,key_t.size(2):,:]

            value_t_pre = einops.rearrange(value_t[:,:,:,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
            value_t_res = value_t[:,:,key_t.size(2):,:]

            attention_mask_pre = einops.rearrange(attention_mask0[:-res,:], "(g c) (g1 c1) -> c c1 g g1", c = GROUP_SIZE, c1 = GROUP_SIZE)
            attention_mask_pre = torch.diagonal(attention_mask_pre, dim1=-1, dim2=-2)
            attention_mask_pre = attention_mask_pre.unsqueeze(0)
            attention_mask_pre = einops.rearrange(attention_mask_pre, "b c c1 dia-> b dia c c1")
    
            attention_mask_res = attention_mask0[-res:, key_t.size(2):]
            attention_mask_res = attention_mask_res.unsqueeze(0)
        '''
        #print("res: ", res_k, key_t.size(), key_t_res.size())

        #print("size: ",query_t_pre.size(),query_t_res.size(),key_t_pre.size(),key_t_res.size(),\
        #    value_t_pre.size(), value_t_res.size(), attention_mask0.size(), attention_mask_res.size(), query_t_res.size())

        if query_t_pre.size(2) > 0:
            local_out_pre = _local_quadratic_attn_p(query_t_pre, key_t_pre, value_t_pre, causal_mask=attention_mask_pre)
            local_out_pre = einops.rearrange(local_out_pre, "b h g c d -> b h (g c) d")

        local_out_res = _local_quadratic_attn_p_g1(query_t_res, key_t_res, value_t_res, causal_mask=attention_mask_res)
        if query_t_pre.size(2) > 0:
            attn_output = torch.cat([local_out_pre, local_out_res], dim=2)
        else:
            attn_output = local_out_res

    if USE_GLOBAL:
        
        if not res:
            global_out0 = _global_linear_attn_p(query_t, key_t, value_t, causal=True)
            global_out = einops.rearrange(global_out0, "b h g c d -> b h (g c) d")
        else:
            global_out0 = _global_linear_attn_causal(query_t_pre, key_t_pre, value_t_pre, query_t_res, key_t_res, value_t_res)
            global_out = global_out0
        
        if NORM_GLOBAL:
            global_out = global_out / (1e-5 + global_out.norm(dim=-1, keepdim=True))
        #print("global_out: ", global_out.norm())
        if attn_output is None:
            attn_output = global_out * (self.global_factor if GLOBAL_FACTOR else 1.0)
        else:
            attn_output = attn_output * (self.local_factor if LOCAL_FACTOR else 1.0) + global_out * (self.global_factor if GLOBAL_FACTOR else 1.0)

    #print("attn_output0: ", attn_output.norm())
    if NORM_TOGETHER:
        attn_output = attn_output / (1e-5 + attn_output.norm(dim=-1, keepdim=True))
    #print("attn_output: ", attn_output.norm())
    if USE_CASTLING:

        self.dwconv.weight.data *= self.conv_mask
        conv_out = self.dwconv(value_states) 


        attn_output = 0.5 * value_states + 1.0 / math.pi * attn_output 
        attn_output = attn_output / attn_output.norm(dim=-1, keepdim=True)
        attn_output += conv_out

    if ADD_CONV:
        self.dwconv.weight.data *= self.conv_mask
        conv_out = self.dwconv(value_states) 

        attn_output += conv_out

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask

def apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids):
    gather_indices = position_ids[:, :, None, None]  # [bsz, seq_len, 1, 1]
    gather_indices = gather_indices.repeat(
        1, 1, cos_sin[0].shape[1], cos_sin[0].shape[3]
    )
    bsz = gather_indices.shape[0]
    cos, sin = (
        torch.gather(x.transpose(1, 2).repeat(bsz, 1, 1, 1), 1, gather_indices)
        for x in cos_sin
    )
    q, k = ((x * cos) + (rotate_half(x) * sin) for x in (q, k))
    return q, k

def linforward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def linear_forward_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    assert padding_mask is None 
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


    assert attention_mask is not None 

    assert USE_LOCAL
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    import einops
    query_t = query_states
    key_t = key_states 
    value_t = value_states
    query_t = query_t / query_t.norm(dim=-1, keepdim=True)
    key_t = key_t / key_t.norm(dim=-1, keepdim=True)
    res = query_t.size(2) % GROUP_SIZE
    res_k = key_t.size(2) % GROUP_SIZE
    attention_mask0 = torch.where(attention_mask < 0, 0, 1).squeeze(0).squeeze(0)

    if not res:
        query_t = einops.rearrange(query_t, "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        key_t = einops.rearrange(key_t, "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        value_t = einops.rearrange(value_t, "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        attention_mask0 = einops.rearrange(attention_mask0, "(g c) (g1 c1) -> c c1 g g1", c = GROUP_SIZE, c1 = GROUP_SIZE)
        attention_mask0 = torch.diagonal(attention_mask0, dim1=-1, dim2=-2)
        attention_mask0 = attention_mask0.unsqueeze(0)
        attention_mask0 = einops.rearrange(attention_mask0, "b c c1 dia-> b dia c c1")

        local_out0 = _local_quadratic_attn_p(query_t, key_t, value_t, causal_mask=attention_mask0)
        local_out = einops.rearrange(local_out0, "b h g c d -> b h (g c) d")
        attn_output = local_out
    else:
        query_t_pre = einops.rearrange(query_t[:,:,:-res,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        query_t_res = query_t[:,:,-res:,:]

        if res_k == 0:
            res_k = -key_t.size(2)

        key_t_pre = einops.rearrange(key_t[:,:,:-res_k,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        key_t_res = key_t[:,:,-res_k:,:]

        value_t_pre = einops.rearrange(value_t[:,:,:-res_k,:], "b h (g c) d-> b h g c d", c = GROUP_SIZE)
        value_t_res = value_t[:,:,-res_k:,:]

        attention_mask_pre = einops.rearrange(attention_mask0[:-res,:-res_k], "(g c) (g1 c1) -> c c1 g g1", c = GROUP_SIZE, c1 = GROUP_SIZE)
        attention_mask_pre = torch.diagonal(attention_mask_pre, dim1=-1, dim2=-2)
        attention_mask_pre = attention_mask_pre.unsqueeze(0)
        attention_mask_pre = einops.rearrange(attention_mask_pre, "b c c1 dia-> b dia c c1")
    
        attention_mask_res = attention_mask0[-res:, -res_k:]
        attention_mask_res = attention_mask_res.unsqueeze(0)

        #print("size: ",query_t_pre.size(),query_t_res.size(),key_t_pre.size(),key_t_res.size(),\
        #    value_t_pre.size(), value_t_res.size(), attention_mask0.size(), attention_mask_res.size(), query_t_res.size())

        if query_t_pre.size(2) > 0:
            local_out_pre = _local_quadratic_attn_p(query_t_pre, key_t_pre, value_t_pre, causal_mask=attention_mask_pre)
            local_out_pre = einops.rearrange(local_out_pre, "b h g c d -> b h (g c) d")

        local_out_res = _local_quadratic_attn_p_g1(query_t_res, key_t_res, value_t_res, causal_mask=attention_mask_res)
        if query_t_pre.size(2) > 0:
            attn_output = torch.cat([local_out_pre, local_out_res], dim=2)
        else:
            attn_output = local_out_res
    
    if USE_GLOBAL:
        #if not res:
        #print("use global:")
        
        if not res:
            global_out0 = _global_linear_attn_p(query_t, key_t, value_t, causal=True)
            global_out = einops.rearrange(global_out0, "b h g c d -> b h (g c) d")
        else:
            global_out0 = _global_linear_attn_causal(query_t_pre, key_t_pre, value_t_pre, query_t_res, key_t_res, value_t_res)
            global_out = global_out0

        attn_output += global_out * (self.global_factor if GLOBAL_FACTOR else 1.0)

    if ADD_CONV:
        self.dwconv.weight.data *= self.conv_mask
        conv_out = self.dwconv(value_states) 
        #print("conv", self.dwconv.weight.size(), self.conv_mask.size(), value_states.size(), attn_output.size(), conv_out.size(), conv_out[:,:,-1,:].size())
        if False: #correct
            from conv_alter import conv_to_weight
            mreal = conv_to_weight(self.dwconv, value_states.size(2), self.num_heads)
            out_real = mreal.matmul(value_states)
            print((out_real - conv_out).abs().max())

        if conv_out.size(2) != attn_output.size(2): 
            attn_output += conv_out[:,:,-1,:].unsqueeze(-2)
        else:
            attn_output += conv_out

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def forward_flashattn_inference(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()
    kv_heads = getattr(self, "num_key_value_heads", self.num_heads)

    q, k, v = (
        op(hidden_states).view(bsz, q_len, nh, self.head_dim)
        for op, nh in (
            (self.q_proj, self.num_heads),
            (self.k_proj, kv_heads),
            (self.v_proj, kv_heads),
        )
    )
    # shape: (b, s, num_heads, head_dim)

    kv_seq_len = k.shape[1]
    past_kv_len = 0
    if past_key_value is not None:
        past_kv_len = past_key_value[0].shape[2]
        kv_seq_len += past_kv_len

    cos_sin = self.rotary_emb(v, seq_len=kv_seq_len)
    q, k = apply_rotary_pos_emb_inference(q, k, cos_sin, position_ids)

    if past_key_value is not None:
        assert (
            flash_attn_version >= "2.1.0"
        ), "past_key_value support requires flash-attn >= 2.1.0"
        # reuse k, v
        k = torch.cat([past_key_value[0].transpose(1, 2), k], dim=1)
        v = torch.cat([past_key_value[1].transpose(1, 2), v], dim=1)

    past_key_value = (k.transpose(1, 2), v.transpose(1, 2)) if use_cache else None

    if attention_mask is None:
        output = flash_attn_func(q, k, v, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(q, attention_mask[:, -q_len:])
        # We can skip concat and call unpad twice but seems better to call unpad only once.
        kv, _, cu_k_lens, max_k = unpad_input(
            torch.stack((k, v), dim=2), attention_mask
        )
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value

def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


def replace_llama_attn(use_flash_attn=True, use_full=False, inference=False):
    if use_flash_attn:
        assert use_full == False
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        if inference:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_inference
        else:
            transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
                _prepare_decoder_attention_mask
            )
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattn_full if use_full else forward_flashattn
    else:
        if USE_LINEAR:
            if inference:
                #transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_inference
                transformers.models.llama.modeling_llama.LlamaAttention.forward = linear_forward_inference
            else:
                transformers.models.llama.modeling_llama.LlamaAttention.forward = linear_forward_noflashattn
        else:
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noflashattn #forward_noflashattn
