# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Pre torch 2.14 mem-efficient rejects enable_gqa and flash rejects fp32, so GQA SDPA falls to math; repeat k/v instead,
# unless SDPA would run the enable_gqa call on cuDNN, which takes GQA natively.

import torch

__all__ = []

try:
    from .misc import _flash_sdpa_accepts
except Exception:
    _flash_sdpa_accepts = None


def _flash_takes_gqa(query, key, attn_mask, dropout_p):
    if attn_mask is not None or _flash_sdpa_accepts is None or not torch.backends.cuda.flash_sdp_enabled():
        return False
    # SDPA is on the autocast cast list, so autocast decides the kernel dtype.
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    else:
        dtype = query.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    return _flash_sdpa_accepts(
        query.device, dtype, query.shape[1], key.shape[1], query.shape[-1], dropout_p, query.requires_grad,
    )


def _efficient_takes(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
    # The eligibility checks read the current device's properties, so ask on the query's card.
    if query.is_cuda and query.device.index is not None and query.device.index != torch.cuda.current_device():
        with torch.cuda.device(query.device):
            return _efficient_takes_here(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
    return _efficient_takes_here(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)


def _efficient_takes_here(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
    try:
        try:
            params = torch.backends.cuda.SDPAParams(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
        except TypeError:
            # torch 2.4: six-argument SDPAParams.
            if enable_gqa:
                return False
            params = torch.backends.cuda.SDPAParams(query, key, value, attn_mask, dropout_p, is_causal)
        return torch.backends.cuda.can_use_efficient_attention(params, False) is True
    except Exception:
        return False


def _torch_version_at_least(major, minor):
    try:
        return tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (major, minor)
    except Exception:
        return False


# torch 2.8 added the GQA head count check to the cuDNN SDPA gate; 2.5 - 2.7 pass mismatched heads unchecked.
_CUDNN_GATE_CHECKS_GQA = _torch_version_at_least(2, 8)


def _cudnn_takes_gqa(query, key, value, attn_mask, dropout_p, is_causal, scale):
    # cuDNN takes GQA natively: when SDPA itself would pick cuDNN for the enable_gqa call, repeating k/v only costs memory.
    if not _CUDNN_GATE_CHECKS_GQA:
        return False
    try:
        if not torch.backends.cuda.cudnn_sdp_enabled():
            return False
        from torch.nn.attention import SDPBackend
        with torch.cuda.device(query.device):
            choice = torch._fused_sdp_choice(
                query, key, value, attn_mask, dropout_p, is_causal, scale = scale, enable_gqa = True,
            )
        return choice == int(SDPBackend.CUDNN_ATTENTION)
    except Exception:
        return False


def make_gqa_sdpa(scaled_dot_product_attention):
    @torch.compiler.disable(recursive = True)
    def gqa_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, groups):
        if groups != 1 and query.is_cuda and torch.backends.cuda.mem_efficient_sdp_enabled() and \
            not _flash_takes_gqa(query, key, attn_mask, dropout_p) and \
            not _efficient_takes(query, key, value, attn_mask, dropout_p, is_causal, True) and \
            not _cudnn_takes_gqa(query, key, value, attn_mask, dropout_p, is_causal, scale):
            b, h_kv, s_kv, d = key.shape
            key_r = key[:, :, None].expand(b, h_kv, groups, s_kv, d).reshape(b, h_kv * groups, s_kv, d)
            value_r = value[:, :, None].expand(b, h_kv, groups, s_kv, value.shape[-1]).reshape(b, h_kv * groups, s_kv, value.shape[-1])
            if _efficient_takes(query, key_r, value_r, attn_mask, dropout_p, is_causal, False):
                return scaled_dot_product_attention(
                    query, key_r, value_r, attn_mask = attn_mask, dropout_p = dropout_p, is_causal = is_causal, scale = scale,
                )
            # Rejected: free the repeated copies before the fallback runs (math GQA repeats k/v itself).
            del key_r, value_r
        return scaled_dot_product_attention(
            query, key, value, attn_mask = attn_mask, dropout_p = dropout_p, is_causal = is_causal, scale = scale,
            enable_gqa = groups != 1,
        )
    return gqa_sdpa
