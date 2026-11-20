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

"""One-token flex decoding: a BlockMask CUDA-graph decode can refill in place, and the attention call.

mask_mod reads only 0-d device offsets and a fixed [B, kv_length] padding tensor (slot j = position kv_offset + j), published
as `_unsloth_graph_closure`; `_unsloth_mask_identity` names mask_function and keys the graph. Sliding window: q - W < k <= q.
Sinks enter the softmax denominator only (float32). A row with no visible key returns zeros."""

__all__ = [
    "FLEX_DECODE_KEY",
    "causal_window_mask_function",
    "decode_padding",
    "decode_block_mask",
    "build_decode_block_mask",
    "dense_decode_mask",
    "flex_decode_attention",
    "flex_decode_supported",
    "flex_decode_kernel_options",
    "FlexDecodeUnavailable",
    "unsloth_flex_decode_mask",
    "unsloth_flex_decode_attention",
    "register_flex_decode",
]

import contextlib
import functools
import math
import os
import weakref

import torch

from .utils import decode_kernel_options

try:
    from torch.nn.attention.flex_attention import BlockMask as _BlockMask
    import inspect as _inspect
    _FROM_KV_BLOCKS_ARGS = _inspect.signature(_BlockMask.from_kv_blocks).parameters
    _HAS_COMPUTE_Q_BLOCKS = "compute_q_blocks" in _FROM_KV_BLOCKS_ARGS
    _HAS_SEQ_LENGTHS = "seq_lengths" in _FROM_KV_BLOCKS_ARGS
    del _FROM_KV_BLOCKS_ARGS
except Exception:
    _BlockMask = None
    _HAS_COMPUTE_Q_BLOCKS = _HAS_SEQ_LENGTHS = False
HAS_FLEX_ATTENTION = _BlockMask is not None

try:
    # torch >= 2.9: return_lse is deprecated (and its warning is a graph break on 2.10).
    from torch.nn.attention.flex_attention import AuxRequest as _AuxRequest
    _LSE_KWARGS = {"return_aux": _AuxRequest(lse = True)}
except Exception:
    _LSE_KWARGS = {"return_lse": True}


def _output_and_lse(output):
    out, aux = output
    return out, (aux.lse if "return_aux" in _LSE_KWARGS else aux)

FLEX_DECODE_KEY = "unsloth_flex_decode"


def _causal(batch_idx, head_idx, q_idx, kv_idx):
    return kv_idx <= q_idx


def causal_window_mask_function(window = None):
    """Causal mask_function, with a sliding window of `window` tokens including the query (q - W < k <= q)."""
    if not window:
        return _causal
    window = int(window)

    def sliding_causal(batch_idx, head_idx, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (kv_idx > q_idx - window)
    return sliding_causal


def _offset(x):
    return int(x) if not isinstance(x, torch.Tensor) else x.to(torch.int64).reshape(()).clone()


def decode_padding(attention_mask, kv_length, kv_offset):
    """[B, kv_length] bool padding (slot j = mask column kv_offset + j); slots past the mask are invisible."""
    full = attention_mask.to(dtype = torch.bool)
    width, kv_length = full.shape[-1], int(kv_length)
    if not isinstance(kv_offset, torch.Tensor) and kv_offset >= 0:
        # Copy, not a view: a view's stride grows every step and decode graphs key on strides.
        columns = full[:, kv_offset : kv_offset + kv_length]
        padding = full.new_zeros((full.shape[0], kv_length))
        padding[:, : columns.shape[-1]].copy_(columns)
        return padding
    position = torch.arange(kv_length, device = full.device) + (
        kv_offset.to(full.device) if isinstance(kv_offset, torch.Tensor) else kv_offset
    )
    return full[:, position.clamp(max = width - 1)] & (position < width)


def _block_starts(n, block, device):
    """[min, block, 2 * block, ...]: bucketize(d, right = True) counts the starts <= d, so it is clamp(d // block + 1, 1, n).
    Cached: a fresh build per mask costs a host-to-device scalar copy every decode step. Never written."""
    return _cached_block_starts(int(n), int(block), torch.device(device))


@functools.lru_cache(maxsize = 64)
def _cached_block_starts(n, block, device):
    starts = torch.arange(n, dtype = torch.int64, device = device) * block
    starts[0] = torch.iinfo(torch.int64).min
    return starts


def _make_mask_mod(mask_function, q_o, kv_o, pad = None):
    if pad is not None:
        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            return mask_function(batch_idx, head_idx, q_idx + q_o, kv_idx + kv_o) & pad[batch_idx, kv_idx]
    else:
        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            return mask_function(batch_idx, head_idx, q_idx + q_o, kv_idx + kv_o)
    return mask_mod


def build_decode_block_mask(device, batch_size, kv_length, q_off, kv_off, padding, mask_function, identity = None):
    """The one-token BlockMask on `device`; with an identity, publishes the graph contract. Used by gpt-oss too."""
    from torch.nn.attention.flex_attention import BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE as BLOCK
    q_o, kv_o = (
        x.to(device) if isinstance(x, torch.Tensor) else torch.full((), x, dtype = torch.int64, device = device)
        for x in (q_off, kv_off)
    )
    pad = padding.to(device) if padding is not None else None
    mask_mod = _make_mask_mod(mask_function, q_o, kv_o, pad)
    # Only blocks up to the query's slot are listed (no host sync); mask_mod decides per key.
    n = (kv_length + BLOCK - 1) // BLOCK
    rows = batch_size or 1
    if isinstance(q_off, torch.Tensor) or isinstance(kv_off, torch.Tensor):
        d = q_o if isinstance(kv_off, int) and kv_off == 0 else q_o - kv_o
        kv_num_blocks = torch.bucketize(d.expand(rows, 1, 1), _block_starts(n, BLOCK, device),
                                        out_int32 = True, right = True)
    else:
        kv_num_blocks = torch.full((rows, 1, 1), min(max((q_off - kv_off) // BLOCK + 1, 1), n),
                                   dtype = torch.int32, device = device)
    kv_indices = torch.arange(n, dtype = torch.int32, device = device).view(1, 1, 1, n).expand(rows, 1, 1, n)
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks, kv_indices.contiguous(), BLOCK_SIZE = BLOCK, mask_mod = mask_mod,
        seq_lengths = (1, kv_length),
        **({"compute_q_blocks": torch.is_grad_enabled()} if _HAS_COMPUTE_Q_BLOCKS else {}),
    )
    if identity is not None:
        # Graph decode refills these; everything else mask_mod reads is named by the identity (the graph key).
        tensors = (q_o, kv_o) if pad is None else (q_o, kv_o, pad)
        block_mask._unsloth_graph_closure = (tensors, lambda ts: _make_mask_mod(mask_function, *ts))
        block_mask._unsloth_mask_identity = identity
    return block_mask


def _mask_identity(mask_function):
    from unsloth_zoo.temporary_patches.gpt_oss import _mask_function_identity
    return _mask_function_identity(mask_function)


@torch.compiler.disable
def decode_block_mask(
    batch_size,
    kv_length,
    q_offset,
    kv_offset = 0,
    *,
    mask_function = None,
    attention_mask = None,
    window = None,
    device = None,
    graph_contract = True,
):
    """One-token decode BlockMask over `kv_length` slots. A mask_function the identity cannot name gets a plain
    BlockMask (no `_unsloth_mask_identity`)."""
    if mask_function is None:
        mask_function = causal_window_mask_function(window)
    elif window:
        raise ValueError("Unsloth: pass either mask_function or window, not both.")
    if device is None:
        device = next(
            (x.device for x in (q_offset, kv_offset, attention_mask) if isinstance(x, torch.Tensor)), torch.device("cpu"),
        )
    device = torch.device(device)
    q_off, kv_off = _offset(q_offset), _offset(kv_offset)
    kv_length = int(kv_length)
    padding = None
    if attention_mask is not None and attention_mask.dim() == 2:
        padding = decode_padding(attention_mask, kv_length, kv_off)
    identity = _mask_identity(mask_function) if graph_contract else None

    def build(target):
        return build_decode_block_mask(target, batch_size, kv_length, q_off, kv_off, padding, mask_function, identity)

    block_mask = build(device)
    # BlockMask.to() does not move mask_mod's captured tensors: rebuild per device. Weakref avoids a cycle.
    this, masks = weakref.ref(block_mask), {}

    def on_device(target):
        target = torch.device(target)
        if target == device:
            return this()
        if target not in masks:
            masks[target] = build(target)
        return masks[target]
    block_mask._unsloth_on_device = on_device
    return block_mask


def dense_decode_mask(block_mask, kv_length = None, device = None):
    """[B, 1, 1, kv_length] bool from a decode BlockMask's mask_mod, for layers flex_decoding cannot run."""
    mask_mod = block_mask.mask_mod
    rows = block_mask.kv_num_blocks.shape[0]
    kv_length = int(kv_length if kv_length is not None else block_mask.seq_lengths[1])
    device = device if device is not None else block_mask.kv_num_blocks.device
    b = torch.arange(rows, device = device).view(rows, 1, 1, 1)
    zero = torch.zeros((1, 1, 1, 1), dtype = torch.int64, device = device)
    kv = torch.arange(kv_length, device = device).view(1, 1, 1, kv_length)
    return mask_mod(b, zero, zero, kv).expand(rows, 1, 1, kv_length)


@functools.lru_cache(maxsize = None)
def _device_capability(index):
    try:
        return torch.cuda.get_device_capability(index)
    except Exception:
        return (0, 0)


def _flex_decode_hd_limit(capability):
    # head_dim 512 needs sm80+ shared memory (no tile size compiles on a T4).
    return 512 if capability >= (8, 0) else 256


@functools.lru_cache(maxsize = 256)
def _supported(head_dim, n_q, n_kv, dtype, device_type, index):
    if not (HAS_FLEX_ATTENTION and _HAS_SEQ_LENGTHS) or device_type != "cuda":
        return False
    if n_kv <= 0 or n_q % n_kv != 0:
        return False
    groups = n_q // n_kv
    # Non power-of-two GQA groups silently fall back to the main flex kernel (torch/_inductor/kernel/flex/flex_decoding.py).
    if groups & (groups - 1):
        return False
    capability = _device_capability(index)
    if dtype == torch.bfloat16 and capability < (8, 0):
        return False
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return 16 <= head_dim <= _flex_decode_hd_limit(capability)


def flex_decode_supported(head_dim, n_q, n_kv, dtype, device):
    """Whether flex_decoding serves this layer shape on this device. UNSLOTH_FLEX_DECODE=0 turns it off."""
    if os.environ.get("UNSLOTH_FLEX_DECODE", "1") == "0":
        return False
    device = torch.device(device)
    index = device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else -1)
    return _supported(int(head_dim), int(n_q), int(n_kv), dtype, device.type, index)


def _kv_tile(head_dim, capability):
    """BLOCK_N fitting shared memory: <sm80 head_dim 256 needs 32 (default 64 has no config); sm80+ head_dim 512 is
    1.6-1.9x faster at 32 (A100, L4)."""
    if capability < (8, 0):
        return 32 if head_dim > 128 else None
    return 32 if head_dim > 256 else None


def flex_decode_kernel_options(head_dim, device = None):
    """flex_decoding defaults with a smaller KV tile where shared memory needs it."""
    options = dict(decode_kernel_options())
    if options.get("FORCE_USE_FLEX_ATTENTION", False):
        return options
    device = torch.device(device) if device is not None else None
    capability = (8, 0)
    if device is not None and device.type == "cuda":
        capability = _device_capability(device.index if device.index is not None else torch.cuda.current_device())
    tile = _kv_tile(int(head_dim), capability)
    if tile is not None:
        options.setdefault("BLOCK_N", tile)
        options.setdefault("num_stages", 1)
    return options


@functools.lru_cache(maxsize = 64)
def _softcap_score_mod(softcap):
    softcap = float(softcap)

    def softcap_score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        return softcap * torch.tanh(score / softcap)
    return softcap_score_mod


def _flex_function(device):
    """(flex_attention, compiled): eager off GPU (CPU Inductor cannot lower the LSE)."""
    from .utils import flex_attention
    if device.type == "cuda" and flex_attention is not None:
        return flex_attention, True
    from torch.nn.attention.flex_attention import flex_attention as eager_flex_attention
    return eager_flex_attention, False


class FlexDecodeUnavailable(RuntimeError):
    """flex_decoding could not be compiled for this layer shape."""


# Dynamo suppress_errors would turn a failed compile into silent slow eager flex: the first call per shape raises
# instead, and failed shapes are reported unavailable so callers keep the dense path.
_FAILED_SHAPES = set()


def _strict_compile():
    # AMD's ROCm torch 2.11 wheels have no suppress_errors: nothing to turn off there.
    config = torch._dynamo.config
    return config.patch(suppress_errors = False) if hasattr(config, "suppress_errors") else contextlib.nullcontext()


def _shape_key(query, key, kernel_options, softcap):
    # Every dimension a compile specialises on (batch is static, see below): a later specialisation must still go
    # through the failure handling below, not the cached-success path.
    return (
        query.shape[0], key.shape[-2], query.shape[-1], query.shape[1], key.shape[1], query.dtype, query.device,
        bool(softcap),
        tuple(sorted((k, repr(v)) for k, v in kernel_options.items())),
    )


def flex_decode_attention(query, key, value, block_mask, scale = None, *, sinks = None, softcap = None, kernel_options = None):
    """query [B, Hq, 1, D], key / value [B, Hkv, S, D] -> [B, Hq, 1, D]. Never writes its inputs."""
    q_len, kv_len = block_mask.seq_lengths
    if key.shape[-2] > kv_len:
        key, value = key[:, :, :kv_len], value[:, :, :kv_len]
    elif key.shape[-2] < kv_len:
        block_mask = block_mask._adjust(q_len, key.shape[-2])
    if not torch.compiler.is_compiling():
        # flex_decoding needs a static batch.
        for x in (query, key, value):
            torch._dynamo.mark_static(x, 0)
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if kernel_options is None:
        kernel_options = flex_decode_kernel_options(query.shape[-1], query.device)
    function, compiled = _flex_function(query.device)

    def call():
        return _output_and_lse(function(
            query,
            key,
            value,
            block_mask = block_mask,
            score_mod = _softcap_score_mod(softcap) if softcap else None,
            enable_gqa = query.shape[1] != key.shape[1],
            scale = scale,
            kernel_options = kernel_options,
            **_LSE_KWARGS,
        ))
    shape = _shape_key(query, key, kernel_options, softcap) if compiled and not torch.compiler.is_compiling() else None
    if shape is None:
        out, lse = call()
    elif shape in _FAILED_SHAPES:
        raise FlexDecodeUnavailable("Unsloth: flex_decoding failed to compile for this layer shape.")
    else:
        # Every call keeps the guard: scale, strides or the mask closure can still trigger a new specialisation,
        # and a suppressed failure would silently run eager flex instead of falling back.
        try:
            with _strict_compile():
                out, lse = call()
        except Exception as error:
            # A failure inside a CUDA graph capture is the caller's (compile before capturing); not a shape verdict.
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise
            _FAILED_SHAPES.add(shape)
            raise FlexDecodeUnavailable(
                f"Unsloth: flex_decoding failed to compile for head_dim {query.shape[-1]}, {query.shape[1]} / "
                f"{key.shape[1]} heads, {query.dtype}: {str(error)[:200]}"
            ) from error
    compute = torch.promote_types(out.dtype, torch.float32)
    lse = lse.to(compute)
    empty = torch.isneginf(lse).unsqueeze(-1)
    if sinks is not None:
        scale_rows = torch.sigmoid(lse - sinks.to(compute).view(1, -1, 1)).unsqueeze(-1)
        out = (out.to(compute) * scale_rows).to(out.dtype)
    return torch.where(empty, torch.zeros((), dtype = out.dtype, device = out.device), out)


def _eager_attention(query, key, value, mask, scale, sinks = None, softcap = None, dropout = 0.0):
    """float32 reference / fallback; mask broadcastable to [B, Hq, Lq, S]: bool (True = visible) or additive float."""
    groups = query.shape[1] // key.shape[1]
    compute = torch.promote_types(query.dtype, torch.float32)
    k = key.to(compute).repeat_interleave(groups, dim = 1)
    v = value.to(compute).repeat_interleave(groups, dim = 1)
    scores = torch.matmul(query.to(compute), k.transpose(-1, -2)) * scale
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.to(scores.dtype)
    if sinks is not None:
        sink = sinks.to(compute).view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
        scores = torch.cat([scores, sink], dim = -1)
    peak = scores.amax(dim = -1, keepdim = True)
    peak = torch.where(torch.isneginf(peak), torch.zeros_like(peak), peak)
    weights = torch.exp(scores - peak)
    total = weights.sum(dim = -1, keepdim = True)
    weights = torch.where(total > 0, weights / total, torch.zeros_like(weights))
    if sinks is not None:
        weights = weights[..., :-1]
    if dropout:
        weights = torch.nn.functional.dropout(weights, p = dropout, training = True)
    return torch.matmul(weights, v).to(query.dtype)


def _call_mask_interface(name, kwargs):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    return ALL_MASK_ATTENTION_FUNCTIONS[name](**kwargs)


def unsloth_flex_decode_mask(
    batch_size,
    q_length = None,
    kv_length = None,
    q_offset = 0,
    kv_offset = 0,
    mask_function = None,
    attention_mask = None,
    device = None,
    cache_position = None,
    **kwargs,
):
    """transformers mask interface (4.57 and 5.x): decode BlockMask for one-token GPU steps, else the SDPA mask."""
    passed = dict(
        batch_size = batch_size, q_length = q_length, kv_length = kv_length, q_offset = q_offset,
        kv_offset = kv_offset, mask_function = mask_function, attention_mask = attention_mask, device = device,
        cache_position = cache_position, **kwargs,
    )
    passed = {k: v for k, v in passed.items() if v is not None or k in ("attention_mask",)}
    q_len = q_length if q_length is not None else (cache_position.shape[0] if cache_position is not None else None)
    q_off = q_offset if q_length is not None else (cache_position[0] if cache_position is not None else q_offset)
    target = device if device is not None else (
        cache_position.device if cache_position is not None else (attention_mask.device if attention_mask is not None else None)
    )
    if (
        q_len == 1 and kv_length is not None and target is not None and torch.device(target).type == "cuda"
        and _HAS_SEQ_LENGTHS and HAS_FLEX_ATTENTION and os.environ.get("UNSLOTH_FLEX_DECODE", "1") != "0"
        and mask_function is not None
    ):
        return decode_block_mask(
            batch_size, kv_length, q_off, kv_offset, mask_function = mask_function, attention_mask = attention_mask,
            device = target,
        )
    return _call_mask_interface("sdpa", passed)


def unsloth_flex_decode_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling = None,
    dropout = 0.0,
    softcap = None,
    s_aux = None,
    **kwargs,
):
    """transformers attention interface: flex_decoding where served, else SDPA (float32 reference for sinks / softcap)."""
    is_block_mask = _BlockMask is not None and isinstance(attention_mask, _BlockMask)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if not getattr(module, "training", False):
        dropout = 0.0
    if (
        is_block_mask and query.shape[-2] == 1 and not dropout
        and flex_decode_supported(query.shape[-1], query.shape[1], key.shape[1], query.dtype, query.device)
    ):
        try:
            out = flex_decode_attention(query, key, value, attention_mask, scaling, sinks = s_aux, softcap = softcap)
            return out.transpose(1, 2).contiguous(), None
        except FlexDecodeUnavailable:
            pass
    if is_block_mask:
        kv_len = attention_mask.seq_lengths[1]
        if key.shape[-2] > kv_len:
            # As the flex path: slots past the mask's KV length are not part of this step.
            key, value = key[:, :, :kv_len], value[:, :, :kv_len]
        attention_mask = dense_decode_mask(attention_mask, key.shape[-2], query.device)
    if softcap or s_aux is not None:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Additive mask: keep finite penalties, the dtype-min fill means masked (-inf, so an empty row is zeros).
            hidden = attention_mask <= torch.finfo(attention_mask.dtype).min / 2
            attention_mask = attention_mask.float().masked_fill(hidden, float("-inf"))
        L, S = query.shape[-2], key.shape[-2]
        if attention_mask is None and L > 1 and kwargs.get("is_causal", getattr(module, "is_causal", True)):
            # SDPA elides an unpadded causal mask (is_causal); the reference needs it built explicitly.
            attention_mask = torch.ones(L, S, dtype = torch.bool, device = query.device).tril()
        out = _eager_attention(
            query, key, value, attention_mask, scaling, sinks = s_aux, softcap = softcap, dropout = dropout
        )
        return out.transpose(1, 2).contiguous(), None
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling = scaling, dropout = dropout, **kwargs)


def install_block_mask_transfer():
    """BlockMask.to rebuilds our decode masks on the target device (plain .to leaves mask_mod's captures behind);
    .contiguous() is a no-op (transformers >= 5.21 generate calls it per mask)."""
    if _BlockMask is None:
        return False
    if not hasattr(_BlockMask, "contiguous"):
        _BlockMask.contiguous = lambda self, *args, **kwargs: self
    if not getattr(_BlockMask.to, "_unsloth_on_device_to", False):
        original = _BlockMask.to

        def to(self, device, *args, **kwargs):
            on_device = getattr(self, "_unsloth_on_device", None)
            if on_device is not None and isinstance(device, (str, int, torch.device)):
                return on_device(device)
            return original(self, device, *args, **kwargs)

        to._unsloth_on_device_to = True
        to._unsloth_gpt_oss = True  # the marker gpt-oss's own registration checks
        _BlockMask.to = to
    return True


_REGISTERED = False


def register_flex_decode():
    """Registers FLEX_DECODE_KEY (not gated by _supports_flex_attn); False if transformers has no interfaces."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if not (HAS_FLEX_ATTENTION and _HAS_SEQ_LENGTHS):
        return False
    try:
        from transformers.modeling_utils import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
    except Exception:
        return False
    install_block_mask_transfer()
    AttentionInterface.register(FLEX_DECODE_KEY, unsloth_flex_decode_attention)
    AttentionMaskInterface.register(FLEX_DECODE_KEY, unsloth_flex_decode_mask)
    _REGISTERED = True
    return True
