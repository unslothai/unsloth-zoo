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

"""One-token flex decoding shared by every model: a BlockMask CUDA-graph decode can refill, and the attention call.

The decode BlockMask lists, per row, the KV blocks up to the query's own slot (bucketize of a device offset, no host
read) and leaves every block partial, so mask_mod decides each key. mask_mod reads only 0-d device offsets and a
fixed [B, kv_length] padding tensor (one column per KV slot, slot j holding position kv_offset + j), so a rolling
sliding window keeps every shape. The tensors and a rebuild function are published as
`_unsloth_graph_closure`, and what mask_function computes as `_unsloth_mask_identity`: Unsloth's decode graphs refill
the tensors in place and key the graph on the identity.

Sliding windows follow transformers and the official gpt-oss implementation: a query at position q sees keys
q - W < k <= q, W tokens including itself. Attention sinks enter the softmax denominator only:
out * sigmoid(lse - sink) == sum exp(z) v / (exp(sink) + sum exp(z)), computed in float32. A row with no visible key
returns exact zeros.

`unsloth_flex_decode` is registered as a transformers attention and mask implementation: one-token steps get this
BlockMask and flex_decoding, everything else (prefill, a layer flex_decoding cannot run) the SDPA path."""

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
    # A copy: StaticLayer passes its live cumulative_length, bumped in place before attention reads the mask.
    return int(x) if not isinstance(x, torch.Tensor) else x.to(torch.int64).reshape(()).clone()


def decode_padding(attention_mask, kv_length, kv_offset):
    """[B, kv_length] bool: slot j is a real token of row b. Read from the 2D mask at position kv_offset + j; positions
    past the mask (unwritten static slots) are invisible. A tensor offset is never read on the host."""
    full = attention_mask.to(dtype = torch.bool)
    width, kv_length = full.shape[-1], int(kv_length)
    if not isinstance(kv_offset, torch.Tensor) and kv_offset >= 0:
        # A host offset: a copy of the columns, no gather kernels. Not a view: its row stride would be the 2D mask's
        # width, which grows every step, and a decode graph keys on strides (re-capture every step).
        columns = full[:, kv_offset : kv_offset + kv_length]
        padding = full.new_zeros((full.shape[0], kv_length))
        padding[:, : columns.shape[-1]].copy_(columns)
        return padding
    position = torch.arange(kv_length, device = full.device) + (
        kv_offset.to(full.device) if isinstance(kv_offset, torch.Tensor) else kv_offset
    )
    return full[:, position.clamp(max = width - 1)] & (position < width)


def _block_starts(n, block, device):
    """[min, block, 2 * block, ...]: bucketize(d, right = True) counts the starts <= d, so it is clamp(d // block + 1, 1, n)."""
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
    """The one-token BlockMask on `device` (offsets: python ints or 0-d int64 tensors; padding: [B, kv_length] bool or
    None). With an identity, publishes the graph contract. gpt-oss's mask builder calls this for its decode steps."""
    from torch.nn.attention.flex_attention import BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE as BLOCK
    q_o, kv_o = (
        x.to(device) if isinstance(x, torch.Tensor) else torch.full((), x, dtype = torch.int64, device = device)
        for x in (q_off, kv_off)
    )
    pad = padding.to(device) if padding is not None else None
    mask_mod = _make_mask_mod(mask_function, q_o, kv_o, pad)
    # Blocks are partial and mask_mod decides per key; for causal, only blocks up to the query's own slot can hold a
    # visible key, so an unwritten static-cache tail is not listed (no host sync).
    n = (kv_length + BLOCK - 1) // BLOCK
    rows = batch_size or 1
    if isinstance(q_off, torch.Tensor) or isinstance(kv_off, torch.Tensor):
        # One kernel: clamp((q - kv_off) // BLOCK + 1, 1, n) as a bucket count, written contiguous int32.
        d = q_o if isinstance(kv_off, int) and kv_off == 0 else q_o - kv_o
        kv_num_blocks = torch.bucketize(d.expand(rows, 1, 1), _block_starts(n, BLOCK, device),
                                        out_int32 = True, right = True)
    else:
        # Integer offsets (dynamic caches): count on the host, one fill.
        kv_num_blocks = torch.full((rows, 1, 1), min(max((q_off - kv_off) // BLOCK + 1, 1), n),
                                   dtype = torch.int32, device = device)
    kv_indices = torch.arange(n, dtype = torch.int32, device = device).view(1, 1, 1, n).expand(rows, 1, 1, n)
    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks, kv_indices.contiguous(), BLOCK_SIZE = BLOCK, mask_mod = mask_mod,
        seq_lengths = (1, kv_length),
        **({"compute_q_blocks": torch.is_grad_enabled()} if _HAS_COMPUTE_Q_BLOCKS else {}),
    )
    if identity is not None:
        # CUDA-graph decode refills these tensors and rebuilds mask_mod over its own copies; every value mask_mod
        # reads besides them is a constant of mask_function, named by the identity (part of the graph key as a
        # BlockMask field).
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
    """BlockMask for a one-token decode step over `kv_length` KV slots (slot j holds position kv_offset + j; the query
    is at q_offset). mask_function defaults to causal (sliding when `window`); a 2D attention_mask adds per-slot
    padding. With graph_contract, a mask_function the identity can name publishes the refill contract; one it cannot
    name (a captured tensor, dict or object) gets a plain BlockMask (no `_unsloth_mask_identity`)."""
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
    """[B, 1, 1, kv_length] bool from a decode BlockMask's mask_mod (every slot evaluated, no block skipping): for a
    layer that cannot run flex_decoding. Fixed shapes, no host reads."""
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
    # flex_decoding keeps a [BLOCK_N, head_dim] K and V tile per stage in shared memory: head_dim 512 needs the larger
    # shared memory of compute capability 8.0+ (on a T4 no tile size compiles at 512).
    return 512 if capability >= (8, 0) else 256


@functools.lru_cache(maxsize = 256)
def _supported(head_dim, n_q, n_kv, dtype, device_type, index):
    if not (HAS_FLEX_ATTENTION and _HAS_SEQ_LENGTHS) or device_type != "cuda":
        return False
    if n_kv <= 0 or n_q % n_kv != 0:
        return False
    groups = n_q // n_kv
    # flex_decoding packs the GQA group into BLOCK_M: a group size that is not a power of two silently falls back to
    # the main flex kernel (torch/_inductor/kernel/flex/flex_decoding.py).
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
    """BLOCK_N for one-stage flex_decoding: the K and V tiles ([BLOCK_N, head_dim] each) plus the query tile must fit
    shared memory. Below compute capability 8.0 (64 KB, T4) head_dim 256 needs 32 rows (the default 64 has no valid
    config; 512 has none even at 16 rows, so flex_decode_supported stops at 256 there); on 8.0+ head_dim 512 is faster
    with 32 rows (A100 and L4 measured 1.6-1.9x)."""
    if capability < (8, 0):
        return 32 if head_dim > 128 else None
    return 32 if head_dim > 256 else None


def flex_decode_kernel_options(head_dim, device = None):
    """Decode kernel options: flex_decoding's defaults, with a smaller one-stage KV tile where shared memory needs it."""
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
    """(flex_attention, compiled): compiled flex on GPUs (flex_decoding); the eager reference elsewhere (CPU Inductor
    cannot lower the LSE)."""
    from .utils import flex_attention
    if device.type == "cuda" and flex_attention is not None:
        return flex_attention, True
    from torch.nn.attention.flex_attention import flex_attention as eager_flex_attention
    return eager_flex_attention, False


class FlexDecodeUnavailable(RuntimeError):
    """flex_decoding could not be compiled for this layer shape (for example no kernel config fits shared memory)."""


# Layer shapes whose first compiled call failed, and those already compiled once. unsloth_zoo sets Dynamo's
# suppress_errors, which would turn a failed flex compile into a silent eager flex: unfused (several times slower than
# SDPA) and not capturable by CUDA graphs. The first call per shape compiles with errors raised instead, and a shape
# that fails is reported as unavailable so callers keep the dense path.
_FAILED_SHAPES = set()
_COMPILED_SHAPES = set()


def _shape_key(query, key, kernel_options, softcap):
    return (
        query.shape[-1], query.shape[1], key.shape[1], query.dtype, query.device, bool(softcap),
        tuple(sorted((k, repr(v)) for k, v in kernel_options.items())),
    )


def flex_decode_attention(query, key, value, block_mask, scale = None, *, sinks = None, softcap = None, kernel_options = None):
    """One-token attention: query [B, Hq, 1, D], key / value [B, Hkv, S, D] (GQA when Hq != Hkv) -> [B, Hq, 1, D].
    sinks: [Hq] learned logits in the denominator only. softcap: tanh capping of the scores before the mask. A row
    with no visible key returns zeros. Never writes its inputs."""
    q_len, kv_len = block_mask.seq_lengths
    if key.shape[-2] > kv_len:
        # Static cache slots past the mask.
        key, value = key[:, :, :kv_len], value[:, :, :kv_len]
    elif key.shape[-2] < kv_len:
        block_mask = block_mask._adjust(q_len, key.shape[-2])
    if not torch.compiler.is_compiling():
        # Inductor picks flex_decoding only for a static batch: a second batch size would make it dynamic.
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
    if shape is None or shape in _COMPILED_SHAPES:
        out, lse = call()
    elif shape in _FAILED_SHAPES:
        raise FlexDecodeUnavailable("Unsloth: flex_decoding failed to compile for this layer shape.")
    else:
        try:
            with torch._dynamo.config.patch(suppress_errors = False):
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
        _COMPILED_SHAPES.add(shape)
    compute = torch.promote_types(out.dtype, torch.float32)
    lse = lse.to(compute)
    empty = torch.isneginf(lse).unsqueeze(-1)
    if sinks is not None:
        # exp(lse) / (exp(lse) + exp(sink)), at least float32; 0 for an empty row.
        scale_rows = torch.sigmoid(lse - sinks.to(compute).view(1, -1, 1)).unsqueeze(-1)
        out = (out.to(compute) * scale_rows).to(out.dtype)
    return torch.where(empty, torch.zeros((), dtype = out.dtype, device = out.device), out)


def _eager_attention(query, key, value, mask, scale, sinks = None, softcap = None):
    """float32 reference / fallback: query [B, Hq, Lq, D], key / value [B, Hkv, S, D], mask bool broadcastable to
    [B, Hq, Lq, S] (True = visible)."""
    groups = query.shape[1] // key.shape[1]
    compute = torch.promote_types(query.dtype, torch.float32)
    k = key.to(compute).repeat_interleave(groups, dim = 1)
    v = value.to(compute).repeat_interleave(groups, dim = 1)
    scores = torch.matmul(query.to(compute), k.transpose(-1, -2)) * scale
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
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
    """transformers mask interface (4.57 and 5.x signatures): a decode BlockMask for one-token steps on a GPU, the
    SDPA mask for everything else."""
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
    """transformers attention interface: query [B, Hq, L, D], key / value [B, Hkv, S, D] -> ([B, L, Hq, D], None).
    flex_decoding for one-token steps whose mask is a decode BlockMask and whose shape flex_decoding serves; SDPA (or
    the float32 reference for sinks / softcap) otherwise."""
    is_block_mask = _BlockMask is not None and isinstance(attention_mask, _BlockMask)
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])
    if (
        is_block_mask and query.shape[-2] == 1
        and flex_decode_supported(query.shape[-1], query.shape[1], key.shape[1], query.dtype, query.device)
    ):
        try:
            out = flex_decode_attention(query, key, value, attention_mask, scaling, sinks = s_aux, softcap = softcap)
            return out.transpose(1, 2).contiguous(), None
        except FlexDecodeUnavailable:
            pass
    if is_block_mask:
        # A layer flex_decoding does not serve (head_dim, GQA, dtype): the same visibility as a dense mask.
        attention_mask = dense_decode_mask(attention_mask, key.shape[-2], query.device)
    if softcap or s_aux is not None:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask > torch.finfo(attention_mask.dtype).min / 2
        L, S = query.shape[-2], key.shape[-2]
        if attention_mask is None and L > 1 and kwargs.get("is_causal", getattr(module, "is_causal", True)):
            # The SDPA mask interface elides an unpadded causal mask (SDPA's is_causal covers it); this reference has
            # no such flag, so causality is built here: query i sees keys up to S - L + i.
            attention_mask = torch.ones(L, S, dtype = torch.bool, device = query.device).tril(S - L)
        out = _eager_attention(query, key, value, attention_mask, scaling, sinks = s_aux, softcap = softcap)
        return out.transpose(1, 2).contiguous(), None
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling = scaling, dropout = dropout, **kwargs)


_REGISTERED = False


def register_flex_decode():
    """Registers FLEX_DECODE_KEY with transformers' attention and mask interfaces (idempotent). A custom key is not
    gated by a model's _supports_flex_attn. Returns False when this transformers has no interfaces to register in."""
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
    AttentionInterface.register(FLEX_DECODE_KEY, unsloth_flex_decode_attention)
    AttentionMaskInterface.register(FLEX_DECODE_KEY, unsloth_flex_decode_mask)
    _REGISTERED = True
    return True
