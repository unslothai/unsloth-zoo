# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "HAS_FLEX_ATTENTION",
    "FLEX_ATTENTION_BLOCK_SIZE",
    "_flex_attention",
    "flex_attention",
    "PREFILL_KERNEL_OPTIONS",
    "decode_kernel_options",
    "decode_kernel_kwargs",
    "create_block_mask_cached",
    "create_block_mask",
    "compiled_create_block_mask",
    "reused_compiled_create_block_mask",
    "clear_block_mask_cache",
    "FlexAttentionCache",

    "causal_mask",
    "generate_causal_mask_with_padding",
    "generate_decoding_causal_mask_with_padding",

    "generate_sliding_window_mask",
    "generate_sliding_window_mask_with_padding",
    "generate_decoding_sliding_window_mask_with_padding",
]

import os
import torch
import functools
from unsloth_zoo.temporary_patches.common import torch_compile, _torch_compile
FLEX_ATTENTION_KV_INCREMENT = 512

try:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as FLEX_ATTENTION_BLOCK_SIZE
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    HAS_FLEX_ATTENTION = True
    from torch.nn.attention.flex_attention import _score_mod_signature, _mask_mod_signature

    # Kernel options depend on VRAM (low-memory OOM, pytorch/pytorch issue 133254); read device properties here, not via
    # mem_get_info (pins a context at import); integrated parts may show only the carve-out; unreadable devices must
    # keep raising (cuda_total_memory's 0 reads as 16GB or less).
    from unsloth_zoo.integrated_device import cuda_total_memory
    vram_of_gpu = min(
        cuda_total_memory(i, props = torch.cuda.get_device_properties(i))/1024/1024/1024
        for i in range(torch.cuda.device_count())
    )
    kernel_options = None
    if vram_of_gpu <= 16:
        kernel_options = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 32,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 32,
        }
        _flex_attention = functools.partial(_flex_attention, kernel_options = kernel_options)
    elif vram_of_gpu <= 24:
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        }
        _flex_attention = functools.partial(_flex_attention, kernel_options = kernel_options)
    pass
    flex_attention = _torch_compile(_flex_attention)
    # Prefill/training: AUTO sends queries under 128 rows to flex_decoding, which has no valid config once q_len * GQA
    # group > 128; Dynamo then runs flex eagerly and checkpoint recompute raises CheckpointError.
    PREFILL_KERNEL_OPTIONS = {**(kernel_options or {}), "FORCE_USE_FLEX_ATTENTION": True}

    # torch >= 2.9: return_lse's deprecation warning is a graph break on 2.10; use AuxRequest.
    try:
        from torch.nn.attention.flex_attention import AuxRequest as _AuxRequest
        FLEX_LSE_KWARGS = {"return_aux": _AuxRequest(lse = True)}
    except Exception:
        FLEX_LSE_KWARGS = {"return_lse": True}

    def flex_output_and_lse(output):
        """(attention output, logsumexp) from a flex_attention call made with FLEX_LSE_KWARGS."""
        out, aux = output
        return out, (aux.lse if "return_aux" in FLEX_LSE_KWARGS else aux)

    @functools.lru_cache
    def create_block_mask_cached(mask_mod, M, N, device = "cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        return _create_block_mask(mask_mod, None, None, M, N, device = device)

    @functools.lru_cache
    def create_block_mask(mask_mod, bsz, head, M, N, device = "cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        return _create_block_mask(mask_mod, bsz, head, M, N, device = device)

    # Largest dense mask (bool elements) built eagerly; above it the compiled builder keeps memory O(blocks).
    _EAGER_BLOCK_MASK_MAX_ELEMENTS = 1 << 27

    # Fixed in 2.13 / 2.14; 2.11 / 2.12 untested, so they keep the workaround.
    _ONE_BLOCK_MISCOMPILE = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 13)

    def _compile_block_mask(bsz, head, M, N):
        # torch 2.10's dynamic create_block_mask miscompiles one-block lengths: block counts come out 2 instead of 1 and
        # read past the buffer (illegal memory access).
        blk = FLEX_ATTENTION_BLOCK_SIZE
        if not _ONE_BLOCK_MISCOMPILE or (M > blk and N > blk):
            return True
        # The eager builder pads both lengths to whole blocks before reducing the dense mask.
        padded = -(-M // blk) * blk * (-(-N // blk) * blk)
        return (bsz or 1) * (head or 1) * padded > _EAGER_BLOCK_MASK_MAX_ELEMENTS

    # Keep the eager build out of an enclosing torch.compile graph, which would hit the same miscompile.
    @torch.compiler.disable
    def _eager_create_block_mask(mask_mod, bsz, head, M, N, device):
        return _create_block_mask(mask_mod, bsz, head, M, N, device = device, _compile = False)

    def _build_block_mask(mask_mod, bsz, head, M, N, device):
        if _compile_block_mask(bsz, head, M, N):
            # _compile MUST stay on to reduce VRAM, otherwise usage is O(N^2) (meta-pytorch/attention-gym issue 15).
            return _create_block_mask(mask_mod, bsz, head, M, N, device = device, _compile = True)
        return _eager_create_block_mask(mask_mod, bsz, head, M, N, device)

    def compiled_create_block_mask_cached(mask_mod, M, N, device = "cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        return _build_block_mask(mask_mod, None, None, M, N, device)

    def compiled_create_block_mask(mask_mod, bsz, head, M, N, device = "cuda"):
        """Create block mask for Flex Attention. Assume bsz=any(None), head=any(None)"""
        return _build_block_mask(mask_mod, bsz, head, M, N, device)

    # Block masks for stateless mask_mods (causal / sliding window) only depend on the
    # mask_mod and shapes, so every layer, pass and checkpoint replay can share one.
    # Never pass mask_mods closing over per batch tensors (padding, document ids).
    _BLOCK_MASK_CACHE = {}
    _BLOCK_MASK_CACHE_SIZE = 32

    def clear_block_mask_cache():
        _BLOCK_MASK_CACHE.clear()

    def reused_compiled_create_block_mask(mask_mod, bsz, head, M, N, device = "cuda"):
        """compiled_create_block_mask cached for stateless mask_mods. UNSLOTH_FLEX_MASK_REUSE=0 disables"""
        if os.environ.get("UNSLOTH_FLEX_MASK_REUSE", "1") == "0":
            _BLOCK_MASK_CACHE.clear()   # turning reuse off also frees the retained masks
            return compiled_create_block_mask(mask_mod, bsz, head, M, N, device = device)
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:   # "cuda" means the current device
            dev = torch.device("cuda", torch.cuda.current_device())
        # Inference tensors cannot be saved for backward, so keep inference mode masks apart
        key = (mask_mod, bsz, head, M, N, dev, torch.is_inference_mode_enabled())
        block_mask = _BLOCK_MASK_CACHE.get(key)
        if block_mask is None:
            # Masks grow ~quadratically with length, so only the current shape's masks are kept (one per mask_mod and
            # inference mode, shared by one forward's layers); other shapes are freed first.
            for k in [k for k in _BLOCK_MASK_CACHE if k[1:] != key[1:]]:
                del _BLOCK_MASK_CACHE[k]
            if len(_BLOCK_MASK_CACHE) >= _BLOCK_MASK_CACHE_SIZE:
                _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
            block_mask = compiled_create_block_mask(mask_mod, bsz, head, M, N, device = dev)
            _BLOCK_MASK_CACHE[key] = block_mask
        return block_mask

    def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
        """Causal mask for Flex Attention"""
        return q_idx >= kv_idx

    def generate_causal_mask_with_padding(padding_start_idx = None):
        """
        Causal mask for Flex Attention with left padding support.
        Normal causal mask:
            k0 k1 k2 k3 k4
        q0   X
        q1   X  X
        q2   X  X  X
        q3   X  X  X  X
        q4   X  X  X  X  X
        If we add 2 tokens as padded tokens, we get:
            #0 #1 k2 k3 k4
        #0
        #1
        q2         X
        q3         X  X
        q4         X  X  X
        Assume padding_start_idx == [2]
        """
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1
        def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
            """Causal mask for Flex Attention"""
            q_start =  q_idx >= padding_start_idx[batch_idx]
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return q_start & k_start & (q_idx >= kv_idx)
        causal_mask.__name__ = causal_mask.__doc__ = f"causal_mask_with_left_padding"
        return causal_mask

    def generate_decoding_causal_mask_with_padding(padding_start_idx = None):
        """
        For decoding purposes only. We remove q_padded since decoding attends to 1 q
        Assume padded tokens = 5
            #0 #1 #2 #3 #4 k5 k6
        #0   #
        #1   #  #
        #2   #  #  #
        #3   #  #  #  #
        #4   #  #  #  #  #
        q5   #  #  #  #  #  X
        q6   #  #  #  #  #  X  X
        """
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1
        def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
            """Causal mask for Flex Attention"""
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return k_start & (q_idx >= kv_idx)
        causal_mask.__name__ = causal_mask.__doc__ = f"decoding_causal_mask_with_left_padding"
        return causal_mask

    @functools.lru_cache
    def generate_sliding_window_mask(window_size: int):
        """Sliding window mask for Flex Attention"""
        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            """
            [NOTE] Official PyTorch uses <= window_size which is (window_size+1) tokens
            Ie if window_size = 3, HuggingFace does the below:
            X
            X X
            X X X
              X X X
                X X X
            Whilst official PyTorch does (ie attends to the last 3 and itself)
            X
            X X
            X X X
            X X X X
              X X X X
            """
            windowed_mask = q_idx - kv_idx < window_size
            # Official PyTorch uses q_idx - kv_idx <= window_size, which attends one extra token; this uses strict <.
            return causal_mask & windowed_mask
        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}"
        return sliding_window

    def generate_sliding_window_mask_with_padding(window_size: int, padding_start_idx = None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1
        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            windowed_mask = q_idx - kv_idx < window_size
            q_padded =  q_idx >= padding_start_idx[batch_idx]
            k_padded = kv_idx >= padding_start_idx[batch_idx]
            return q_padded & k_padded & causal_mask & windowed_mask
        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_with_left_padding_{window_size}"
        return sliding_window

    def generate_decoding_sliding_window_mask_with_padding(window_size: int, padding_start_idx = None):
        """SWA decoding ignores padding_start_idx: with KV size equal to the window, a start like 3406 would mask every
        key; padded tokens are always 0, so the generic SWA is returned."""
        return generate_sliding_window_mask(window_size)

    # For inference see https://pytorch.org/blog/flexattention-for-inference
    def get_score_mod_w_offset(score_mod: _score_mod_signature, _offset: torch.tensor):
        def _score_mod(score, b, h, q, kv):
            return score_mod(score, b, h, q + _offset, kv)
        return _score_mod

    def get_mask_mod_w_offset(mask_mod: _mask_mod_signature, _offset: torch.tensor):
        def _mask_mod(b, h, q, kv):
            return mask_mod(b, h, q + _offset, kv)
        return _mask_mod

    # Used for every attention layer
    class FlexAttentionCache:
        __slots__ = \
            "offset", "offset_tensor", "mask_mod_with_offset", "block_mask", "mask_mod", \
            "max_length", "block_size", "sliding_window", "block_mask_slice",

        def __init__(self, key, mask_mod, sliding_window):
            bsz, heads_KV, qlen_KV, dim = key.shape
            if sliding_window is None:
                """
                Normal causal mask:
                    k0 k1 k2 k3 k4
                q0   X
                q1   X  X
                q2   X  X  X
                q3   X  X  X  X
                q4   X  X  X  X  X
                During decoding:
                    k0 k1 k2 k3 k4
                q0
                q1
                q2
                q3
                q4   X  X  X  X  X
                But q_index = 0, so we need an offset = 4
                If no offset, we get:
                   k0 k1 k2 k3 k4
                q0
                q1
                q2
                q3
                q4  X
                Which is wrong. We need q_idx=0 + offset=4 = 0+4 = 4
                Note it's offset==index since it's indexed from 0 as seen in
                https://pytorch.org/blog/flexattention-for-inference/
                https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L91
                """
                # Get next multiple of FLEX_ATTENTION_KV_INCREMENT
                div, mod = divmod(qlen_KV, FLEX_ATTENTION_KV_INCREMENT)
                n = FLEX_ATTENTION_KV_INCREMENT*div + (FLEX_ATTENTION_KV_INCREMENT if mod != 0 else 0)
                self.offset = qlen_KV - 2 # Minus two since we need the block mask to use the saved offset_tensor
                # Minus 2, not -1, since offset is pre-incremented (see self.offset += 1), not post-incremented.
                if self.offset <= -2:
                    # Minimum is -1
                    self.offset = -1
                    # During decoding we do self.offset += 1, so self.offset = 0
                self.sliding_window = None
            else:
                """
                Sliding window of 2 + causal mask:
                    k0 k1 k2 k3 k4
                q0   X
                q1   X  X
                q2      X  X
                q3         X  X
                q4            X  X
                During decoding:
                    k0 k1 k2 k3 k4
                q0
                q1
                q2
                q3
                q4            X  X
                If we set cache_implementation = "static" which we assume, we don't use an offset
                since the K is a rolling matrix of the past window size.
                Ie if sliding_window = 2, K is shape (2, dim). So in actuality, we get:
                    -- -- -- k3 k4
                q0
                q1
                q2
                q3
                q4            X  X
                But since we use a rolling matrix, offset = sliding_window-1 always ie 2-1 = 1
                    -- -- -- k0 k1
                q0
                q1
                q2
                q3(0)
                q4(1)         X  X
                Note it's offset==index since it's indexed from 0 as seen in
                https://pytorch.org/blog/flexattention-for-inference/
                https://github.com/meta-pytorch/gpt-fast/blob/6ecad9b5b6b987d17ac4303965545873d0192086/generate.py#L91
                For sliding window, it's always sliding_window - 1
                since 128 means index 127
                """
                n = sliding_window
                self.offset = min(sliding_window, qlen_KV) - 2 # Minus 2 since block mask is indexing
                if self.offset <= -2:
                    # Minimum is -1
                    self.offset = -1
                    # During decoding we do self.offset += 1, so self.offset = 0
                self.sliding_window = sliding_window - 1 # Minus 1 since token 128 means index 127
            self.offset_tensor = torch.tensor(self.offset, device = key.device, dtype = torch.int32)
            self.block_mask = compiled_create_block_mask(mask_mod, bsz, heads_KV, n, n, device = key.device)
            self.mask_mod = mask_mod
            self.max_length = n
            self.block_size = self.block_mask.BLOCK_SIZE[0]
            self.mask_mod_with_offset = get_mask_mod_w_offset(self.mask_mod, self.offset_tensor)
            self.block_mask_slice = None

        def __call__(self, key):
            bsz, heads_KV, qlen_KV, dim = key.shape
            # Increment beforehand since offset_tensor is the index; e.g. sliding_window = 127 (128 - 1) and offset 126 gives offset + 1 = 127.
            if (self.sliding_window is None) or (self.offset < self.sliding_window):
                self.offset += 1
                self.offset_tensor.add_(1)
            elif (self.sliding_window is not None):
                # Quick return: the sliding window block mask is always the same; reached only once offset >=
                # sliding_window (max sliding window already reached).
                return self.block_mask_slice
            if self.offset >= self.max_length:
                # Must be >= since offset=127 with max_length=128 means size 128, because indexing is zero based.
                self.max_length += FLEX_ATTENTION_KV_INCREMENT
                self.block_mask = compiled_create_block_mask(self.mask_mod, bsz, heads_KV, self.max_length, self.max_length, device = key.device)
                self.block_size = self.block_mask.BLOCK_SIZE[0]
            block_offset = self.offset // self.block_size
            block_mask_slice = self.block_mask[:, :, block_offset]
            block_mask_slice.mask_mod = self.mask_mod_with_offset
            # Must set seq_lengths as in the gpt-fast generate.py (line 80).
            block_mask_slice.seq_lengths = (1, qlen_KV)
            self.block_mask_slice = block_mask_slice
            return block_mask_slice
    pass

except:
    HAS_FLEX_ATTENTION = False
    FLEX_ATTENTION_BLOCK_SIZE = None
    flex_attention = None
    PREFILL_KERNEL_OPTIONS = None
    create_block_mask_cached = None
    causal_mask = None
    generate_sliding_window_mask = None
    FlexAttentionCache = None
    reused_compiled_create_block_mask = None
    clear_block_mask_cache = None
    FLEX_LSE_KWARGS = {"return_lse": True}
    flex_output_and_lse = None
pass


def _triton_block_ptr_removed():
    # Triton 3.9 (and triton-windows 3.9) turned tl.make_block_ptr into a NotImplementedError. Generated kernels call the
    # public triton.language name, so a shim that restores only that one counts as present. Only Triton's own @builtin
    # wrapper is unwrapped: a working shim made with functools.wraps(old) also points __wrapped__ at the removed stub.
    try:
        import inspect
        import triton.language as _tl
        import triton.language.core as _core
    except Exception:
        return False
    fn = getattr(_tl, "make_block_ptr", None)
    if fn is None:
        return True
    core_file = os.path.abspath(getattr(_core, "__file__", "") or "")
    for _ in range(8):
        code = getattr(fn, "__code__", None)
        if code is None:
            return False
        if (code.co_name == "wrapper" and os.path.abspath(code.co_filename) == core_file
                and getattr(fn, "__wrapped__", None) is not None):
            fn = fn.__wrapped__
            continue
        try:
            return _raises_removed_error(inspect.getsource(code))
        except Exception:
            return False
    return False


def _raises_removed_error(source):
    # Triton 3.9's make_block_ptr is a docstring plus one `raise NotImplementedError("... have been removed ...")`. Only
    # that statement counts: a working shim may mention the removal in a comment or docstring.
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    fn = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body
    if len(body) != 1 or not isinstance(body[0], ast.Raise) or body[0].exc is None:
        return False
    return "have been removed" in ast.unparse(body[0].exc)


def _torch_template_uses_block_ptr(*candidates):
    # First candidate group present in the installed torch decides; read the files, no import. True / False once every
    # file of that group was read, None when the answer is unknown (no group found, a member missing or unreadable).
    import importlib.util
    try:
        spec = importlib.util.find_spec("torch")
        roots = list(spec.submodule_search_locations or []) if spec is not None else []
    except Exception:
        return None
    for root in roots:
        for group in candidates:
            paths = [os.path.join(root, *rel.split("/")) for rel in group]
            if not os.path.exists(paths[0]):
                continue
            uses = False
            for path in paths:
                try:
                    with open(path, encoding = "utf-8") as f:
                        if "make_block_ptr" in f.read():
                            uses = True
                except Exception:
                    return None
            return uses
    return None


def _flex_decode_template_uses_block_ptr():
    # torch 2.6 - 2.12's flex_decoding template builds block pointers; 2.13+ does not.
    return _torch_template_uses_block_ptr(
        ("_inductor/kernel/flex/templates/flex_decode.py.jinja",),
        ("_inductor/kernel/flex_decoding.py",),
    )


def _flex_main_template_uses_block_ptr():
    # The main flex kernel, forward and backward: block pointers up to torch 2.8 (non-TMA path), tensor descriptors from
    # 2.9. Every file of the 2.9+ group must be read before it counts as free of block pointers.
    return _torch_template_uses_block_ptr(
        (
            "_inductor/kernel/flex/templates/flex_attention.py.jinja",
            "_inductor/kernel/flex/templates/common.py.jinja",
            "_inductor/kernel/flex/templates/flex_backwards.py.jinja",
            "_inductor/kernel/flex/templates/utilities.py.jinja",
        ),
        ("_inductor/kernel/flex_attention.py",),
    )


def _kernel_options_position():
    # flex_attention(query, key, value, score_mod, block_mask, scale, enable_gqa, return_lse, kernel_options, *, ...)
    try:
        import inspect
        from torch.nn.attention.flex_attention import flex_attention as _torch_flex_attention
        return list(inspect.signature(_torch_flex_attention).parameters).index("kernel_options")
    except Exception:
        return 8


def _main_kernel_options(requested, forced_options):
    """kernel_options that keep a short query off flex_decoding. With no options the compiled call's own memory presets
    apply; an explicit dict replaces them as it does without the guard (a partial tile override must not be mixed with
    preset tiles), so it only gains FORCE_USE_FLEX_ATTENTION. torch 2.9 has no BACKEND and only reads
    FORCE_USE_FLEX_ATTENTION, and 2.10+ rejects BACKEND together with it, so an explicit AUTO or TRITON is replaced by
    the legacy knob (TRITON asks for the main kernel, so it is forced on). Other named backends chose the kernel."""
    if requested is None:
        return dict(forced_options)
    requested = dict(requested)
    backend = requested.get("BACKEND")
    if backend in ("AUTO", "TRITON"):
        del requested["BACKEND"]
        if backend == "TRITON":
            requested["FORCE_USE_FLEX_ATTENTION"] = True
    if "BACKEND" in requested:
        return requested
    return {"FORCE_USE_FLEX_ATTENTION": True, **requested}


def _main_kernel_for_short_queries(flex_attention_fn, forced_options):
    """flex_attention_fn with the main kernel for queries AUTO would send to flex_decoding (see _main_kernel_options).
    kernel_options may come positionally or by keyword. No functools.wraps: copying the compiled function's __dict__
    would hand Dynamo its original callable and skip these options."""
    forced_options = dict(forced_options)
    position = _kernel_options_position()
    def flex_attention(*args, **kwargs):
        if len(args) > position:
            if "kernel_options" in kwargs:
                return flex_attention_fn(*args, **kwargs)  # both given: flex_attention raises its own TypeError
            kwargs["kernel_options"] = args[position]
            args = args[:position] + args[position + 1:]
        kwargs["kernel_options"] = _main_kernel_options(kwargs.get("kernel_options"), forced_options)
        return flex_attention_fn(*args, **kwargs)
    return flex_attention


def _flex_decoding_unavailable(block_ptr_removed, decode_uses_block_ptr, main_uses_block_ptr):
    # An unknown (None) template answer never turns the workaround on.
    return bool(block_ptr_removed) and decode_uses_block_ptr is True and main_uses_block_ptr is False


# Decided at import (no file reads or logging inside traced decode code). When torch's flex_decoding cannot compile
# against the installed Triton, the failed compile makes Dynamo skip flex_attention for the rest of the process and
# every later call (prefill too) builds the full score matrix; short queries take the main flex kernel instead, which
# only helps when that kernel is known not to need block pointers too (torch 2.9+).
_TRITON_BLOCK_PTR_REMOVED = bool(HAS_FLEX_ATTENTION) and _triton_block_ptr_removed()
_DECODE_USES_BLOCK_PTR = _flex_decode_template_uses_block_ptr() if _TRITON_BLOCK_PTR_REMOVED else None
_MAIN_USES_BLOCK_PTR = _flex_main_template_uses_block_ptr() if _TRITON_BLOCK_PTR_REMOVED else None
FLEX_DECODING_UNAVAILABLE = _flex_decoding_unavailable(_TRITON_BLOCK_PTR_REMOVED, _DECODE_USES_BLOCK_PTR, _MAIN_USES_BLOCK_PTR)
if _TRITON_BLOCK_PTR_REMOVED and (_DECODE_USES_BLOCK_PTR is True or _MAIN_USES_BLOCK_PTR is True):
    import triton
    from unsloth_zoo.log import logger
    if FLEX_DECODING_UNAVAILABLE:
        logger.warning(
            f"Unsloth: Triton {triton.__version__} has no block pointers, which torch {torch.__version__}'s "
            "flex_decoding needs; short queries use the main flex kernel instead. Install the Triton your torch pins "
            "to get flex_decoding back."
        )
    elif _MAIN_USES_BLOCK_PTR is True:
        logger.warning(
            f"Unsloth: Triton {triton.__version__} has no block pointers, which torch {torch.__version__}'s flex "
            "attention kernels need, so flex attention cannot compile and runs eagerly with much more memory. "
            "Install the Triton your torch pins."
        )
    else:
        logger.warning(
            f"Unsloth: Triton {triton.__version__} has no block pointers, which torch {torch.__version__}'s "
            "flex_decoding needs, and its main flex kernel could not be checked, so flex attention may run eagerly "
            "with much more memory. Install the Triton your torch pins."
        )
def _guarded_exports(compiled_flex_attention, raw_flex_attention, forced_options):
    """Both exported callables with the main kernel for short queries. The uncompiled one reaches the same kernel
    choice inside a caller's torch.compile."""
    return (
        _main_kernel_for_short_queries(compiled_flex_attention, forced_options),
        _main_kernel_for_short_queries(raw_flex_attention, forced_options),
    )


if FLEX_DECODING_UNAVAILABLE and flex_attention is not None:
    flex_attention, _flex_attention = _guarded_exports(flex_attention, _flex_attention, PREFILL_KERNEL_OPTIONS)


def decode_kernel_options():
    """kernel_options for a one-token query: AUTO (flex_decoding) unless it cannot compile here."""
    return dict(PREFILL_KERNEL_OPTIONS) if FLEX_DECODING_UNAVAILABLE else {}


def decode_kernel_kwargs():
    """Extra flex_attention kwargs for a query AUTO may send to flex_decoding: none (the compiled flex_attention keeps
    its own tile options) unless flex_decoding cannot compile here."""
    return {"kernel_options": dict(PREFILL_KERNEL_OPTIONS)} if FLEX_DECODING_UNAVAILABLE else {}
