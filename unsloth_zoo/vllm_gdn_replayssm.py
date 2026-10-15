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

"""ReplaySSM decode for Gated DeltaNet (Qwen3.5 / Qwen3.6 / Qwen3-Next) on released vLLM.

Upstream vLLM ships ReplaySSM for Mamba2 only. This backports the GDN half from
vllm-project/vllm#48792 (closed unmerged when its fork was deleted) as an idempotent
runtime patch: the decode kernel (vendored under ``_vendored/vllm``), the three extra
ring tensors in the GDN state page, the per-row ring cursor / flush flag in the GDN
metadata builder, and the decode dispatch in ``QwenGatedDeltaNetAttention``.

Every piece is matched against the installed vLLM's source first; any mismatch leaves
vLLM untouched and ``patch_vllm_gdn_replayssm`` returns False, so the caller simply
does not request ReplaySSM.
"""

__all__ = ["GDN_REPLAYSSM_ARCHITECTURES", "patch_vllm_gdn_replayssm"]

import dataclasses
import inspect
import textwrap

GDN_REPLAYSSM_ARCHITECTURES = (
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3NextForCausalLM",
)
# Classes that define the class-level GDN state hooks (subclasses inherit them).
_MODEL_HOOK_CLASSES = (
    ("qwen3_5", "Qwen3_5ForCausalLMBase"),
    ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    ("qwen3_next", "Qwen3NextForCausalLM"),
)


def _model_modules():
    from vllm.model_executor.models import qwen3_5, qwen3_next
    return {"qwen3_5": qwen3_5, "qwen3_next": qwen3_next}

# (method, exact source fragment in vLLM, replacement). A fragment that is not found
# exactly once aborts the whole patch.
_LAYER_SOURCE_EDITS = {
    "_fused_gdn_decode_unsupported_reason": (
        (
            "conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()\n",
            "conv_state_dtype, recurrent_state_dtype = self.get_state_dtype()[:2]\n",
        ),
    ),
    "_warmup_prefill_kernels": (
        (
            "_, state_dtype = self.get_state_dtype()\n",
            "state_dtype = self.get_state_dtype()[1]\n",
        ),
    ),
    "_forward_core": (
        (
            "if (\n"
            "    self.enable_packed_recurrent_decode\n"
            "    and attn_metadata.spec_sequence_masks is None\n"
            "    and attn_metadata.num_prefills == 0\n"
            "    and attn_metadata.num_decodes > 0\n"
            "):\n",
            "if (\n"
            "    getattr(self, \"use_replayssm\", False)\n"
            "    and attn_metadata.spec_sequence_masks is None\n"
            "    and attn_metadata.num_prefills == 0\n"
            "    and attn_metadata.num_decodes > 0\n"
            "):\n"
            "    return self._forward_core_decode_non_spec_replayssm(\n"
            "        mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out, attn_metadata=attn_metadata,\n"
            "    )\n"
            "if (\n"
            "    self.enable_packed_recurrent_decode\n"
            "    and attn_metadata.spec_sequence_masks is None\n"
            "    and attn_metadata.num_prefills == 0\n"
            "    and attn_metadata.num_decodes > 0\n"
            "):\n",
        ),
        (
            # Decode rows of a mixed prefill + decode batch must go through the ring
            # too, or the conventional update would advance a state the ring replays.
            "if split_non_spec:\n"
            "    query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(\n",
            "if split_non_spec and getattr(self, \"use_replayssm\", False):\n"
            "    core_attn_out_decode = self._replayssm_decode_rows(\n"
            "        mixed_qkv_non_spec[:num_decode_tokens], a[:num_decode_tokens], b[:num_decode_tokens],\n"
            "        non_spec_state_indices_tensor[: attn_metadata.num_decodes], attn_metadata,\n"
            "    ).transpose(0, 1)\n"
            "elif split_non_spec:\n"
            "    query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(\n",
        ),
    ),
}


def _gdn_ring_shapes(base_shapes, num_k_heads, tp_size, buffer_len):
    # (d_cache, k_cache, g_cache) after (conv, ssm); ssm is (local_v_heads, head_v_dim, head_k_dim).
    local_v_heads, head_v_dim, head_k_dim = base_shapes[1]
    local_k_heads = num_k_heads // tp_size
    return (
        *base_shapes,
        (local_v_heads, buffer_len, head_v_dim),
        (local_k_heads, buffer_len, head_k_dim),
        (local_v_heads, buffer_len),
    )


def _gdn_ring_dtypes(base_dtypes, model_dtype):
    # d / k are kept in fp16 for bf16 activations (3 more mantissa bits, same size).
    import torch
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype

    activation = get_kv_cache_torch_dtype("auto", model_dtype)
    cache = torch.float16 if activation == torch.bfloat16 else activation
    return (*base_dtypes, cache, cache, torch.float32)


def _apply_edit(source, old, new):
    # Replace `old` (written at depth 0) where it appears exactly once as whole lines at
    # some indentation depth; None if no depth gives exactly one match.
    for depth in range(8):
        pad = " " * (4 * depth)
        old_i = "\n" + textwrap.indent(old, pad)
        if source.count(old_i) == 1:
            return source.replace(old_i, "\n" + textwrap.indent(new, pad))
    return None


def _edit_method(cls, name, edits, namespace):
    source = textwrap.dedent(inspect.getsource(cls.__dict__[name]))
    for old, new in edits:
        source = _apply_edit(source, old, new)
        if source is None:
            return None
    scope = {}
    exec(compile(source, f"<unsloth gdn replayssm: {cls.__name__}.{name}>", "exec"), namespace, scope)
    return scope[name]


def _replayssm_decode_rows(self, mixed_qkv, a, b, state_indices, attn_metadata, out = None):
    # Ring decode for one token per row into `out` (rows, 1, local_v_heads, head_v_dim).
    import torch
    from unsloth_zoo._vendored.vllm.gdn_replayssm_decode import fused_recurrent_gated_delta_rule_replayssm
    kv_cache = self.kv_cache
    if out is None:
        out = torch.empty(
            mixed_qkv.shape[0], 1, self.num_v_heads // self.tp_size, self.head_v_dim,
            dtype = mixed_qkv.dtype, device = mixed_qkv.device,
        )
    fused_recurrent_gated_delta_rule_replayssm(
        mixed_qkv = mixed_qkv.contiguous(),
        a = a,
        b = b,
        A_log = self.A_log,
        dt_bias = self.dt_bias,
        scale = self.head_k_dim ** -0.5,
        initial_state = kv_cache[1],
        d_cache = kv_cache[2],
        k_cache = kv_cache[3],
        g_cache = kv_cache[4],
        out = out,
        ssm_state_indices = state_indices,
        write_pos = attn_metadata.write_pos_d,
        is_flush = attn_metadata.is_flush_d,
        use_qk_l2norm_in_kernel = True,
    )
    return out


def _make_decode_non_spec(layer_module):
    causal_conv1d_update = layer_module.causal_conv1d_update
    is_conv_state_dim_first = layer_module.is_conv_state_dim_first

    def _forward_core_decode_non_spec_replayssm(self, mixed_qkv, b, a, core_attn_out, attn_metadata):
        # Mirrors vLLM's _forward_core_decode_non_spec with the ring kernel in place of
        # the packed recurrent decode.
        state_indices = attn_metadata.non_spec_state_indices_tensor
        kv_cache = self.kv_cache
        conv_state = kv_cache[0] if is_conv_state_dim_first() else kv_cache[0].transpose(-1, -2)
        num_actual_tokens = attn_metadata.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices = state_indices[:num_actual_tokens],
            validate_data = False,
        )
        self._replayssm_decode_rows(
            mixed_qkv, a[:num_actual_tokens], b[:num_actual_tokens],
            state_indices[:num_actual_tokens], attn_metadata,
            out = core_attn_out[:num_actual_tokens].unsqueeze(1),
        )

    return _forward_core_decode_non_spec_replayssm


def _ring_cursor(builder, m, num_decodes, query_start_loc_cpu):
    # write_pos = decode steps since the ring's last checkpoint write, mod the buffer length
    # (vllm-project/vllm#48792, GDNAttentionMetadataBuilder.build).
    import torch
    decode_base_cpu = m.replayssm_decode_base_cpu
    seq_lens_cpu = m.seq_lens_cpu_upper_bound
    if decode_base_cpu is None or seq_lens_cpu is None:
        raise ValueError(
            "--use-replayssm requires exact CPU sequence lengths and decode-base counts "
            "to derive decode write positions"
        )
    buffer_len = builder._unsloth_replayssm_buffer_len
    query_lens_cpu = query_start_loc_cpu[1 : num_decodes + 1] - query_start_loc_cpu[:num_decodes]
    num_computed = seq_lens_cpu[:num_decodes] - query_lens_cpu
    base = decode_base_cpu[:num_decodes]
    align_mode = builder.vllm_config.cache_config.mamba_cache_mode == "align"
    block_size = builder.kv_cache_spec.block_size
    if align_mode:
        # A prefill chunk may end mid-block, so decode_base wins until decode crosses the boundary.
        base = torch.maximum(base, (num_computed // block_size) * block_size)
    steps = num_computed - base
    valid = query_lens_cpu > 0
    # A single-token prefill row replayed as decode: flush straight off the checkpoint.
    leftover_prompt = valid & (steps < 0)
    steps = torch.where(valid & ~leftover_prompt, steps, torch.zeros_like(steps))
    write_pos = torch.remainder(steps, buffer_len)
    is_flush = (write_pos == buffer_len - 1) | leftover_prompt
    if align_mode:
        # Materialize the state on the step that completes a block, for prefix caching.
        is_flush = is_flush | (valid & ((num_computed + query_lens_cpu) % block_size == 0))
    return write_pos.to(torch.int32), is_flush.to(torch.int8)


def _patch_builder(builder_module):
    import torch
    from vllm.utils.torch_utils import async_tensor_h2d
    Builder = builder_module.GDNAttentionMetadataBuilder
    Metadata = builder_module.GDNAttentionMetadata

    @dataclasses.dataclass
    class GDNReplaySSMAttentionMetadata(Metadata):
        # Per decode row: ring write position and flush flag (None unless decoding).
        write_pos_d: "torch.Tensor | None" = None
        is_flush_d: "torch.Tensor | None" = None

    init = Builder.__init__
    build = Builder.build

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs):
        init(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs)
        cache_config = vllm_config.cache_config
        self._unsloth_replayssm = bool(
            getattr(cache_config, "use_replayssm", False)
            and not getattr(cache_config, "use_kda_recoverssm", False)
        )
        if not self._unsloth_replayssm:
            return
        from vllm.config.mamba import MambaBackendEnum
        mamba_config = vllm_config.mamba_config
        if getattr(mamba_config, "backend", MambaBackendEnum.TRITON) != MambaBackendEnum.TRITON:
            raise ValueError("GDN ReplaySSM requires --mamba-backend triton")
        if getattr(mamba_config, "enable_stochastic_rounding", False):
            raise ValueError("GDN ReplaySSM does not support Mamba cache stochastic rounding")
        self._unsloth_replayssm_buffer_len = cache_config.replayssm_buffer_len
        self._unsloth_write_pos = torch.zeros((self.decode_cudagraph_max_bs,), dtype = torch.int32, device = device)
        self._unsloth_is_flush = torch.zeros((self.decode_cudagraph_max_bs,), dtype = torch.int8, device = device)

    def build_(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        md = build(self, common_prefix_len, common_attn_metadata, *args, **kwargs)
        if not getattr(self, "_unsloth_replayssm", False):
            return md
        write_pos = is_flush = None
        num_decodes = md.num_decodes
        if md.spec_sequence_masks is None and num_decodes > 0:
            m = common_attn_metadata
            write_pos_cpu, is_flush_cpu = _ring_cursor(self, m, num_decodes, m.query_start_loc_cpu)
            indices = md.non_spec_state_indices_tensor
            if indices is not None and indices.data_ptr() == self.non_spec_state_indices_tensor.data_ptr():
                # Decode staged into the persistent CUDA-graph buffers: keep fixed addresses.
                rows = indices.shape[0]
                self._unsloth_write_pos[:num_decodes].copy_(write_pos_cpu, non_blocking = True)
                self._unsloth_is_flush[:num_decodes].copy_(is_flush_cpu, non_blocking = True)
                write_pos = self._unsloth_write_pos[:rows]
                is_flush = self._unsloth_is_flush[:rows]
                # Padded rows point at the null slot and return early; never read.
                write_pos[num_decodes:].fill_(0)
                is_flush[num_decodes:].fill_(0)
            else:
                device = m.query_start_loc.device
                write_pos = async_tensor_h2d(write_pos_cpu.tolist(), dtype = torch.int32, device = device)
                is_flush = async_tensor_h2d(is_flush_cpu.tolist(), dtype = torch.int8, device = device)
        fields = {f.name: getattr(md, f.name) for f in dataclasses.fields(md)}
        fields.update(write_pos_d = write_pos, is_flush_d = is_flush)
        return GDNReplaySSMAttentionMetadata(**fields)

    Builder.__init__ = __init__
    Builder.build = build_
    Builder._unsloth_replayssm_patched = True


def _patch_model_hooks(model_cls):
    shape_fn = model_cls.__dict__["get_mamba_state_shape_from_config"].__func__
    dtype_fn = model_cls.__dict__["get_mamba_state_dtype_from_config"].__func__

    def get_mamba_state_shape_from_config(cls, vllm_config):
        shapes = shape_fn(cls, vllm_config)
        cache_config = vllm_config.cache_config
        if not getattr(cache_config, "use_replayssm", False) or len(shapes) != 2:
            return shapes
        hf_config = vllm_config.model_config.hf_text_config
        return _gdn_ring_shapes(
            shapes, hf_config.linear_num_key_heads,
            vllm_config.parallel_config.tensor_parallel_size, cache_config.replayssm_buffer_len,
        )

    def get_mamba_state_dtype_from_config(cls, vllm_config):
        dtypes = dtype_fn(cls, vllm_config)
        if not getattr(vllm_config.cache_config, "use_replayssm", False) or len(dtypes) != 2:
            return dtypes
        return _gdn_ring_dtypes(dtypes, vllm_config.model_config.dtype)

    model_cls.get_mamba_state_shape_from_config = classmethod(get_mamba_state_shape_from_config)
    model_cls.get_mamba_state_dtype_from_config = classmethod(get_mamba_state_dtype_from_config)


def _patch_layer(layer_module):
    Layer = layer_module.QwenGatedDeltaNetAttention
    edited = {}
    for name, edits in _LAYER_SOURCE_EDITS.items():
        fn = _edit_method(Layer, name, edits, layer_module.__dict__)
        if fn is None:
            return False
        edited[name] = fn
    shape_fn = Layer.get_state_shape
    dtype_fn = Layer.get_state_dtype
    init = Layer.__init__

    def get_state_shape(self):
        shapes = shape_fn(self)
        cache_config = self.cache_config
        if not getattr(cache_config, "use_replayssm", False) or len(shapes) != 2:
            return shapes
        return _gdn_ring_shapes(shapes, self.num_k_heads, self.tp_size, cache_config.replayssm_buffer_len)

    def get_state_dtype(self):
        dtypes = dtype_fn(self)
        if not getattr(self.cache_config, "use_replayssm", False) or len(dtypes) != 2:
            return dtypes
        return _gdn_ring_dtypes(dtypes, self.model_config.dtype)

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.use_replayssm = bool(getattr(self.cache_config, "use_replayssm", False))

    for name, fn in edited.items():
        setattr(Layer, name, fn)
    Layer.get_state_shape = get_state_shape
    Layer.get_state_dtype = get_state_dtype
    Layer.__init__ = __init__
    Layer._replayssm_decode_rows = _replayssm_decode_rows
    Layer._forward_core_decode_non_spec_replayssm = _make_decode_non_spec(layer_module)
    Layer._unsloth_replayssm_patched = True
    return True


def patch_vllm_gdn_replayssm():
    """Install GDN ReplaySSM into the imported vLLM. Idempotent; False leaves vLLM untouched."""
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as layer_module
        from vllm.v1.attention import backend as backend_module
        from vllm.v1.attention.backends import gdn_attn as builder_module
        Layer = layer_module.QwenGatedDeltaNetAttention
        Builder = builder_module.GDNAttentionMetadataBuilder
    except Exception:
        return False
    if getattr(Layer, "_unsloth_replayssm_patched", False) and getattr(Builder, "_unsloth_replayssm_patched", False):
        return True
    try:
        required = (
            hasattr(layer_module, "causal_conv1d_update")
            and hasattr(layer_module, "is_conv_state_dim_first")
            and "replayssm_decode_base_cpu" in {
                f.name for f in dataclasses.fields(backend_module.CommonAttentionMetadata)
            }
            and all(
                n in Layer.__dict__ for n in (*_LAYER_SOURCE_EDITS, "get_state_shape")
            )
        )
        if not required:
            return False
        modules = _model_modules()
        model_classes = [getattr(modules[m], c) for m, c in _MODEL_HOOK_CLASSES]
        for model_cls in model_classes:
            for hook in ("get_mamba_state_shape_from_config", "get_mamba_state_dtype_from_config"):
                if hook not in model_cls.__dict__:
                    return False
        # Dry-run the source edits before touching anything.
        for name, edits in _LAYER_SOURCE_EDITS.items():
            if _edit_method(Layer, name, edits, layer_module.__dict__) is None:
                return False
        from unsloth_zoo._vendored.vllm.gdn_replayssm_decode import fused_recurrent_gated_delta_rule_replayssm  # noqa: F401
    except Exception:
        return False
    if not _patch_layer(layer_module):
        return False
    _patch_builder(builder_module)
    for model_cls in model_classes:
        _patch_model_hooks(model_cls)
    return True
