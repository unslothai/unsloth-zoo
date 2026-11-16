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

"""Qwen4Exp (Qwen3.8-Flash-Next) patches: MoE uses the Qwen3.5-MoE grouped-GEMM backend; QSA indexer skips its per-query
nonzero/topk loop when kv_length < budget + compress_ratio; UNSLOTH_QWEN4_EXP_FAST_QSA=0 disables. From the budget up
(training / prefill without a cache), QSA attention selects its blocks vectorised and attends to the selected tokens
only (``unsloth_zoo.sparse_attention``), instead of a per-query Python loop plus an O(S^2) masked dense attention;
UNSLOTH_QWEN4_EXP_SPARSE_ATTN=0 disables. Where that path has no kernel (T4 / sm75, CPU) the dense reference
attention stays, but its indexer builds the selection mask vectorised too (UNSLOTH_QWEN4_EXP_VECTORISED_QSA=0
keeps transformers' per-query loop). The ~102 GB PLE n-gram table is streamed from the checkpoint by
``unsloth_zoo.frozen_table_store`` unless UNSLOTH_FROZEN_TABLE=gpu."""

__all__ = ["patch_qwen4_exp", "patch_qwen4_exp_frozen_table"]

import os

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, logger

def _fast_qsa_enabled():
    return os.environ.get("UNSLOTH_QWEN4_EXP_FAST_QSA", "1") != "0"


def _vectorised_qsa_enabled():
    return os.environ.get("UNSLOTH_QWEN4_EXP_VECTORISED_QSA", "1") != "0"


QWEN4_EXP_INDEXER_STATS = {"vectorised": 0}


_reference_qsa_forward = None
# Whether the reference converts its selection back to an additive float mask for a float (eager) attention mask:
# transformers up to 5.22 does; 5.23 always returns the boolean selection and combines it in the attention layer.
_qsa_returns_float_mask = True


def qwen4_exp_qsa_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
    # Compiler copies this source into its cache: use only `torch`, `self`, the lazy import.
    if (
        attention_mask is not None
        and attention_mask.dim() == 4
        and attention_mask.shape[-1] < self.token_budget + self.compress_ratio
    ):
        if past_key_values is not None:
            batch_size, seq_length, _ = hidden_states.shape
            token_k = self.index_qk_proj(hidden_states)[..., self.index_n_heads * self.index_head_dim:]
            raw_keys = token_k.reshape(batch_size, seq_length, -1, self.index_head_dim).squeeze(2)
            past_key_values.update_indexer(raw_keys, self.layer_idx)
        if attention_mask.dtype == torch.bool:
            return attention_mask
        from unsloth_zoo.temporary_patches.qwen4_exp import _qsa_returns_float_mask
        if not _qsa_returns_float_mask:
            return attention_mask == 0
        return torch.where(
            attention_mask == 0, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min
        )
    if past_key_values is None:
        from unsloth_zoo.temporary_patches.qwen4_exp import _qwen4_exp_vectorised_indexer_mask
        selected_token_mask = _qwen4_exp_vectorised_indexer_mask(self, hidden_states, position_embeddings, attention_mask)
        if selected_token_mask is not None:
            return selected_token_mask
    from unsloth_zoo.temporary_patches.qwen4_exp import qwen4_exp_qsa_indexer_forward_guarded
    return qwen4_exp_qsa_indexer_forward_guarded(self, hidden_states, position_embeddings, attention_mask, past_key_values)


@torch.compiler.disable
def _qwen4_exp_vectorised_indexer_mask(indexer, hidden_states, position_embeddings, attention_mask):
    """transformers' indexer mask (no cache) from the vectorised block selection, or None to run its loop.

    The reference loops over every (batch, query) pair with a host sync each (nonzero, topk): 4k packed tokens are
    4096 iterations per QSA layer, forward and gradient checkpoint recompute, which dominated a T4 training step.
    Token identity with the loop, ties included: rows whose k-th and (k+1)-th block scores tie are recomputed the
    reference's way (``qsa_select_tokens(exact_ties=True)``), unless UNSLOTH_QWEN4_EXP_EXACT_TIES=0."""
    if not _vectorised_qsa_enabled():
        return None
    if (
        not isinstance(attention_mask, torch.Tensor)
        or attention_mask.dim() != 4
        or attention_mask.shape[1] != 1
        or attention_mask.shape[-1] != hidden_states.shape[1]
    ):
        return None
    import sys
    from unsloth_zoo.sparse_attention import qsa_select_tokens
    rope = getattr(sys.modules.get(type(indexer).__module__), "apply_rotary_pos_emb", None)
    if rope is None:   # a subclass defined elsewhere
        from transformers.models.qwen4_exp import modeling_qwen4_exp
        rope = modeling_qwen4_exp.apply_rotary_pos_emb
    with torch.no_grad():
        selected = qsa_select_tokens(indexer, hidden_states, tuple(position_embeddings), attention_mask, rope,
                                     exact_ties=os.environ.get("UNSLOTH_QWEN4_EXP_EXACT_TIES", "1") != "0")
    if selected is None:
        return None
    kv_length = attention_mask.shape[-1]
    # As the reference: every -1 is scattered to an extra last slot, which is dropped.
    selected_token_mask = torch.zeros(
        (*selected.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool
    )
    scatter_indices = torch.where(selected >= 0, selected.long(), kv_length)
    selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
    if not torch.compiler.is_compiling():
        QWEN4_EXP_INDEXER_STATS["vectorised"] += 1
    if attention_mask.is_floating_point() and _qsa_returns_float_mask:
        return torch.where(selected_token_mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
    return selected_token_mask


def qwen4_exp_qsa_indexer_forward_guarded(self, hidden_states, position_embeddings, attention_mask,
                                          past_key_values=None):
    """transformers' indexer on the filled cache slots only; installed alone under UNSLOTH_QWEN4_EXP_FAST_QSA=0.

    The reference reads the position table by kv slot, which has one row per token seen so far. A static cache's
    mask spans max_cache_len, and a query row that attends to nothing (a left pad) is given every slot by Unsloth's
    sdpa_mask fix, unfilled ones included: the reference then indexes past the table (a CUDA device assert in
    batched generate). Real tokens never see an unfilled slot, so hiding them changes no row that is read."""
    # Compiler copies this source into its cache: use only `torch`, `self`, the arguments, the lazy import.
    if attention_mask is not None and attention_mask.dim() == 4:
        n_positions = position_embeddings[0].shape[-2]
        if attention_mask.shape[-1] > n_positions:
            filled = torch.arange(attention_mask.shape[-1], device=attention_mask.device) < n_positions
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & filled
            elif attention_mask.is_floating_point():
                attention_mask = attention_mask.masked_fill(~filled, torch.finfo(attention_mask.dtype).min)
    from unsloth_zoo.temporary_patches.qwen4_exp import _reference_qsa_forward
    return _reference_qsa_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)


_reference_attention_forward = None
QWEN4_EXP_SPARSE_STATS = {"sparse": 0, "reference": 0}


def _sparse_attn_enabled():
    return os.environ.get("UNSLOTH_QWEN4_EXP_SPARSE_ATTN", "1") != "0"


def _qwen4_exp_sparse_attention(module, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs):
    """transformers' Qwen4ExpTextAttention.forward (5.22 / 5.23) on the selected tokens only, or None to run it."""
    indexer = module.indexer
    # The configured default counts too (transformers resolves output_attentions from the config).
    output_attentions = kwargs.get("output_attentions")
    if output_attentions is None:
        output_attentions = getattr(module.config, "output_attentions", False)
    if not _sparse_attn_enabled() or output_attentions:
        return None
    # One selection is shared by every head: a per-head mask ([b, h > 1, s, n]) keeps the reference.
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 4 or attention_mask.shape[1] != 1:
        return None
    seq = hidden_states.shape[1]
    if attention_mask.shape[-1] != seq or seq < indexer.token_budget + indexer.compress_ratio:
        return None
    if module.training and module.attention_dropout:
        return None
    # Only prefill into an empty cache is sparse. Probed after the shape checks: a StaticCache answers with a tensor.
    if past_key_values is not None:
        try:
            if past_key_values.get_seq_length(module.layer_idx) != 0:
                return None
        except Exception:
            return None
    import sys
    from unsloth_zoo.sparse_attention import qsa_select_tokens, sparse_gqa_attn, sparse_gqa_fast_path
    # No kernel for QSA on this device (T4 / sm75: no flex, Triton without tensor cores): transformers' own path.
    # Probes have the real [b, h, s, d] shapes (the flex membership-table budget is b * s * s) without allocating.
    batch = hidden_states.shape[0]
    point = hidden_states.new_empty(())
    probe_q = point.expand(batch, module.config.num_attention_heads, seq, module.head_dim)
    probe_kv = point.expand(batch, module.config.num_key_value_heads, seq, module.head_dim)
    if not sparse_gqa_fast_path(probe_q, probe_kv, probe_kv):
        return None
    rope = getattr(sys.modules.get(type(module).__module__), "apply_rotary_pos_emb", None)
    if rope is None:   # a subclass defined elsewhere
        from transformers.models.qwen4_exp import modeling_qwen4_exp
        rope = modeling_qwen4_exp.apply_rotary_pos_emb
    position_embeddings = tuple(position_embeddings)
    # Traced: run attention eagerly with the selection (same break count); the traced path lacks flex and is 16x slower.
    fused = torch.compiler.is_compiling() and past_key_values is None
    if not fused:
        with torch.no_grad():
            selected = qsa_select_tokens(indexer, hidden_states, position_embeddings, attention_mask, rope,
                                         past_key_values = past_key_values)
        if selected is None:
            return None

    cos, sin = (x[:, -seq:, :] for x in position_embeddings)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states, gate = torch.chunk(module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states = rope(query_states, key_states, cos, sin)
    # fp32 RoPE tables promote Q / K (4-bit / fp16 projections): back to the projection dtype, which the kernels
    # take (the compiler's reference rewrite instead upcasts V; the attention math is the same up to rounding).
    if query_states.dtype != value_states.dtype:
        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx)
    if fused:
        attn_output = _select_and_attend(indexer, hidden_states, position_embeddings, attention_mask, rope,
                                         query_states, key_states, value_states, module.scaling)
        if attn_output is None:   # not interval-shaped: the reference (projections recomputed there)
            return None
    else:
        attn_output = sparse_gqa_attn(query_states, key_states, value_states, selected, module.scaling)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = (attn_output * torch.sigmoid(gate)).to(value_states.dtype)
    return module.o_proj(attn_output), None


@torch.compiler.disable
def _select_and_attend(indexer, hidden_states, position_embeddings, attention_mask, rope, query, key, value, scaling):
    """QSA selection + sparse attention outside any trace (eager dispatch); None when the mask is not interval-shaped."""
    from unsloth_zoo.sparse_attention import qsa_select_tokens, sparse_gqa_attn
    with torch.no_grad():
        selected = qsa_select_tokens(indexer, hidden_states, position_embeddings, attention_mask, rope)
    if selected is None:
        return None
    return sparse_gqa_attn(query, key, value, selected, scaling)


def qwen4_exp_attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    # Compiler copies this source into its cache: use only `torch`, `self`, the arguments, the lazy import.
    from unsloth_zoo.temporary_patches.qwen4_exp import (
        _qwen4_exp_sparse_attention, _reference_attention_forward, QWEN4_EXP_SPARSE_STATS,
    )
    out = _qwen4_exp_sparse_attention(self, hidden_states, position_embeddings, attention_mask, past_key_values, kwargs)
    if out is not None:
        if not torch.compiler.is_compiling():
            QWEN4_EXP_SPARSE_STATS["sparse"] += 1
        return out
    if not torch.compiler.is_compiling():
        QWEN4_EXP_SPARSE_STATS["reference"] += 1
    # fp32 RoPE tables with half-precision projections (4-bit) would promote Q / K but not V after RoPE. The
    # compiler's rewrites (RoPE tables cast to the activation dtype, V aligned to Q) never reach this reference,
    # since the class forward it rewrites is this wrapper; cast the tables like its RoPE rewrite does.
    if (
        hidden_states.dtype in (torch.float16, torch.bfloat16)
        and isinstance(position_embeddings, (tuple, list))
        and len(position_embeddings) == 2
        and position_embeddings[0].dtype != hidden_states.dtype
    ):
        position_embeddings = tuple(x.to(hidden_states.dtype) for x in position_embeddings)
    return _reference_attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)


def _attention_forward_mirrored(function):
    """The sparse path re-implements q/k/v/gate/o_proj from transformers 5.22 / 5.23: patch only that code."""
    import inspect
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    markers = ("self.indexer(", "torch.chunk(", "self.head_dim * 2", "self.q_norm(", "self.k_norm(",
               "apply_rotary_pos_emb(query_states, key_states, cos, sin)", "torch.sigmoid(gate)", "self.o_proj(")
    return all(m in source for m in markers)


_reference_ple_forward = None


def qwen4_exp_ple_layer_forward(self, hidden_states, input_ids, past_key_values, conv_mask=None):
    # Compiler copies this source into its cache: use only `torch`, `self`, the arguments, the lazy import.
    from unsloth_zoo.temporary_patches.qwen4_exp import _reference_ple_forward
    # Keep a half residual stream half: the float32 gate sum under autocast would make it and every later MoE input
    # float32; a float32 residual keeps autocast, which casts it for the half short convolution.
    device_type = hidden_states.device.type
    if (
        hidden_states.dtype in (torch.bfloat16, torch.float16)
        and torch.amp.is_autocast_available(device_type)
        and torch.is_autocast_enabled(device_type)
    ):
        with torch.autocast(device_type, enabled=False):
            output = _reference_ple_forward(self, hidden_states, input_ids, past_key_values, conv_mask=conv_mask)
    else:
        output = _reference_ple_forward(self, hidden_states, input_ids, past_key_values, conv_mask=conv_mask)
    return output.to(hidden_states.dtype)


def _is_transformers_indexer_forward(function):
    """By code file: the compiled copy keeps transformers' ``__module__``."""
    code = getattr(function, "__code__", None)
    filename = str(getattr(code, "co_filename", "")).replace("\\", "/")
    return "/transformers/models/qwen4_exp/" in filename


def _reference_returns_float_mask(function):
    """True when the reference turns its boolean selection into an additive mask for a float input mask."""
    import inspect
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return True
    return "is_floating_point()" in source


def _make_fast_indexer_forward(original_forward):
    """Test hook: bind `original_forward` as the long-context reference."""
    global _reference_qsa_forward, _qsa_returns_float_mask
    if original_forward not in (qwen4_exp_qsa_indexer_forward, qwen4_exp_qsa_indexer_forward_guarded):
        _reference_qsa_forward = original_forward
        _qsa_returns_float_mask = _reference_returns_float_mask(original_forward)
    return qwen4_exp_qsa_indexer_forward


# The FP8 checkpoint (Qwen3.8-Flash-Next-FP8) adds the table's per-tensor weight_scale, read by the store.
_NGRAM_UNEXPECTED = r"ple_embedding\.ngram_embedding\.((shard_\d+\.)?weight|weight_scale)"


def _ngram_table_wanted(num_embeddings, embedding_dim, dtype):
    from unsloth_zoo.frozen_table_store import resolve_frozen_table_mode
    nbytes = int(num_embeddings) * int(embedding_dim) * torch.empty((), dtype=dtype).element_size()
    mode = os.environ.get("UNSLOTH_FROZEN_TABLE", "auto").strip().lower()
    if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
        # Full finetuning trains every weight: auto keeps the table a real parameter. An explicit disk / ram
        # request still wins, but the table then stays frozen, so say so.
        if mode not in ("disk", "ram"):
            return False
        logger.warning(f"Unsloth: UNSLOTH_FROZEN_TABLE={mode} keeps the Qwen4Exp n-gram table frozen during full "
                       "finetuning; set UNSLOTH_FROZEN_TABLE=gpu to train it.")
    return resolve_frozen_table_mode(nbytes) != "gpu"


def _ple_dynamic_conv_enabled():
    return os.environ.get("UNSLOTH_QWEN4_EXP_PLE_DYNAMIC_CONV", "1") != "0"


def dilated_depthwise_conv1d(x, weight, bias, dilation):
    """Depthwise Conv1d (no padding, stride 1) as shifted multiply-adds, accumulated in at least float32."""
    dtype = x.dtype
    if torch.is_autocast_enabled(x.device.type):
        dtype = torch.get_autocast_dtype(x.device.type)
    taps = weight.shape[-1]
    length = x.shape[-1] - (taps - 1) * dilation
    acc = torch.promote_types(dtype, torch.float32)
    xf = x.to(dtype).to(acc)
    w = weight.to(dtype).to(acc)
    out = xf[:, :, 0:length] * w[:, 0, 0, None]
    for j in range(1, taps):
        out = out + xf[:, :, j * dilation : j * dilation + length] * w[:, 0, j, None]
    if bias is not None:
        out = out + bias.to(dtype).to(acc)[:, None]
    return out.to(dtype)


class UnslothDilatedDepthwiseConv1d(torch.nn.Conv1d):
    """Inductor's conv backward pins the sequence length (a recompile per length); traced, use multiply-adds."""

    def forward(self, input):
        if (
            torch.compiler.is_compiling()
            and input.dim() == 3
            and self.padding_mode == "zeros"
            and tuple(self.padding) == (0,)
            and tuple(self.stride) == (1,)
            and self.groups == self.in_channels == self.out_channels
            and self.weight.shape[1] == 1
        ):
            return dilated_depthwise_conv1d(input, self.weight, self.bias, self.dilation[0])
        return super().forward(input)


def patch_qwen4_exp_ple_dynamic_conv(modeling=None):
    """Patches __init__, not _short_conv: Unsloth's compiler rebinds methods from its cache but keeps __init__."""
    if modeling is None:
        import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling
    ple_cls = getattr(modeling, "Qwen4ExpTextPLELayer", None)
    if ple_cls is None:
        return False
    if getattr(ple_cls.__init__, "_unsloth_ple_dynamic_conv", False):
        return True
    original_init = ple_cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        conv = getattr(self, "conv1d", None)
        if type(conv) is torch.nn.Conv1d and _ple_dynamic_conv_enabled():
            conv.__class__ = UnslothDilatedDepthwiseConv1d

    __init__._unsloth_ple_dynamic_conv = True
    __init__.__wrapped__ = original_init
    ple_cls.__init__ = __init__
    return True


def patch_qwen4_exp_frozen_table(modeling=None):
    """Build the PLE n-gram table as a parameter-free UnslothFrozenTableEmbedding, so the load never reads
    (or concatenates on CPU) the 128 ``ngram_embedding.shard_*`` tensors; they are bound from the checkpoint
    files right after the load. UNSLOTH_FROZEN_TABLE=gpu (or auto on a small table) keeps the upstream path."""
    if modeling is None:
        import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling
    from unsloth_zoo.frozen_table_store import UnslothFrozenTableEmbedding, install_frozen_table_load_hook

    ngram_cls = getattr(modeling, "Qwen4ExpTextNGramEmbedding", None)
    if ngram_cls is None:
        return False
    # The load hook first: a constructor patch without it would leave tables nothing binds.
    if not install_frozen_table_load_hook():
        raise RuntimeError("Unsloth: could not install the frozen table load hook for Qwen4Exp.")
    if not getattr(ngram_cls.__init__, "_unsloth_frozen_table", False):
        original_init = ngram_cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            emb = getattr(self, "ngram_embedding", None)
            # Only a checkpoint load (meta init); from_config keeps a real, trainable-shaped table.
            if type(emb) is torch.nn.Embedding and emb.weight.is_meta and _ngram_table_wanted(
                emb.num_embeddings, emb.embedding_dim, emb.weight.dtype
            ):
                self.ngram_embedding = UnslothFrozenTableEmbedding(
                    emb.num_embeddings, emb.embedding_dim, dtype=emb.weight.dtype,
                    table_name=f"qwen4_exp ple layer {getattr(self, 'layer_idx', '?')}",
                )

        __init__._unsloth_frozen_table = True
        __init__.__wrapped__ = original_init
        ngram_cls.__init__ = __init__

    # The shard keys now match no parameter: transformers files them as unexpected without reading them.
    from transformers.modeling_utils import PreTrainedModel
    for obj in vars(modeling).values():
        if isinstance(obj, type) and issubclass(obj, PreTrainedModel) and (
            "_keys_to_ignore_on_load_unexpected" in obj.__dict__ or obj.__name__ == "Qwen4ExpPreTrainedModel"
        ):
            current = list(obj.__dict__.get("_keys_to_ignore_on_load_unexpected") or [])
            if _NGRAM_UNEXPECTED not in current:
                obj._keys_to_ignore_on_load_unexpected = current + [_NGRAM_UNEXPECTED]
    return True


def patch_qwen4_exp():
    try:
        import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling
    except Exception:
        return

    table_mode = os.environ.get("UNSLOTH_FROZEN_TABLE", "auto").strip().lower()
    table_error = None
    if table_mode != "gpu":
        try:
            patch_qwen4_exp_frozen_table(modeling)
        except Exception as e:
            table_error = e
            logger.warning(f"Unsloth: Qwen4Exp frozen n-gram table patch failed ({e}); the table will load into RAM.")
    try:
        _patch_qwen4_exp_rest(modeling)
    finally:
        # Without the patch the 102 GB table is concatenated in host RAM: an explicit disk / ram request must not
        # silently become that. Raised after the other Qwen4Exp patches are in.
        if table_error is not None and table_mode in ("disk", "ram"):
            raise RuntimeError(
                f"Unsloth: UNSLOTH_FROZEN_TABLE={table_mode} but the Qwen4Exp n-gram table patch failed: {table_error}"
            ) from table_error


def _patch_qwen4_exp_rest(modeling):
    indexer_cls = getattr(modeling, "Qwen4ExpTextQSAIndexer", None)
    if indexer_cls is not None:
        # The kill switch drops the fast within-budget path but keeps the filled-slot guard.
        wanted = qwen4_exp_qsa_indexer_forward if _fast_qsa_enabled() else qwen4_exp_qsa_indexer_forward_guarded
        current = indexer_cls.forward
        try:
            if current in (qwen4_exp_qsa_indexer_forward, qwen4_exp_qsa_indexer_forward_guarded):
                if _reference_qsa_forward is not None:
                    indexer_cls.forward = wanted
            # Never take the compiler's copy (which calls us) as the reference.
            elif _is_transformers_indexer_forward(current):
                _make_fast_indexer_forward(current)
                indexer_cls.forward = wanted
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: Could not patch Qwen4ExpTextQSAIndexer.forward: {e}")

    global _reference_attention_forward
    attention_cls = getattr(modeling, "Qwen4ExpTextAttention", None)
    if (
        attention_cls is not None
        and _sparse_attn_enabled()
        and attention_cls.forward is not qwen4_exp_attention_forward
        and _is_transformers_indexer_forward(attention_cls.forward)
        and _attention_forward_mirrored(attention_cls.forward)
    ):
        try:
            _reference_attention_forward = attention_cls.forward
            attention_cls.forward = qwen4_exp_attention_forward
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: Could not patch Qwen4ExpTextAttention.forward: {e}")

    global _reference_ple_forward
    ple_cls = getattr(modeling, "Qwen4ExpTextPLELayer", None)
    if (
        ple_cls is not None
        and ple_cls.forward is not qwen4_exp_ple_layer_forward
        and _is_transformers_indexer_forward(ple_cls.forward)
    ):
        try:
            _reference_ple_forward = ple_cls.forward
            ple_cls.forward = qwen4_exp_ple_layer_forward
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: Could not patch Qwen4ExpTextPLELayer.forward: {e}")
    try:
        patch_qwen4_exp_ple_dynamic_conv(modeling)
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch Qwen4ExpTextPLELayer's short conv: {e}")

    experts_cls = getattr(modeling, "Qwen4ExpTextExperts", None)
    block_cls = getattr(modeling, "Qwen4ExpTextSparseMoeBlock", None)
    if experts_cls is None or block_cls is None:
        return
    try:
        from .moe_utils import patch_param_wrapper_for_moe
        from .qwen3_moe import (
            _make_qwen_moe_lora_extractor,
            _make_qwen_moe_experts_forward,
            _make_qwen_moe_sparse_moe_block_forward,
        )
        patch_param_wrapper_for_moe()
        experts_cls._unsloth_lora_extractor_fn = staticmethod(_make_qwen_moe_lora_extractor())
        patch_function(
            experts_cls, "forward",
            _make_qwen_moe_experts_forward(module_name="unsloth_zoo.temporary_patches.qwen4_exp"),
        )
        patch_function(
            block_cls, "forward",
            _make_qwen_moe_sparse_moe_block_forward(
                use_shared_expert=True, module_name="unsloth_zoo.temporary_patches.qwen4_exp",
            ),
        )
    except Exception as e:
        if UNSLOTH_ENABLE_LOGGING:
            logger.warning(f"Unsloth: Could not patch Qwen4Exp MoE: {e}")


TEMPORARY_PATCHES.append(patch_qwen4_exp)
