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

"""Qwen4Exp (Qwen3.8-Flash-Next) patches.

MoE takes the Qwen3.5-MoE grouped-GEMM backend. QSA indexer: when kv_length < budget +
compress_ratio every complete block is selected, so the mask is the visible mask and the
per-query nonzero/topk loop (B*T host syncs) is skipped. Kill switch: UNSLOTH_QWEN4_EXP_FAST_QSA=0.
PLE: under autocast the gate's `sum` runs in float32, and the PLE output added to the residual
made the residual stream (and every later MoE input) float32; it is returned in the residual's dtype.
"""

__all__ = ["patch_qwen4_exp"]

import os

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, logger

def _fast_qsa_enabled():
    return os.environ.get("UNSLOTH_QWEN4_EXP_FAST_QSA", "1") != "0"


_reference_qsa_forward = None


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
        return torch.where(
            attention_mask == 0, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min
        )
    from unsloth_zoo.temporary_patches.qwen4_exp import _reference_qsa_forward
    return _reference_qsa_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)


_reference_ple_forward = None


def qwen4_exp_ple_layer_forward(self, hidden_states, input_ids, past_key_values, conv_mask=None):
    # Compiler copies this source into its cache: use only `torch`, `self`, the arguments, the lazy import.
    from unsloth_zoo.temporary_patches.qwen4_exp import _reference_ple_forward
    # Autocast runs the gate `sum` in float32, and the float32 PLE output would turn the residual stream
    # (and every later MoE input) float32. Run PLE in the model's dtype, as inference does.
    device_type = hidden_states.device.type
    if torch.amp.is_autocast_available(device_type) and torch.is_autocast_enabled(device_type):
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


def _make_fast_indexer_forward(original_forward):
    """Test hook: bind `original_forward` as the long-context reference."""
    global _reference_qsa_forward
    if original_forward is not qwen4_exp_qsa_indexer_forward:
        _reference_qsa_forward = original_forward
    return qwen4_exp_qsa_indexer_forward


def patch_qwen4_exp():
    try:
        import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling
    except Exception:
        return

    indexer_cls = getattr(modeling, "Qwen4ExpTextQSAIndexer", None)
    if (
        indexer_cls is not None
        and _fast_qsa_enabled()
        and indexer_cls.forward is not qwen4_exp_qsa_indexer_forward
        # Never take the compiler's copy (which calls us) as the reference.
        and _is_transformers_indexer_forward(indexer_cls.forward)
    ):
        try:
            _make_fast_indexer_forward(indexer_cls.forward)
            indexer_cls.forward = qwen4_exp_qsa_indexer_forward
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: Could not patch Qwen4ExpTextQSAIndexer.forward: {e}")

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
