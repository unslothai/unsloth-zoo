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

"""Qwen4Exp (Qwen3.8-Flash-Next, transformers `qwen4_exp`).

1. MoE: Qwen4ExpTextExperts / Qwen4ExpTextSparseMoeBlock are the Qwen3.5-MoE layout
   (3D gate_up_proj [E, 2I, H] / down_proj [E, H, I], softmax top-k router, sigmoid-gated
   shared expert), so they take the same grouped-GEMM backend and LoRA extractor.
2. QSA indexer: the reference Qwen4ExpTextQSAIndexer.forward loops over batch x query in
   Python with a torch.nonzero / topk per query (B*T host syncs per full-attention layer).
   Its output is a token-selection mask: the top `indexer_budget // compress_ratio` blocks of
   `compress_ratio` visible tokens plus the incomplete tail. A query sees at most kv_length
   tokens, so when kv_length < budget + compress_ratio every complete block is selected and
   the mask is exactly the visible mask. That case is answered from the input mask with no
   host sync; longer contexts keep the reference. The selection is a topk over indices, so
   no gradient reaches index_qk_proj through the LM loss either way.
   Kill switch: UNSLOTH_QWEN4_EXP_FAST_QSA=0.
"""

__all__ = ["patch_qwen4_exp"]

import os

import torch

from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, logger

def _fast_qsa_enabled():
    return os.environ.get("UNSLOTH_QWEN4_EXP_FAST_QSA", "1") != "0"


# The transformers forward, kept here because Unsloth's compiler replaces the modeling
# class with a generated copy whose forward is the function below.
_reference_qsa_forward = None


def qwen4_exp_qsa_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None):
    # Self-contained on purpose: Unsloth's compiler copies this source into its cache
    # module, so it may only use `torch`, `self` and the lazy import below.
    if (
        attention_mask is not None
        and attention_mask.dim() == 4
        and attention_mask.shape[-1] < self.token_budget + self.compress_ratio
    ):
        # Every complete block is selected (block_topk >= complete blocks), plus the tail:
        # the selection is exactly the visible mask.
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


def _is_transformers_indexer_forward(function):
    """The forward transformers ships, read off its code object: Unsloth's compiled copy
    keeps the transformers ``__module__`` for identity checks but lives in its cache file."""
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

    # ---- QSA indexer
    indexer_cls = getattr(modeling, "Qwen4ExpTextQSAIndexer", None)
    if (
        indexer_cls is not None
        and _fast_qsa_enabled()
        and indexer_cls.forward is not qwen4_exp_qsa_indexer_forward
        # Only ever wrap transformers' own forward: after Unsloth's compiler swaps in its
        # generated class, a re-run must not take that copy (which calls us) as the reference.
        and _is_transformers_indexer_forward(indexer_cls.forward)
    ):
        try:
            _make_fast_indexer_forward(indexer_cls.forward)
            indexer_cls.forward = qwen4_exp_qsa_indexer_forward
        except Exception as e:
            if UNSLOTH_ENABLE_LOGGING:
                logger.warning(f"Unsloth: Could not patch Qwen4ExpTextQSAIndexer.forward: {e}")

    # ---- MoE (same layout as Qwen3.5-MoE)
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
