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
nonzero/topk loop when kv_length < budget + compress_ratio; UNSLOTH_QWEN4_EXP_FAST_QSA=0 disables. The ~102 GB PLE
n-gram table is streamed from the checkpoint by ``unsloth_zoo.frozen_table_store`` unless UNSLOTH_FROZEN_TABLE=gpu."""

__all__ = ["patch_qwen4_exp", "patch_qwen4_exp_frozen_table"]

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


def _make_fast_indexer_forward(original_forward):
    """Test hook: bind `original_forward` as the long-context reference."""
    global _reference_qsa_forward
    if original_forward is not qwen4_exp_qsa_indexer_forward:
        _reference_qsa_forward = original_forward
    return qwen4_exp_qsa_indexer_forward


_NGRAM_UNEXPECTED = r"ple_embedding\.ngram_embedding\.(shard_\d+\.)?weight"


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
