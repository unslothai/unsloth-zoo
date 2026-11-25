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

"""Sparse skip / flex / GLM-5.3-Flash patches install on 5.24 and use the `cache=` they are handed."""

import importlib.util
import os
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_sparse_attention_skip as S  # noqa: E402
from unsloth_zoo.temporary_patches import sparse_attention as skip  # noqa: E402
from unsloth_zoo.temporary_patches import sparse_attention_flex as flex  # noqa: E402
from unsloth_zoo.temporary_patches import glm5_next_sparse_flex as glm5  # noqa: E402


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("UNSLOTH_SPARSE_"):
            monkeypatch.delenv(name, raising = False)
    # No dynamo.config.patch(disable=True): it leaks into later tests' compiled mask builders.
    yield


def _glm5_next():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel
    config = Glm5NextTextConfig(
        hidden_size = 64, intermediate_size = 64, moe_intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 4, qk_nope_head_dim = 16, qk_rope_head_dim = 0,
        v_head_dim = 16, kv_lora_rank = 32, q_lora_rank = 32, vocab_size = 97, index_topk = 8,
        index_n_heads = 4, index_head_dim = 16, index_kpool = 4, n_routed_experts = 4,
        num_experts_per_tok = 2, linear_num_heads = 2, linear_head_dim = 16, pad_token_id = 0,
        layer_types = ["indexed_attention", "indexed_attention"], indexer_types = ["full", "shared"],
        mlp_layer_types = ["dense", "dense"],
    )
    return config, Glm5NextTextModel


BUILDERS = {
    "deepseek_v32": S._deepseek_v32,
    "glm_moe_dsa": S._glm_moe_dsa,
    "minimax_m3_vl": S._minimax_m3,
    "glm5_next": _glm5_next,
}
ATTENTION = {
    "deepseek_v32": ("deepseek_v32", "DeepseekV32Attention", "DeepseekV32Indexer"),
    "glm_moe_dsa": ("glm_moe_dsa", "GlmMoeDsaAttention", "GlmMoeDsaIndexer"),
    "minimax_m3_vl": ("minimax_m3_vl", "MiniMaxM3VLAttention", "MiniMaxM3VLIndexer"),
    "glm5_next": ("glm5_next", "Glm5NextTextAttention", "Glm5NextTextIndexer"),
}


def _available(family):
    name = ATTENTION[family][0]
    return importlib.util.find_spec(f"transformers.models.{name}") is not None and skip._transformers_tested()


def _patch_all():
    S._patch_all()
    flex.patch_minimax_m3_sparse_flex()
    flex.patch_dsa_sparse_flex()
    glm5.patch_glm5_next_sparse()


def _outputs(out):
    return out.logits if getattr(out, "logits", None) is not None else out.last_hidden_state


@pytest.mark.parametrize("family", list(BUILDERS))
def test_sparse_patches_install_and_decode_with_the_cache(family):
    if not _available(family):
        pytest.skip(f"transformers without {family}, or older than 5.17")
    _patch_all()
    module_name, attn_name, indexer_name = ATTENTION[family]
    modeling = importlib.import_module(f"transformers.models.{module_name}.modeling_{module_name}")
    for name in (attn_name, indexer_name):
        assert getattr(getattr(modeling, name).forward, "_unsloth_sparse_flex", False), f"{name} left unpatched"

    config, model_cls = BUILDERS[family]()
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = model_cls(config).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                p.normal_(0, 0.02)
    ids = torch.randint(3, 97, (2, 14), generator = torch.Generator().manual_seed(1))
    patched = _cached_decode(model, ids, prefill = 6)
    if family == "glm_moe_dsa":
        # Shared-indexer layers: upstream cached decode differs from uncached, so compare cached vs cached.
        attn, indexer = getattr(modeling, attn_name), getattr(modeling, indexer_name)
        saved = attn.forward, indexer.forward
        attn.forward, indexer.forward = attn._unsloth_upstream_forward, indexer._unsloth_original_forward
        try:
            expected = _cached_decode(model, ids, prefill = 6)
        finally:
            attn.forward, indexer.forward = saved
    else:
        with torch.no_grad():
            expected = _outputs(model(input_ids = ids, use_cache = False))[:, 5:]
    torch.testing.assert_close(patched, expected, rtol = 1e-4, atol = 1e-4)


def _cached_decode(model, ids, prefill):
    with torch.no_grad():
        out = model(input_ids = ids[:, :prefill], use_cache = True)
        cache = out.past_key_values
        assert cache is not None
        steps = [_outputs(out)[:, -1]]
        for t in range(prefill, ids.shape[1]):
            out = model(input_ids = ids[:, t : t + 1], past_key_values = cache, use_cache = True)
            cache = out.past_key_values
            steps.append(_outputs(out)[:, -1])
    return torch.stack(steps, dim = 1)
