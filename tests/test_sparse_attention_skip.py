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

"""Exact full-selection skip for sparse-attention indexers (temporary_patches/sparse_attention.py).

Tiny random models with shrunk selection budgets so both regimes are reachable: below the budget
the patched model must match upstream bit for bit (logits and every gradient) while engaging the
skip; above it the reference indexer runs. A short prefill followed by decoding past the budget
must reproduce upstream's tokens, logits and indexer cache.
"""

import importlib.util

import pytest
import torch

from unsloth_zoo.flex_attention.sparse import SPARSE_ATTN_STATS, reset_sparse_attn_stats
import unsloth_zoo.temporary_patches.sparse_attention as sparse_attention

# The consolidated CUDA spoof makes is_available() True on CPU-only runners.
_REAL_CUDA = torch.cuda.is_available() and not getattr(torch.cuda, "_unsloth_consolidated_spoof", False)
DEVICES = ["cpu"] + (["cuda"] if _REAL_CUDA else [])


def _has(model_type):
    return (
        importlib.util.find_spec(f"transformers.models.{model_type}") is not None
        and sparse_attention._transformers_tested()
    )


def _deepseek_v32():
    from transformers import DeepseekV32Config, DeepseekV32ForCausalLM
    config = DeepseekV32Config(
        vocab_size=97, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        index_n_heads=4, index_head_dim=32, index_topk=8, n_routed_experts=4, num_experts_per_tok=2,
        first_k_dense_replace=1, max_position_embeddings=256, n_group=1, topk_group=1,
    )
    return config, DeepseekV32ForCausalLM


def _glm_moe_dsa():
    from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
    config = GlmMoeDsaConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        index_n_heads=4, index_head_dim=32, index_topk=8, n_routed_experts=4, num_experts_per_tok=2,
        first_k_dense_replace=1, max_position_embeddings=256, indexer_types=["full", "shared", "full"],
        n_group=1, topk_group=1,
    )
    return config, GlmMoeDsaForCausalLM


def _minimax_m3():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM
    config = MiniMaxM3VLTextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, dense_intermediate_size=96,
        shared_intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, rotary_dim=16, index_n_heads=2, index_head_dim=16, index_block_size=4,
        index_topk_blocks=3, index_local_blocks=1, num_local_experts=4, num_experts_per_tok=2,
        layer_types=["minimax_m3_sparse", "minimax_m3_sparse"], mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=256,
    )
    return config, MiniMaxM3VLForCausalLM


def _deepseek_v4():
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
    config = DeepseekV4Config(
        vocab_size=97, hidden_size=64, moe_intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=1, head_dim=32, qk_rope_head_dim=8,
        partial_rotary_factor=0.25, q_lora_rank=32, o_groups=2, o_lora_rank=16,
        index_n_heads=2, index_head_dim=16, index_topk=3, sliding_window=8,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        mlp_layer_types=["moe", "moe", "moe"], n_routed_experts=4, num_experts_per_tok=2,
        max_position_embeddings=256, hc_mult=2, num_nextn_predict_layers=0,
    )
    return config, DeepseekV4ForCausalLM


# (builder, model_type, sequence length within the budget, sequence length over it)
FAMILIES = {
    "deepseek_v32": (_deepseek_v32, 8, 13),
    "glm_moe_dsa": (_glm_moe_dsa, 8, 13),
    "minimax_m3_vl": (_minimax_m3, 12, 19),
    "deepseek_v4": (_deepseek_v4, 15, 23),
}


def _patch_all():
    sparse_attention.patch_deepseek_v32_sparse_skip()
    sparse_attention.patch_glm_moe_dsa_sparse_skip()
    sparse_attention.patch_minimax_m3_sparse_skip()
    sparse_attention.patch_deepseek_v4_sparse_skip()


def _build(family, device, attn_implementation):
    if family == "deepseek_v4" and attn_implementation == "sdpa":
        attn_implementation = "eager"  # upstream DeepSeek-V4 is eager-only
    builder, short_len, long_len = FAMILIES[family]
    config, model_cls = builder()
    config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    model = model_cls(config).to(device=device, dtype=torch.float32)
    # Random init leaves some parameters (e.g. V4 sinks / position_bias) empty: make them finite.
    with torch.no_grad():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                p.normal_(0, 0.02)
    model.train()
    return model, short_len, long_len


def _run(model, input_ids, attention_mask=None):
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logits.float().square().mean().backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}
    return logits.detach().clone(), grads


def _assert_same(a, b):
    logits_a, grads_a = a
    logits_b, grads_b = b
    assert torch.equal(logits_a, logits_b)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        if grads_a[name] is None or grads_b[name] is None:
            assert grads_a[name] is None and grads_b[name] is None, name
        else:
            assert torch.equal(grads_a[name], grads_b[name]), name


def _skips(family):
    return SPARSE_ATTN_STATS.get("skip_full", {}).get(family, 0)


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    _patch_all()
    reset_sparse_attn_stats()
    yield
    reset_sparse_attn_stats()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_skip_is_exact_and_engaged(family, attn_implementation, device, monkeypatch):
    if not _has(family):
        pytest.skip(f"transformers without {family}, or older than 5.17")
    if family == "deepseek_v4" and attn_implementation == "sdpa":
        pytest.skip("upstream DeepSeek-V4 is eager-only")
    model, short_len, long_len = _build(family, device, attn_implementation)
    generator = torch.Generator().manual_seed(1)
    for seq_len, expect_skip in ((short_len, True), (long_len, False)):
        input_ids = torch.randint(3, 97, (2, seq_len), generator=generator).to(device)
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "0")
        reset_sparse_attn_stats()
        reference = _run(model, input_ids)
        assert _skips(family) == 0
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "1")
        patched = _run(model, input_ids)
        _assert_same(reference, patched)
        assert (_skips(family) > 0) == expect_skip, (seq_len, SPARSE_ATTN_STATS)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layout", ["padded", "padded_positions", "packed_positions"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_padded_and_packed_batches_are_exact(family, layout, device, monkeypatch):
    if not _has(family):
        pytest.skip(f"transformers without {family}, or older than 5.17")
    model, short_len, _ = _build(family, device, "sdpa")
    input_ids = torch.randint(3, 97, (3, short_len), generator=torch.Generator().manual_seed(2)).to(device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = None
    if layout == "packed_positions":
        # Two documents per row, positions restarting at the boundary (no padding mask).
        attention_mask = None
        split = short_len // 2
        position_ids = torch.cat(
            [torch.arange(split), torch.arange(short_len - split)]
        ).to(device).expand(3, -1).contiguous()
    else:
        attention_mask[0, :3] = 0  # left padding
        attention_mask[1, -2:] = 0  # right padding
        if layout == "padded_positions":
            # What generate() passes: content positions, 0 at each row's first real token.
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

    def run():
        model.zero_grad(set_to_none=True)
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        ).logits
        logits.float().square().mean().backward()
        grads = {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in model.named_parameters()}
        return logits.detach().clone(), grads

    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "0")
    reference = run()
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "1")
    reset_sparse_attn_stats()
    patched = run()
    _assert_same(reference, patched)
    assert _skips(family) > 0


def _decode(model, input_ids, steps):
    """Greedy decode with a cache; returns tokens, per-step logits and the cache."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        cache, logits = out.past_key_values, [out.logits[:, -1].clone()]
        tokens = input_ids
        for _ in range(steps):
            next_token = logits[-1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=-1)
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits.append(out.logits[:, -1].clone())
    return tokens, torch.stack(logits), cache


def _cache_tensors(cache):
    found = {}
    for i, layer in enumerate(getattr(cache, "layers", [])):
        for name, value in vars(layer).items():
            if torch.is_tensor(value):
                found[f"{i}.{name}"] = value
            elif isinstance(value, dict):
                for key, inner in value.items():
                    if torch.is_tensor(inner):
                        found[f"{i}.{name}.{key}"] = inner
    return found


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", list(FAMILIES))
def test_decode_across_the_budget(family, device, monkeypatch):
    """Prefill under the budget (skip engaged), then decode past it: the indexer cache must be complete."""
    if not _has(family):
        pytest.skip(f"transformers without {family}, or older than 5.17")
    model, short_len, long_len = _build(family, device, "sdpa")
    model.eval()
    prompt = torch.randint(3, 97, (2, short_len - 1), generator=torch.Generator().manual_seed(3)).to(device)
    steps = long_len - short_len + 3
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "0")
    tokens_ref, logits_ref, cache_ref = _decode(model, prompt, steps)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_SKIP_FULL", "1")
    reset_sparse_attn_stats()
    tokens, logits, cache = _decode(model, prompt, steps)
    assert _skips(family) > 0
    assert torch.equal(tokens_ref, tokens)
    assert torch.equal(logits_ref, logits)
    ref_state, state = _cache_tensors(cache_ref), _cache_tensors(cache)
    assert ref_state.keys() == state.keys() and ref_state
    for name in ref_state:
        assert torch.equal(ref_state[name], state[name]), name


def test_patch_is_idempotent_and_keeps_the_reference():
    if not _has("deepseek_v32"):
        pytest.skip("transformers without deepseek_v32")
    from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer
    reference = sparse_attention._REFERENCE["deepseek_v32.indexer"]
    _patch_all()
    _patch_all()
    # The sparse flex patch (sparse_attention_flex.py) wraps this forward and delegates to it.
    forward = DeepseekV32Indexer.forward
    if getattr(forward, "_unsloth_sparse_flex", False):
        forward = DeepseekV32Indexer._unsloth_inner_forward
    assert forward is sparse_attention.deepseek_v32_indexer_forward
    assert sparse_attention._REFERENCE["deepseek_v32.indexer"] is reference
    assert sparse_attention._is_transformers_function(reference, "deepseek_v32")


def test_missing_model_is_a_no_op(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    before = dict(sparse_attention._REFERENCE)
    _patch_all()
    assert sparse_attention._REFERENCE == before


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v4"])
def test_no_grad_indexers_match_unpatched_upstream(family, device, monkeypatch):
    """Upstream runs these two indexers with autograd on; their integer output carries no gradient,
    so running them under no_grad must leave logits and every gradient unchanged."""
    if not _has(family):
        pytest.skip(f"transformers without {family}, or older than 5.17")
    modeling = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    model, _, long_len = _build(family, device, "eager")
    input_ids = torch.randint(3, 97, (2, long_len), generator=torch.Generator().manual_seed(4)).to(device)
    patched = _run(model, input_ids)
    if family == "minimax_m3_vl":
        cls = modeling.MiniMaxM3VLIndexer
        monkeypatch.setattr(cls, "forward", sparse_attention._REFERENCE["minimax_m3_vl.indexer"])
    else:
        monkeypatch.setattr(modeling.DeepseekV4Indexer, "forward", sparse_attention._REFERENCE["deepseek_v4.indexer"])
        monkeypatch.setattr(
            modeling.DeepseekV4IndexerScorer, "forward", sparse_attention._REFERENCE["deepseek_v4.scorer"]
        )
    upstream = _run(model, input_ids)
    _assert_same(upstream, patched)


def test_older_transformers_keeps_upstream(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "5.16.1")
    assert not sparse_attention._transformers_tested()
    assert sparse_attention._modeling_module("deepseek_v32") is None
    monkeypatch.setattr(transformers, "__version__", "5.21.0.dev0")
    assert sparse_attention._transformers_tested()
