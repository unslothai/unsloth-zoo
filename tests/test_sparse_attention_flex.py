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

"""Long-context sparse flex attention (MiniMax M3, DeepSeek V3.2, GLM-MoE-DSA).

CPU: chunked indexers select exactly what upstream selects, block ids are ascending, the
patches install on transformers 5.17 and stay off where they must. GPU (sm80+): flex output
and Q / K / V gradients match upstream's dense masked attention, and whole tiny models match
with flex on vs off."""

import importlib
import os

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")
minimax = importlib.util.find_spec("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
dsa = importlib.util.find_spec("transformers.models.deepseek_v32.modeling_deepseek_v32")
glm = importlib.util.find_spec("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")
needs_models = pytest.mark.skipif(
    minimax is None or dsa is None or glm is None, reason = "transformers without MiniMax M3 / DSA models",
)
gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability() < (8, 0),
    reason = "flex attention parity needs CUDA sm80+",
)

from unsloth_zoo.flex_attention import sparse_blocks as sb  # noqa: E402


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in ("UNSLOTH_SPARSE_ATTN_FLEX", "UNSLOTH_SPARSE_INDEXER_CHUNKED", "UNSLOTH_SPARSE_INDEXER_Q_CHUNK",
                 "UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS"):
        monkeypatch.delenv(name, raising = False)
    sb._FAILED.clear()


def _patched():
    from unsloth_zoo.temporary_patches import sparse_attention_flex as s
    s.patch_minimax_m3_sparse_flex()
    s.patch_dsa_sparse_flex()
    return s


# ---------------------------------------------------------------- configs
def _minimax_config(**kw):
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig
    base = dict(
        hidden_size = 64, intermediate_size = 64, num_hidden_layers = 2, num_attention_heads = 8,
        num_key_value_heads = 2, head_dim = 32, vocab_size = 256, index_n_heads = 2, index_head_dim = 16,
        index_block_size = 128, index_topk_blocks = 2, index_local_blocks = 1,
        layer_types = ["minimax_m3_sparse", "minimax_m3_sparse"],
        mlp_layer_types = ["dense", "dense"], dense_intermediate_size = 64,
    )
    base.update(kw)
    config = MiniMaxM3VLTextConfig(**base)
    config._attn_implementation = kw.get("attn", "sdpa")
    return config


def _dsa_config(family, **kw):
    if family == "deepseek_v32":
        from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config as C
    else:
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig as C
    base = dict(
        hidden_size = 64, intermediate_size = 64, moe_intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 4, qk_nope_head_dim = 16, qk_rope_head_dim = 16,
        v_head_dim = 16, kv_lora_rank = 32, q_lora_rank = 32, vocab_size = 256, index_topk = 64,
        index_n_heads = 4, index_head_dim = 32, n_routed_experts = 4, num_experts_per_tok = 2,
        first_k_dense_replace = 2, mlp_layer_types = ["dense", "dense"],
    )
    if family == "glm_moe_dsa":
        base["indexer_types"] = ["full", "shared"]
    base.update(kw)
    config = C(**base)
    config._attn_implementation = "sdpa"
    return config


def _indexer_module(family, config):
    m = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    cls = {"deepseek_v32": "DeepseekV32Indexer", "glm_moe_dsa": "GlmMoeDsaIndexer", "minimax_m3_vl": "MiniMaxM3VLIndexer"}[family]
    return m, getattr(m, cls)(config, 0)


def _rope(config, family, seq, device = "cpu"):
    m = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    rope_cls = {"deepseek_v32": "DeepseekV32RotaryEmbedding", "glm_moe_dsa": "GlmMoeDsaRotaryEmbedding",
                "minimax_m3_vl": "MiniMaxM3VLRotaryEmbedding"}[family]
    rope = getattr(m, rope_cls)(config).to(device)
    pos = torch.arange(seq, device = device)[None]
    x = torch.zeros(1, seq, config.hidden_size, device = device)
    return rope(x, pos), pos


# ---------------------------------------------------------------- CPU
def test_pack_ascending_orders_active_ids():
    sel = torch.tensor([[[[True, False, True, True, False, True]]]])
    num, ids = sb._pack_ascending(sel)
    assert num.item() == 4
    assert ids[0, 0, 0, :4].tolist() == [0, 2, 3, 5]


@needs_models
def test_patches_install_and_are_idempotent():
    s = _patched()
    s.patch_minimax_m3_sparse_flex()
    s.patch_dsa_sparse_flex()
    from transformers.models.deepseek_v32 import modeling_deepseek_v32 as a
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as b
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as c
    for cls in (a.DeepseekV32Attention, a.DeepseekV32Indexer, b.GlmMoeDsaAttention, b.GlmMoeDsaIndexer,
                c.MiniMaxM3VLAttention, c.MiniMaxM3VLIndexer):
        assert getattr(cls.forward, "_unsloth_sparse_flex", False), cls.__name__
    assert callable(a.unsloth_deepseek_v32_sparse_attention_forward)
    assert callable(c.unsloth_minimax_m3_sparse_attention_forward)


@needs_models
@pytest.mark.parametrize("seq", [200, 333])
def test_minimax_chunked_indexer_selects_like_upstream(monkeypatch, seq):
    _patched()
    torch.manual_seed(0)
    config = _minimax_config()
    m, indexer = _indexer_module("minimax_m3_vl", config)
    (cos, sin), pos = _rope(config, "minimax_m3_vl", seq)
    h = torch.randn(2, seq, config.hidden_size)
    pos = pos.expand(2, -1)
    ref = indexer._unsloth_original_forward(h, (cos, sin), None, pos)
    monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK", "64")
    before = sb.SPARSE_FLEX_STATS["minimax_m3_indexer_chunked"]
    got = indexer(h, (cos, sin), None, pos)
    assert sb.SPARSE_FLEX_STATS["minimax_m3_indexer_chunked"] == before + 1
    assert got.shape == ref.shape
    assert torch.equal(got.sort(dim = -1).values, ref.sort(dim = -1).values)


@needs_models
@pytest.mark.parametrize("family", ["deepseek_v32", "glm_moe_dsa"])
def test_dsa_chunked_indexer_selects_like_upstream(monkeypatch, family):
    _patched()
    torch.manual_seed(0)
    seq = 300
    config = _dsa_config(family)
    m, indexer = _indexer_module(family, config)
    (cos, sin), pos = _rope(config, family, seq)
    h = torch.randn(2, seq, config.hidden_size)
    q_resid = torch.randn(2, seq, config.q_lora_rank)
    mask = torch.ones(seq, seq, dtype = torch.bool).tril()[None].expand(2, -1, -1).clone()
    mask[1, :, :40] = False  # left padding on row 1
    ref = indexer._unsloth_original_forward(h, q_resid, (cos, sin), mask, pos)
    monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK", "64")
    got = indexer(h, q_resid, (cos, sin), mask, pos)
    # Rows whose valid keys are fewer than topk select some -inf keys; those ids are arbitrary
    # but masked out downstream. Compare the selected valid keys.
    keep_ref = torch.zeros(2, seq, seq, dtype = torch.bool).scatter(-1, ref.long(), True) & mask
    keep_got = torch.zeros(2, seq, seq, dtype = torch.bool).scatter(-1, got.long(), True) & mask
    assert torch.equal(keep_ref, keep_got)


@needs_models
def test_flex_not_engaged_on_cpu_or_when_disabled(monkeypatch):
    q = torch.randn(1, 4, 32, 16)
    assert not sb.sparse_flex_device_ok(q)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0")
    assert not sb.sparse_flex_enabled()


@needs_models
def test_unsupported_source_layout_is_not_patched(monkeypatch):
    from unsloth_zoo.temporary_patches import sparse_attention_flex as s
    assert not s._source_has(lambda x: x, s._DSA_ATTN_NEEDLES)


# ---------------------------------------------------------------- GPU kernels
def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def _grads(fn, *xs):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    out = fn(*xs)
    # Fixed fp32 cotangent: randn_like would follow the output's dtype and strides.
    g = torch.randn(out.shape, generator = torch.Generator(out.device).manual_seed(1), device = out.device)
    (out.float() * g).sum().backward()
    return out.detach(), [x.grad for x in xs]


def _dense(q, k, v, mask, scale):
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim = 1)
    v = v.repeat_interleave(groups, dim = 1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask = mask, scale = scale).transpose(1, 2)


def _valid_rows(mask4, b_rows):
    return mask4.any(dim = -1)  # [B, H or 1, S]


@needs_models
@gpu
@pytest.mark.parametrize("mask_kind", ["none", "bool_right_pad", "float"])
def test_minimax_flex_matches_dense_block_mask(mask_kind):
    _patched()
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    B, Hq, Hkv, D, S = 2, 16, 4, 128, 1024
    config = _minimax_config(num_attention_heads = Hq, num_key_value_heads = Hkv, head_dim = D, index_n_heads = Hkv,
                             index_topk_blocks = 3, hidden_size = 256)
    m, indexer = _indexer_module("minimax_m3_vl", config)
    indexer = indexer.to(dev)
    (cos, sin), pos = _rope(config, "minimax_m3_vl", S, dev)
    pos = pos.expand(B, -1)
    h = torch.randn(B, S, config.hidden_size, device = dev)
    blocks = indexer(h, (cos, sin), None, pos)
    q = torch.randn(B, Hq, S, D, device = dev, dtype = dt)
    k = torch.randn(B, Hkv, S, D, device = dev, dtype = dt)
    v = torch.randn(B, Hkv, S, D, device = dev, dtype = dt)
    if mask_kind == "none":
        base = None
    else:
        keep = torch.ones(S, S, dtype = torch.bool, device = dev).tril()[None, None].expand(B, 1, S, S).clone()
        keep[1, :, :, S - 200:] = False
        keep[1, :, S - 200:, :] = False
        base = keep if mask_kind == "bool_right_pad" else torch.zeros(B, 1, S, S, device = dev, dtype = dt).masked_fill(~keep, torch.finfo(dt).min)
    dense_mask = indexer.build_block_mask(blocks, base, S, dt, dev, pos)
    scale = D ** -0.5
    ref, ref_g = _grads(lambda a, b, c: _dense(a, b, c, dense_mask, scale), q, k, v)
    got, got_g = _grads(lambda a, b, c: sb.minimax_m3_flex_attention(a, b, c, blocks, base, pos, scale, 128), q, k, v)
    rows = (dense_mask > torch.finfo(dt).min).any(dim = -1).transpose(1, 2)  # [B, S, H]
    err = (got.float() - ref.float()).abs()[rows].max().item()
    assert err < 2e-2, err
    for name, a, b in zip("qkv", got_g, ref_g):
        # Rows with no allowed key are garbage in both (uniform vs zero): exclude their upstream grads.
        r = _rel(a, b) if mask_kind == "none" else _rel(a[:1], b[:1])
        assert r < 2e-2, (name, r)


@needs_models
@gpu
@pytest.mark.parametrize("dims", [(192, 128), (256, 256), (64, 64)])
@pytest.mark.parametrize("bool_mask", [True, False])
def test_dsa_flex_matches_dense_masked_sdpa(dims, bool_mask):
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    dq, dv = dims
    B, H, S, topk = 2, 4, 1024, 256
    q = torch.randn(B, H, S, dq, device = dev, dtype = dt)
    k = torch.randn(B, H, S, dq, device = dev, dtype = dt)
    v = torch.randn(B, H, S, dv, device = dev, dtype = dt)
    keep = torch.ones(S, S, dtype = torch.bool, device = dev).tril()[None, None].expand(B, 1, S, S).clone()
    keep[1, :, :, :100] = False  # left padding
    keep[1, :, :100, :100] = torch.eye(100, dtype = torch.bool, device = dev)
    scores = torch.randn(B, S, S, device = dev).masked_fill(~keep[:, 0], float("-inf"))
    topk_idx = scores.topk(topk, dim = -1).indices.to(torch.int32)
    index_mask = torch.ones(B, S, S, dtype = torch.bool, device = dev).scatter(-1, topk_idx.long(), False).unsqueeze(1)
    if bool_mask:
        base, dense = keep, keep & ~index_mask
    else:
        base = torch.zeros(B, 1, S, S, device = dev, dtype = dt).masked_fill(~keep, torch.finfo(dt).min)
        dense = base.masked_fill(index_mask, torch.finfo(dt).min)
    scale = dq ** -0.5
    ref, ref_g = _grads(lambda a, b, c: _dense(a, b, c, dense, scale), q, k, v)
    got, got_g = _grads(lambda a, b, c: sb.dsa_flex_attention(a, b, c, topk_idx, base, scale), q, k, v)
    assert _rel(got, ref) < 2e-2
    for name, a, b in zip("qkv", got_g, ref_g):
        assert _rel(a, b) < 2e-2, (name, _rel(a, b))


@needs_models
@gpu
def test_dsa_flex_reuses_block_mask_for_shared_layers():
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    B, H, S, D = 1, 4, 512, 64
    q, k, v = (torch.randn(B, H, S, D, device = dev, dtype = dt) for _ in range(3))
    keep = torch.ones(S, S, dtype = torch.bool, device = dev).tril()[None, None]
    topk_idx = torch.randn(B, S, S, device = dev).masked_fill(~keep[:, 0], float("-inf")).topk(128, dim = -1).indices.int()
    before = sb.SPARSE_FLEX_STATS["dsa_flex_reused"]
    a = sb.dsa_flex_attention(q, k, v, topk_idx, keep, D ** -0.5)
    b = sb.dsa_flex_attention(q, k, v, topk_idx, keep, D ** -0.5)
    assert sb.SPARSE_FLEX_STATS["dsa_flex_reused"] == before + 1
    assert torch.equal(a, b)


# ---------------------------------------------------------------- GPU whole tiny models
def _model(family, config):
    if family == "minimax_m3_vl":
        from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM as M
    elif family == "deepseek_v32":
        from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM as M
    else:
        from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM as M
    torch.manual_seed(0)
    return M(config).to("cuda", torch.bfloat16)


def _step(model, ids, labels):
    model.zero_grad(set_to_none = True)
    out = model(input_ids = ids, labels = labels)
    out.loss.backward()
    grads = {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return out.logits.float().detach(), out.loss.item(), grads


@needs_models
@gpu
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v32", "glm_moe_dsa"])
def test_tiny_model_flex_on_vs_off(monkeypatch, family):
    """bf16 flex and bf16 dense, each against an fp32 dense run: flex must be no further off."""
    _patched()
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS", "65")
    if family == "minimax_m3_vl":
        config = _minimax_config(num_attention_heads = 8, num_key_value_heads = 2, head_dim = 64, hidden_size = 128)
        key = "minimax_m3_flex"
    else:
        config = _dsa_config(family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
        key = "dsa_flex"
    model = _model(family, config)
    S = 512
    ids = torch.randint(0, config.vocab_size, (2, S), device = "cuda")
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0")
    _, dense_loss, dense_grads = _step(model, ids, ids)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1")
    before = sb.SPARSE_FLEX_STATS[key]
    _, flex_loss, flex_grads = _step(model, ids, ids)
    assert sb.SPARSE_FLEX_STATS[key] >= before + 2, sb.SPARSE_FLEX_STATS
    assert not sb._FAILED, sb._FAILED
    model.float()
    _, ref_loss, ref_grads = _step(model, ids, ids)  # fp32: flex gated off, dense path
    names = [n for n in ref_grads if ref_grads[n].numel() and ref_grads[n].abs().max() > 0]
    flex_err = max(_rel(flex_grads[n], ref_grads[n]) for n in names)
    dense_err = max(_rel(dense_grads[n], ref_grads[n]) for n in names)
    print(f"{family}: loss fp32 {ref_loss:.5f} dense {dense_loss:.5f} flex {flex_loss:.5f}; "
          f"max grad rel vs fp32: dense {dense_err:.4f} flex {flex_err:.4f}")
    assert abs(flex_loss - ref_loss) <= abs(dense_loss - ref_loss) + 5e-3
    assert flex_err <= 1.5 * dense_err + 1e-2, (flex_err, dense_err)
