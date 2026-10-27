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


def _find_spec(name):
    # find_spec on a dotted name imports the parent package, which raises on transformers
    # releases that do not ship it (4.57.6 has no minimax_m3_vl).
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


minimax = _find_spec("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
dsa = _find_spec("transformers.models.deepseek_v32.modeling_deepseek_v32")
glm = _find_spec("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")


def _tested_transformers():
    from packaging.version import Version
    return Version(transformers.__version__.split("+")[0]) >= Version("5.17.0")


# The patches install only on transformers >= 5.17 (older indexers differ).
needs_models = pytest.mark.skipif(
    minimax is None or dsa is None or glm is None or not _tested_transformers(),
    reason = "transformers older than 5.17 or without MiniMax M3 / DSA models",
)
gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or getattr(torch.cuda, "_unsloth_consolidated_spoof", False)  # CPU runner with the CUDA spoof
    or torch.cuda.get_device_capability() < (8, 0),  # ROCm reports its gfx major (gfx1151: 11), and runs flex
    reason = "flex attention parity needs CUDA sm80+ or ROCm",
)

from unsloth_zoo.flex_attention import sparse_blocks as sb  # noqa: E402
from unsloth_zoo.flex_attention import sparse as sparse_gates  # noqa: E402


def _stat(kind, model_type):
    return sparse_gates.SPARSE_ATTN_STATS.get(kind, {}).get(model_type, 0)


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
@pytest.mark.parametrize("seq", [300, 333])  # above the 2 x 128 budget; within it the exact skip runs
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
    before = _stat("indexer_chunked", "minimax_m3_vl")
    got = indexer(h, (cos, sin), None, pos)
    assert _stat("indexer_chunked", "minimax_m3_vl") == before + 1
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
    assert not sparse_gates.sparse_flex_enabled()


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
    before = _stat("flex_reused", "dsa")
    a = sb.dsa_flex_attention(q, k, v, topk_idx, keep, D ** -0.5)
    b = sb.dsa_flex_attention(q, k, v, topk_idx, keep, D ** -0.5)
    assert _stat("flex_reused", "dsa") == before + 1
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
    else:
        config = _dsa_config(family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
    model = _model(family, config)
    S = 512
    ids = torch.randint(0, config.vocab_size, (2, S), device = "cuda")
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0")
    _, dense_loss, dense_grads = _step(model, ids, ids)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1")
    before = _stat("flex", family)
    _, flex_loss, flex_grads = _step(model, ids, ids)
    assert _stat("flex", family) >= before + 2, sparse_gates.SPARSE_ATTN_STATS
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


@needs_models
def test_dsa_cache_layout_matches_installed_source():
    s = _patched()
    import inspect
    from transformers.models.deepseek_v32 import modeling_deepseek_v32 as a
    source = inspect.getsource(a.DeepseekV32Attention._unsloth_upstream_forward)
    assert s._DSA_LATENT_CACHE["deepseek_v32"] == (s._DSA_CACHE_LATENT in source)


def test_dsa_flex_engages_above_the_budget(monkeypatch):
    from types import SimpleNamespace
    q = torch.randn(1, 4, 32, 16)
    mod = SimpleNamespace(config = SimpleNamespace(_attn_implementation = "sdpa", index_topk = 8), training = False, attention_dropout = 0.0)
    mask = torch.ones(1, 1, 32, 32, dtype = torch.bool)
    monkeypatch.setattr(sb, "sparse_flex_device_ok", lambda query: True)
    assert sb.dsa_flex_eligible(mod, q, q, q, mask, {}, None)
    assert not sb.dsa_flex_eligible(mod, q, q, q, mask, {}, 64)  # override above this key length
    k_short = torch.randn(1, 4, 8, 16)
    assert not sb.dsa_flex_eligible(mod, q, k_short, k_short, mask[..., :8], {}, None)  # within budget


@needs_models
@gpu
@pytest.mark.parametrize("family", ["deepseek_v32", "glm_moe_dsa", "minimax_m3_vl"])
def test_tiny_model_cached_decode_after_flex_prefill(monkeypatch, family):
    """Prefill above the budget (flex), then decode with the cache: same logits as flex off.

    MiniMax uses one layer: a second layer's indexer reselects blocks from bf16 hidden states,
    and near-tie block scores flip between any two attention backends."""
    _patched()
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS", "65")
    if family == "minimax_m3_vl":
        config = _minimax_config(num_attention_heads = 8, num_key_value_heads = 2, head_dim = 64, hidden_size = 128,
                                 num_hidden_layers = 1, layer_types = ["minimax_m3_sparse"], mlp_layer_types = ["dense"])
    else:
        config = _dsa_config(family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
    model = _model(family, config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 512), device = "cuda")
    nxt = torch.randint(0, config.vocab_size, (2, 3), device = "cuda")

    def run():
        with torch.no_grad():
            out = model(input_ids = ids, use_cache = True)
            past, logits = out.past_key_values, [out.logits[:, -1].float()]
            for i in range(nxt.shape[1]):
                out = model(input_ids = nxt[:, i : i + 1], past_key_values = past, use_cache = True)
                past = out.past_key_values
                logits.append(out.logits[:, -1].float())
        return torch.stack(logits)

    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0")
    ref = run()
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1")
    before = _stat("flex", family)
    got = run()
    assert _stat("flex", family) > before
    assert _rel(got, ref) < 5e-2, _rel(got, ref)


@needs_models
@pytest.mark.parametrize("family", ["deepseek_v32", "glm_moe_dsa", "minimax_m3_vl"])
@pytest.mark.parametrize("order", ["skip_first", "flex_first"])
def test_skip_and_flex_compose(order, family):
    """Both patch sets in either order: exact skip within the budget, chunked / flex above it."""
    import subprocess
    import sys
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sparse_compose_worker.py")
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_SPARSE_")}
    proc = subprocess.run([sys.executable, worker, order, family], capture_output = True, text = True, env = env, timeout = 600)
    assert "COMPOSE_OK" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]


@needs_models
@gpu
@pytest.mark.parametrize("family", ["minimax_m3_vl", "glm_moe_dsa"])
def test_flex_graphs_stay_bounded_over_lengths(family):
    """Varying lengths (incl. non multiples of 128) and batch 1 / 2, fwd + bwd: at most four graphs (static
    first length, then dynamic, batch 1 vs > 1), none added per new length after that."""
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    dt = torch.bfloat16

    def call(S, B):
        if family == "minimax_m3_vl":
            nb = -(-S // 128)
            causal = torch.arange(nb, device = "cuda") > (torch.arange(S, device = "cuda") // 128)[:, None]
            v, blocks = torch.rand(B, 2, S, nb, device = "cuda").masked_fill(causal, float("-inf")).topk(2, -1)
            blocks = blocks.masked_fill(v == float("-inf"), -1)
            pos = torch.arange(S, device = "cuda")[None].expand(B, -1)
            q = torch.randn(B, 8, S, 64, device = "cuda", dtype = dt, requires_grad = True)
            k = torch.randn(B, 2, S, 64, device = "cuda", dtype = dt, requires_grad = True)
            out = sb.minimax_m3_flex_attention(q, k, k, blocks, None, pos, 0.125, 128)
        else:
            q = torch.randn(B, 4, S, 64, device = "cuda", dtype = dt, requires_grad = True)
            k = torch.randn(B, 4, S, 64, device = "cuda", dtype = dt, requires_grad = True)
            keep = torch.ones(S, S, dtype = torch.bool, device = "cuda").tril()[None, None].expand(B, 1, S, S)
            topk = torch.randn(B, S, S, device = "cuda").masked_fill(~keep[:, 0], float("-inf")).topk(64, -1).indices.int()
            out = sb.dsa_flex_attention(q, k, k, topk, keep, 0.125, family)
        out.float().sum().backward()

    for B in (1, 2):
        for S in (300, 384, 513, 777):
            call(S, B)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    for S in (260, 640, 1000, 1025):
        call(S, 2)
        call(S, 1)
    assert graphs <= 4, graphs
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


def _packed_logits(model, ids, pos, docs, packed):
    pieces = [(0, ids.shape[1])] if packed else [(sum(docs[:i]), sum(docs[: i + 1])) for i in range(len(docs))]
    out = []
    for a, b in pieces:
        with torch.no_grad():
            out.append(model(input_ids = ids[:, a:b], position_ids = pos[:, a:b] if packed else None, use_cache = False).logits.float())
    return torch.cat(out, 1)


@needs_models
@gpu
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v32", "glm_moe_dsa"])
def test_packed_sequences(monkeypatch, family):
    """Padding-free packing (position_ids reset per document, no attention_mask): transformers builds a
    block-diagonal mask. DSA keeps documents apart, so packed flex must match each document run alone.
    MiniMax's indexer anchors blocks to absolute key slots and mixes documents upstream: flex must match
    upstream's packed result instead."""
    _patched()
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS", "65")
    if family == "minimax_m3_vl":
        config = _minimax_config(num_attention_heads = 8, num_key_value_heads = 2, head_dim = 64, hidden_size = 128)
        docs = [400, 150, 333]
    else:
        config = _dsa_config(family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
        docs = [150, 40, 120]
    model = _model(family, config)
    ids = torch.randint(0, config.vocab_size, (1, sum(docs)), device = "cuda")
    pos = torch.cat([torch.arange(n, device = "cuda") for n in docs])[None]
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0")
    upstream = _packed_logits(model, ids, pos, docs, packed = family == "minimax_m3_vl")
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1")
    before = _stat("flex", family)
    got = _packed_logits(model, ids, pos, docs, packed = True)
    assert _stat("flex", family) >= before + 2, sparse_gates.SPARSE_ATTN_STATS
    err = ((got - upstream).norm() / upstream.norm()).item()
    assert err < 3e-2, err


@needs_models
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v32", "glm_moe_dsa"])
def test_indexer_forward_without_class_attributes(monkeypatch, family):
    """Unsloth's compiler rebuilds the classes from source, dropping attributes set on them at patch time:
    the patched indexer forwards must not read them."""
    _patched()
    config = _minimax_config() if family == "minimax_m3_vl" else _dsa_config(family)
    m, indexer = _indexer_module(family, config)
    cls = type(indexer)
    for name in ("_unsloth_inner_forward", "_unsloth_dsa_family"):
        monkeypatch.delattr(cls, name, raising = False)
    S = 128 if family == "minimax_m3_vl" else 48  # within the budget: the exact skip / upstream forward
    (cos, sin), pos = _rope(config, family, S)
    x = torch.randn(1, S, config.hidden_size)
    with torch.no_grad():
        if family == "minimax_m3_vl":
            out = indexer(x, (cos, sin), None, pos)
        else:
            mask = torch.zeros(1, S, S).masked_fill(~torch.ones(S, S, dtype = torch.bool).tril(), float("-inf"))
            out = indexer(x, torch.randn(1, S, config.q_lora_rank), (cos, sin), mask, pos)
    assert out.shape[1] == (S if family != "minimax_m3_vl" else config.index_n_heads)


@needs_models
@gpu
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v32"])
def test_flex_runs_outside_unsloth_block_compile(monkeypatch, family):
    """Unsloth compiles each decoder layer (fullgraph=False). Above the budget the flex call leaves that graph
    (one break) and engages, matching eager flex; within the budget nothing breaks the graph. Without the
    block flag a compiled call keeps the model's path."""
    import sys
    import types
    import torch._dynamo
    from torch._dynamo.utils import counters
    _patched()
    torch.manual_seed(0)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS", "65")
    if family == "minimax_m3_vl":
        config = _minimax_config(hidden_size = 128, head_dim = 64)
        m = importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
        attn = m.MiniMaxM3VLAttention(config, 0)
        within = 200  # budget 2 x 128 keys
    else:
        config = _dsa_config(family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
        m = importlib.import_module("transformers.models.deepseek_v32.modeling_deepseek_v32")
        attn = m.DeepseekV32Attention(config, 0)
        within = 48  # budget 64 keys
    attn = attn.cuda().bfloat16()
    block = types.ModuleType("unsloth.models._block_compile")
    block._IN_BLOCK = [0]
    monkeypatch.setitem(sys.modules, "unsloth.models._block_compile", block)

    def run(S, compiled, in_block):
        (cos, sin), pos = _rope(config, family, S, "cuda")
        torch.manual_seed(1)
        x = torch.randn(1, S, config.hidden_size, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        mask = torch.zeros(1, 1, S, S, device = "cuda", dtype = torch.bfloat16).masked_fill(
            ~torch.ones(S, S, dtype = torch.bool, device = "cuda").tril(), float("-inf"))

        def fn(t):
            return attn(t, (cos.bfloat16(), sin.bfloat16()), mask, position_ids = pos)[0]

        torch._dynamo.reset()
        counters.clear()
        if compiled:
            fn = torch.compile(fn, fullgraph = False)
        block._IN_BLOCK[0] = 1 if in_block else 0
        try:
            out = fn(x)
        finally:
            block._IN_BLOCK[0] = 0
        out.float().square().mean().backward()
        return out.detach().float(), x.grad.float(), sum(counters["graph_break"].values())

    def nrel(a, b):
        return ((a - b).norm() / b.norm()).item()

    eager, eager_g, _ = run(512, False, False)
    before = _stat("flex", family)
    got, got_g, breaks = run(512, True, True)
    assert _stat("flex", family) == before + 1
    assert breaks >= 1
    assert nrel(got, eager) < 2e-2 and nrel(got_g, eager_g) < 2e-2, (nrel(got, eager), nrel(got_g, eager_g))
    before = _stat("flex", family)
    run(512, True, False)
    assert _stat("flex", family) == before  # any other compile: the model's path, as before
    _, _, breaks = run(within, True, True)
    assert breaks == 0, breaks
    assert not sb._FAILED, sb._FAILED


@needs_models
@gpu
@pytest.mark.skipif(
    torch.version.hip is not None,
    reason = "ROCm compiled-vs-compiled DSA differs by ~5% with flex off too (Inductor reorders the top-k)",
)
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v32"])
def test_attention_inside_torch_compile_matches_eager(monkeypatch, family):
    """An attention layer under an enclosing torch.compile (fullgraph, dynamic): the BlockMask build plus
    flex call miscompiled there (MiniMax output 95% off, silently), so the model's own path runs inside a
    compiled graph. Compiled with flex on == compiled upstream, and eager flex stays close to eager upstream."""
    import torch._dynamo
    _patched()
    torch.manual_seed(0)
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_DSA_MIN_KEYS", "65")
    S = 512
    if family == "minimax_m3_vl":
        config = _minimax_config(hidden_size = 128, head_dim = 64)
        m = importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")
        attn = m.MiniMaxM3VLAttention(config, 0)
    else:
        config = _dsa_config(family, qk_nope_head_dim = 32, qk_rope_head_dim = 32, v_head_dim = 64)
        m = importlib.import_module("transformers.models.deepseek_v32.modeling_deepseek_v32")
        attn = m.DeepseekV32Attention(config, 0)
    attn = attn.cuda().bfloat16()
    (cos, sin), pos = _rope(config, family, S, "cuda")
    rope = (cos.bfloat16(), sin.bfloat16())
    x = torch.randn(1, S, config.hidden_size, device = "cuda", dtype = torch.bfloat16)
    mask = torch.zeros(1, 1, S, S, device = "cuda", dtype = torch.bfloat16).masked_fill(
        ~torch.ones(S, S, dtype = torch.bool, device = "cuda").tril(), float("-inf"))

    def run(flex, compiled):
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1" if flex else "0")
        torch._dynamo.reset()
        attn.zero_grad(set_to_none = True)
        xx = x.detach().clone().requires_grad_()

        def fn(t):
            return attn(t, rope, mask, position_ids = pos)[0]

        if compiled:
            fn = torch.compile(fn, fullgraph = True, dynamic = True)
        out = fn(xx)
        out.float().square().mean().backward()
        return out.detach().float(), xx.grad.float()

    def nrel(a, b):
        return ((a - b).norm() / b.norm()).item()

    ref, ref_g = run(False, False)
    before = _stat("flex", family)
    flex, flex_g = run(True, False)
    assert _stat("flex", family) == before + 1
    compiled_ref, compiled_ref_g = run(False, True)
    got, got_g = run(True, True)
    # Inside the compiled graph the patch keeps the model's path: same result as compiling upstream.
    assert nrel(got, compiled_ref) < 1e-3 and nrel(got_g, compiled_ref_g) < 1e-3, (nrel(got, compiled_ref), nrel(got_g, compiled_ref_g))
    # (Compiled upstream vs eager upstream is not compared: Inductor's fp32 indexer scores can reorder the
    # DSA top-k, ~5% here, which is upstream's own behaviour.)
    assert nrel(flex, ref) < 3e-2 and nrel(flex_g, ref_g) < 3e-2, (nrel(flex, ref), nrel(flex_g, ref_g))
    assert not sb._FAILED, sb._FAILED


@needs_models
@pytest.mark.skipif(
    getattr(torch.cuda, "_unsloth_consolidated_spoof", False),
    reason = "Inductor initialises the spoofed CUDA device",
)
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v32"])
def test_chunked_indexer_compiles_once_over_lengths(monkeypatch, family):
    """Inside torch.compile (dynamic) the chunk loop's trip count specialised the graph on the length:
    one graph per length. One graph now, same selection as eager."""
    import torch._dynamo
    monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK", "64")
    torch.manual_seed(0)
    config = _minimax_config() if family == "minimax_m3_vl" else _dsa_config(family)
    m, indexer = _indexer_module(family, config)
    graphs = []

    def backend(gm, example):
        graphs.append(gm)
        return gm.forward

    def fn(x, qr, cos, sin, mask, pos):
        if family == "minimax_m3_vl":
            return sb.minimax_m3_select_blocks(indexer, x, (cos, sin), None, pos, m.apply_rotary_pos_emb)
        return sb.dsa_select_topk(indexer, x, qr, (cos, sin), mask, None, m.apply_rotary_pos_emb)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend = backend, fullgraph = True, dynamic = True)
    for S in (257, 289, 321, 353, 385):
        (cos, sin), pos = _rope(config, family, S)
        x = torch.randn(1, S, config.hidden_size)
        qr = torch.randn(1, S, config.q_lora_rank if family != "minimax_m3_vl" else 32)
        mask = torch.zeros(1, S, S).masked_fill(~torch.ones(S, S, dtype = torch.bool).tril(), float("-inf"))
        got = compiled(x, qr, cos, sin, mask, pos)
        assert torch.equal(got, fn(x, qr, cos, sin, mask, pos))
    assert len(graphs) == 1, len(graphs)


@needs_models
@gpu
def test_dsa_flex_keeps_finite_mask_biases_in_float32():
    """A float DSA mask is additive upstream: finite entries (here -0.5 / +0.25 on visible keys) shift the
    scores. They were read as hard exclusions (output ~59% off)."""
    torch.manual_seed(0)
    B, H, S, D, topk = 2, 4, 384, 64, 96
    keep = torch.ones(S, S, dtype = torch.bool, device = "cuda").tril()[None, None].expand(B, 1, S, S)
    bias = torch.where(torch.rand(B, 1, S, S, device = "cuda") < 0.5, -0.5, 0.25)
    mask = torch.where(keep, bias, torch.finfo(torch.float32).min)
    topk_idx = torch.randn(B, S, S, device = "cuda").masked_fill(~keep[:, 0], float("-inf")).topk(topk, -1).indices.int()
    index_mask = torch.ones(B, S, S, dtype = torch.bool, device = "cuda").scatter(-1, topk_idx.long(), False).unsqueeze(1)
    dense = mask.masked_fill(index_mask, torch.finfo(torch.float32).min)
    q, k, v = (torch.randn(B, H, S, D, device = "cuda") for _ in range(3))
    ref, ref_g = _grads(lambda a, b, c: _dense(a, b, c, dense, 0.125), q, k, v)
    got, got_g = _grads(lambda a, b, c: sb.dsa_flex_attention(a, b, c, topk_idx, mask, 0.125), q, k, v)
    # fp32 flex runs its dots in TF32 on torch 2.10 (7e-4 here); dropping the biases was 0.59.
    assert _rel(got, ref) < 2e-3, _rel(got, ref)
    for name, a, b in zip("qkv", got_g, ref_g):
        assert _rel(a, b) < 5e-3, (name, _rel(a, b))


@needs_models
@gpu
@pytest.mark.parametrize("shape", ["query", "key", "batch"])
def test_broadcast_masks_match_their_expansion(monkeypatch, shape):
    """Callers' masks broadcast upstream over queries ([B, 1, 1, T]), keys ([B, 1, S, 1]) or batch
    ([1, 1, S, T]). Padding a singleton dim hid keys (only key 0, or only query 0, stayed visible), and the
    chunked DSA indexer failed slicing a query-singleton mask on the second chunk."""
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    B, H, S, D = 2, 4, 512, 64
    if shape == "query":
        pad = torch.ones(B, 1, 1, S, dtype = torch.bool, device = dev)
        pad[1, ..., S - 100:] = False  # right padding on row 1
    elif shape == "key":
        pad = torch.ones(B, 1, S, 1, dtype = torch.bool, device = dev)
        pad[1, :, S - 100:] = False  # padded query rows on row 1
    else:
        pad = torch.ones(S, S, dtype = torch.bool, device = dev).tril()[None, None]
    for base in (pad, torch.zeros(pad.shape, device = dev, dtype = dt).masked_fill(~pad, torch.finfo(dt).min)):
        full = base.expand(B, 1, S, S).contiguous()
        q, k, v = (torch.randn(B, H, S, D, device = dev, dtype = dt) for _ in range(3))
        topk_idx = torch.randn(B, S, S, device = dev).topk(128, dim = -1).indices.int()
        got = sb.dsa_flex_attention(q, k, v, topk_idx, base, D ** -0.5)
        ref = sb.dsa_flex_attention(q, k, v, topk_idx.clone(), full, D ** -0.5)
        assert _rel(got, ref) < 1e-3, _rel(got, ref)
        blocks = torch.arange(S // 128, device = dev).expand(B, H, S, S // 128).clone()
        pos = torch.arange(S, device = dev)[None].expand(B, -1)
        got = sb.minimax_m3_flex_attention(q, k, v, blocks, base, pos, D ** -0.5, 128)
        ref = sb.minimax_m3_flex_attention(q, k, v, blocks, full, pos, D ** -0.5, 128)
        assert _rel(got, ref) < 1e-3, _rel(got, ref)

    if shape != "query":
        return
    _patched()
    config = _dsa_config("deepseek_v32")
    m, indexer = _indexer_module("deepseek_v32", config)
    seq = 300
    (cos, sin), pos = _rope(config, "deepseek_v32", seq)
    h = torch.randn(2, seq, config.hidden_size)
    q_resid = torch.randn(2, seq, config.q_lora_rank)
    mask = torch.ones(2, 1, seq, dtype = torch.bool)
    mask[1, :, seq - 60:] = False
    monkeypatch.setenv("UNSLOTH_SPARSE_INDEXER_Q_CHUNK", "64")
    got = indexer(h, q_resid, (cos, sin), mask, pos)
    ref = indexer(h, q_resid, (cos, sin), mask.expand(2, seq, seq).contiguous(), pos)
    assert torch.equal(got.sort(dim = -1).values, ref.sort(dim = -1).values)


@gpu  # without CUDA flex is unavailable anyway, so the check would pass vacuously
@pytest.mark.parametrize("value, expected", [("1", False), ("partial", False), ("0", None)])
def test_compile_disable_turns_sparse_flex_off(monkeypatch, value, expected):
    """UNSLOTH_COMPILE_DISABLE=1 and =partial both leave flex uncompiled (dense score math): stay on the model path."""
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", value)
    sb._flex_available.cache_clear()
    try:
        got = sb._flex_available()
    finally:
        sb._flex_available.cache_clear()
    if expected is not None:
        assert got is expected


@needs_models
@gpu
@pytest.mark.parametrize("family", ["minimax_m3_vl", "deepseek_v32"])
def test_config_output_attentions_keeps_the_model_path(family):
    """transformers records attention weights when config.output_attentions is set, without the kwarg."""
    import types
    q = torch.randn(1, 4, 512, 64, device = "cuda", dtype = torch.bfloat16)
    config = types.SimpleNamespace(_attn_implementation = "sdpa", index_topk = 64, output_attentions = True)
    if family == "minimax_m3_vl":
        module = types.SimpleNamespace(config = config, training = False, attention_dropout = 0.0,
                                       indexer = types.SimpleNamespace(topk_blocks = 2, block_size = 128))
        assert not sb.minimax_m3_flex_eligible(module, q, q, None, {})
        config.output_attentions = False
        assert sb.minimax_m3_flex_eligible(module, q, q, None, {})
    else:
        module = types.SimpleNamespace(config = config, training = False, attention_dropout = 0.0)
        mask = torch.zeros(1, 1, 512, 512, device = "cuda", dtype = torch.bfloat16)
        assert not sb.dsa_flex_eligible(module, q, q, q, mask, {}, None)
        config.output_attentions = False
        assert sb.dsa_flex_eligible(module, q, q, q, mask, {}, None)


@needs_models
@gpu
@pytest.mark.parametrize("finite_bias", [False, True])
def test_dsa_flex_cuda_graph_capture_matches_eager(finite_bias):
    """The finite-bias check is a host sync; under stream capture it invalidated the capture
    (cudaErrorStreamCaptureInvalidated). Capturing keeps the bias instead: the replay matches the eager call."""
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    B, H, S, T, D = 1, 4, 64, 1024, 64
    q = torch.randn(B, H, S, D, device = dev, dtype = dt)
    k = torch.randn(B, H, T, D, device = dev, dtype = dt)
    topk_idx = torch.randperm(T, device = dev)[:256].sort().values.int().expand(B, S, -1)
    mask = torch.zeros(B, 1, S, T, device = dev, dtype = dt)
    mask[..., -64:] = torch.finfo(dt).min
    if finite_bias:
        mask = mask + torch.rand_like(mask).mul(0.5)

    def fn():
        if hasattr(topk_idx, "_unsloth_dsa_flex"):
            del topk_idx._unsloth_dsa_flex  # rebuild the block mask (and run the bias check) every call
        return sb.dsa_flex_attention(q, k, k, topk_idx, mask, D ** -0.5)

    eager = fn().clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream = stream):
        out = fn()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert _rel(out, eager) < 1e-2, _rel(out, eager)


@needs_models
@gpu
def test_dsa_flex_capture_same_shape_mixed_bias():
    """Two same-shape calls in one capture, one with a finite bias and one 0 / min only: the warm-up of the
    biased call must not be overwritten by the plain one (a stale no-bias answer asserted on replay)."""
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    B, H, S, T, D = 1, 4, 64, 1024, 64
    q = torch.randn(B, H, S, D, device = dev, dtype = dt)
    k = torch.randn(B, H, T, D, device = dev, dtype = dt)
    topk_idx = torch.randperm(T, device = dev)[:256].sort().values.int().expand(B, S, -1)
    plain = torch.zeros(B, 1, S, T, device = dev, dtype = dt)
    plain[..., -64:] = torch.finfo(dt).min
    biased = plain + torch.rand_like(plain).mul(0.5)

    def fn():
        outs = []
        for mask in (biased, plain):
            if hasattr(topk_idx, "_unsloth_dsa_flex"):
                del topk_idx._unsloth_dsa_flex
            outs.append(sb.dsa_flex_attention(q, k, k, topk_idx, mask, D ** -0.5))
        return outs

    eager = [o.clone() for o in fn()]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream = stream):
        outs = fn()
    for o in outs:
        o.zero_()
    graph.replay()
    torch.cuda.synchronize()
    for o, e in zip(outs, eager):
        assert _rel(o, e) < 1e-2, _rel(o, e)


def test_flex_failure_under_capture_is_raised_not_latched(monkeypatch):
    """A failure while a CUDA graph is being captured has invalidated the capture, so the dense fallback cannot
    run either: it is raised, and the shape class stays on flex."""
    import types
    from unsloth_zoo.temporary_patches import sparse_attention_flex as s
    query = types.SimpleNamespace(
        is_cuda = True, device = types.SimpleNamespace(index = 0), dtype = torch.bfloat16, shape = (1, 4, 64, 64),
    )

    def fail():
        raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match = "capturing"):
        s._flex_or_none("dsa", "DSA", fail, query, query, query)
    assert not sb._FAILED
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert s._flex_or_none("dsa", "DSA", fail, query, query, query) is None
    assert len(sb._FAILED) == 1
