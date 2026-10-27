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

"""DeepSeek-V4 attention on flex attention: routing, block mask, and parity with transformers' eager
attention (fp32 exact-math check, bf16 layers against an fp32 oracle, prefill + decode through the cache)."""

import copy

import types

import pytest
import torch

modeling = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
import transformers  # noqa: E402
from packaging.version import Version  # noqa: E402

if Version(transformers.__version__).release < (5, 17):
    pytest.skip("the patch installs on transformers >= 5.17 only", allow_module_level = True)

from unsloth_zoo.temporary_patches import deepseek_v4_sparse_flex as F4  # noqa: E402

F4.patch_deepseek_v4_sparse_flex()
# The patch declines (torch < 2.10, no flex BlockMask, changed upstream source); then there is nothing to test.
if not hasattr(modeling.eager_attention_forward, "_unsloth_original"):
    pytest.skip("the DeepSeek-V4 flex patch did not install here", allow_module_level = True)
ORIGINAL = modeling.eager_attention_forward._unsloth_original

CUDA = torch.cuda.is_available() and not getattr(torch.cuda, "_unsloth_consolidated_spoof", False)
needs_flex_gpu = pytest.mark.skipif(
    not CUDA
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() < (8, 0)
    # L4 / A10 / RTX 30-40 (~101 KB) are sm80+ but the patch keeps them on eager.
    or getattr(torch.cuda.get_device_properties(0), "shared_memory_per_block_optin", 0) < F4._MIN_SHARED_MEMORY,
    reason = "flex attention path needs an sm80+ NVIDIA GPU with enough shared memory",
)


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_FLEX", raising = False)
    F4.DEEPSEEK_V4_FLEX_STATS.update(flex = 0, eager = 0)
    F4._DEVICE_OK.clear()
    getattr(F4, "_FAILED", {}).clear()
    from unsloth_zoo.flex_attention import sparse as sparse_gates
    getattr(sparse_gates, "_BACKWARD_OK", set()).clear()
    # Small shapes keep the tests fast; the default policy is checked on its own below.
    monkeypatch.setattr(F4, "_MIN_KEYS", {8: {None: 16}, 9: {None: 16}, 10: {None: 16}})
    yield


class _Attn(torch.nn.Module):
    def __init__(self, heads, device):
        super().__init__()
        self.sinks = torch.nn.Parameter(torch.randn(heads, device = device))
        self.num_key_value_groups = heads


def _mask(bsz, q_len, window, compressed, rate, topk, dtype, device, pad = 0, seed = 0):
    """Additive [B, 1, S, S + C] mask like V4's: sliding window + left padding over tokens, then causal
    top-k over compressed entries (0 = attend, finfo.min / -inf = masked)."""
    g = torch.Generator(device = "cpu").manual_seed(seed)
    q = torch.arange(q_len)[:, None]
    k = torch.arange(q_len)[None]
    tokens = ((q - k) >= 0) & ((q - k) < window)
    tokens = tokens[None].expand(bsz, -1, -1).clone()
    if pad:
        tokens[1, :, :pad] = False
    visible = torch.arange(compressed)[None] < ((torch.arange(q_len) + 1) // rate)[:, None]
    scores = torch.rand(bsz, q_len, compressed, generator = g).masked_fill(~visible, -1)
    picked = torch.zeros(bsz, q_len, compressed, dtype = torch.bool)
    picked.scatter_(-1, scores.topk(min(topk, compressed), -1).indices, True)
    picked &= visible
    token_bias = torch.zeros(bsz, 1, q_len, q_len, dtype = dtype).masked_fill(~tokens[:, None], torch.finfo(dtype).min)
    comp_bias = torch.zeros(bsz, 1, q_len, compressed, dtype = dtype).masked_fill(~picked[:, None], float("-inf"))
    return torch.cat([token_bias, comp_bias], -1).to(device)


def test_patch_is_installed_and_idempotent():
    forward = modeling.eager_attention_forward
    assert forward._unsloth_deepseek_v4_flex
    F4.patch_deepseek_v4_sparse_flex()
    assert modeling.eager_attention_forward is forward
    assert forward._unsloth_original is ORIGINAL


def _gate_inputs(device, q_len = 32, dtype = torch.bfloat16):
    q = torch.randn(1, 4, q_len, 64, device = device, dtype = dtype)
    kv = torch.randn(1, 1, q_len + 8, 64, device = device, dtype = dtype)
    mask = _mask(1, q_len, 16, 8, 4, 4, dtype, device)
    return _Attn(4, device).to(dtype), q, kv, mask


def test_cpu_and_gates_fall_back_to_eager(monkeypatch):
    attn, q, kv, mask = _gate_inputs("cpu")
    out, weights = modeling.eager_attention_forward(attn, q, kv, kv, mask, 0.125)
    ref, ref_weights = ORIGINAL(attn, q, kv, kv, mask, 0.125)
    assert torch.equal(out, ref) and torch.equal(weights, ref_weights)
    assert F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 0, "eager": 1}


@needs_flex_gpu
@pytest.mark.parametrize("case", ["env_off", "dropout", "output_attentions", "config_output_attentions", "short_query", "k_not_v", "no_mask", "fp32"])
def test_gpu_gates_fall_back_to_eager(monkeypatch, case):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0" if case == "env_off" else "1")
    attn, q, kv, mask = _gate_inputs("cuda", q_len = 8 if case == "short_query" else 32,
                                     dtype = torch.float32 if case == "fp32" else torch.bfloat16)
    value = kv.clone() if case == "k_not_v" else kv
    kwargs = {"output_attentions": True} if case == "output_attentions" else {}
    if case == "config_output_attentions":
        attn.config = types.SimpleNamespace(output_attentions = True)
    modeling.eager_attention_forward(attn, q, kv, value, None if case == "no_mask" else mask, 0.125,
                                     dropout = 0.1 if case == "dropout" else 0.0, **kwargs)
    assert F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 0, "eager": 1}


class _Props:
    def __init__(self, major, minor, smem):
        self.major, self.minor, self.shared_memory_per_block_optin = major, minor, smem


@pytest.mark.parametrize("major, minor, smem, forced, layer, expected", [
    (10, 0, 232448, False, "compressed_sparse_attention", 4096),
    (10, 0, 232448, False, "sliding_attention", 4096),
    (9, 0, 232448, False, "compressed_sparse_attention", 4096),
    (9, 0, 232448, False, "heavily_compressed_attention", 4096),
    (8, 0, 166912, False, "compressed_sparse_attention", 8192),
    (8, 0, 166912, False, "heavily_compressed_attention", 4096),
    (8, 0, 166912, False, "sliding_attention", 4096),
    (8, 9, 101376, False, "sliding_attention", None),  # L4 / RTX 4090
    (8, 9, 101376, True, "sliding_attention", None),  # too little shared memory even when forced
    (8, 6, 101376, False, "sliding_attention", None),
    (12, 0, 101376, True, "sliding_attention", None),
    (7, 5, 65536, True, "sliding_attention", None),  # T4
    (12, 1, 232448, False, "sliding_attention", None),  # unmeasured generation
    (12, 1, 232448, True, "sliding_attention", 4096),
])
def test_default_policy_table(monkeypatch, major, minor, smem, forced, layer, expected):
    monkeypatch.undo()
    F4._DEVICE_OK.clear()
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _Props(major, minor, smem))
    module = type("M", (), {"layer_type": layer})()
    assert F4._min_keys(module, torch.device("cuda", 0), forced) == expected
    if expected is not None:
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX_MIN_KEYS", "1234")
        assert F4._min_keys(module, torch.device("cuda", 0), forced) == 1234
    F4._DEVICE_OK.clear()


def test_rocm_stays_eager(monkeypatch):
    monkeypatch.undo()
    F4._DEVICE_OK.clear()
    monkeypatch.setattr(torch.version, "hip", "6.4")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: _Props(9, 4, 65536 * 4))
    assert F4._min_keys(type("M", (), {})(), torch.device("cuda", 0), True) is None
    F4._DEVICE_OK.clear()


@needs_flex_gpu
def test_default_policy_keeps_short_prompts_eager(monkeypatch):
    monkeypatch.undo()
    attn, q, kv, mask = _gate_inputs("cuda", q_len = 300)
    modeling.eager_attention_forward(attn, q, kv, kv, mask, 0.125)
    assert F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 0, "eager": 1}


@needs_flex_gpu
def test_block_mask_matches_the_additive_mask():
    mask = _mask(2, 300, 64, 75, 4, 16, torch.bfloat16, "cuda", pad = 50)
    block_mask, _ = F4.deepseek_v4_block_mask(mask, 2)
    dense = block_mask.to_dense()  # [B, 1, nq, nk] tiles that are visited
    allowed = mask[:, 0] == 0
    nq, nk = dense.shape[-2:]
    padded = torch.zeros(2, nq * 128, nk * 128, dtype = torch.bool, device = "cuda")
    padded[:, : allowed.shape[1], : allowed.shape[2]] = allowed
    needed = padded.view(2, nq, 128, nk, 128).any(4).any(2)
    assert torch.equal(dense[:, 0].bool(), needed)
    for count, index in ((block_mask.kv_num_blocks, block_mask.kv_indices),
                         (block_mask.full_kv_num_blocks, block_mask.full_kv_indices)):
        for b in range(2):
            for row in range(nq):
                used = index[b, 0, row, : count[b, 0, row]]
                assert torch.all(used[1:] > used[:-1])


def _function_case(dtype, seed = 0):
    torch.manual_seed(seed)
    bsz, heads, q_len, head_dim, comp = 2, 8, 300, 128, 75
    mask = _mask(bsz, q_len, 64, comp, 4, 16, dtype, "cuda", pad = 60, seed = seed)
    q = (torch.randn(bsz, heads, q_len, head_dim, device = "cuda") * 2).to(dtype).requires_grad_(True)
    kv = torch.randn(bsz, 1, q_len + comp, head_dim, device = "cuda").to(dtype).requires_grad_(True)
    attn = _Attn(heads, "cuda").to(dtype)
    grad = torch.randn(bsz, q_len, heads, head_dim, device = "cuda").to(dtype)
    return attn, q, kv, mask, grad


def _run(fn, attn, q, kv, mask, grad):
    attn.zero_grad(set_to_none = True)
    q.grad = kv.grad = None
    out = fn()
    (out.float() * grad.float()).sum().backward()
    return out.detach().float(), q.grad.float(), kv.grad.float(), attn.sinks.grad.float()


@needs_flex_gpu
def test_flex_equals_eager_in_float32():
    """Same math in fp32 (sliding window + padding + top-k compressed entries + sink): only the
    reduction order differs, including the fully masked rows of the left-padded batch row."""
    attn, q, kv, mask, grad = _function_case(torch.float32)
    scale = q.shape[-1] ** -0.5
    ref = _run(lambda: ORIGINAL(attn, q, kv, kv, mask, scale)[0], attn, q, kv, mask, grad)
    got = _run(lambda: F4.deepseek_v4_flex_attention(attn, q, kv, mask, scale), attn, q, kv, mask, grad)
    for name, a, b in zip(("out", "dq", "dkv", "dsinks"), got, ref):
        err = ((a - b).norm() / b.norm()).item()
        assert err < 2e-5, (name, err)


@needs_flex_gpu
def test_flex_keeps_finite_mask_biases_in_float32():
    """A custom additive mask with finite biases (here -1 and +0.5 on visible keys) adds to the logits like eager."""
    attn, q, kv, mask, grad = _function_case(torch.float32)
    torch.manual_seed(1)
    visible = mask > torch.finfo(mask.dtype).min / 2
    bias = torch.where(torch.rand_like(mask) < 0.5, -1.0, 0.5)
    mask = torch.where(visible, mask + bias, mask)
    scale = q.shape[-1] ** -0.5
    ref = _run(lambda: ORIGINAL(attn, q, kv, kv, mask, scale)[0], attn, q, kv, mask, grad)
    got = _run(lambda: F4.deepseek_v4_flex_attention(attn, q, kv, mask, scale), attn, q, kv, mask, grad)
    for name, a, b in zip(("out", "dq", "dkv", "dsinks"), got, ref):
        err = ((a - b).norm() / b.norm()).item()
        assert err < 2e-5, (name, err)


@needs_flex_gpu
def test_flex_compiles_fullgraph_with_a_biased_mask():
    """Inside an enclosing fullgraph compile the bias check must not branch on data; output and dq match
    eager-called flex. Compiled inference (no_grad) goes to eager instead of failing to lower."""
    attn, q, kv, mask, _ = _function_case(torch.bfloat16)
    visible = mask > torch.finfo(mask.dtype).min / 2
    mask = torch.where(visible, mask - 0.5, mask)
    scale = q.shape[-1] ** -0.5

    def fn(q, kv, mask):
        return F4.deepseek_v4_flex_attention(attn, q, kv, mask, scale)

    def dispatch(q, kv, mask):
        return modeling.eager_attention_forward(attn, q, kv, kv, mask, scale)[0]

    torch._dynamo.reset()
    outs = []
    for f in (fn, torch.compile(fn, fullgraph = True, dynamic = True)):
        qq = q.detach().requires_grad_(True)
        y = f(qq, kv.detach(), mask)
        y.float().square().mean().backward()
        outs.append((y.detach().float(), qq.grad.float()))
    for got, ref in zip(outs[1], outs[0]):
        assert (got - ref).norm() / ref.norm() < 2e-2
    with torch.no_grad():
        ref = ORIGINAL(attn, q, kv, kv, mask, scale)[0].float()
        got = torch.compile(dispatch, fullgraph = True, dynamic = True)(q, kv, mask).float()
    assert (got - ref).norm() / ref.norm() < 2e-2


def _rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()


def _tiny_config(topk = 16):
    from transformers import DeepseekV4Config
    config = DeepseekV4Config(
        hidden_size = 256, num_attention_heads = 64, num_key_value_heads = 1, head_dim = 512,
        q_lora_rank = 64, o_groups = 8, o_lora_rank = 32, index_n_heads = 4, index_head_dim = 128,
        index_topk = topk, sliding_window = 128, num_hidden_layers = 4, vocab_size = 512,
        moe_intermediate_size = 64, n_routed_experts = 4, num_experts_per_tok = 2,
        num_nextn_predict_layers = 0,
        layer_types = ["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention", "sliding_attention"],
    )
    config._attn_implementation = "eager"
    return config


def _tiny_model():
    torch.manual_seed(0)
    model = modeling.DeepseekV4ForCausalLM(_tiny_config()).cuda().float()
    for name, param in model.named_parameters():
        if "sinks" in name:
            torch.nn.init.normal_(param, 0, 1.0)
    return model


def _to(x, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, dict):
        return {k: _to(v, dtype) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(_to(v, dtype) for v in x)
    return x


def _captured_layers(monkeypatch):
    """Inputs every attention layer of the shrunk model receives on a left-padded batch."""
    model = _tiny_model()
    bsz, q_len, pad = 2, 1280, 200
    ids = torch.randint(0, 512, (bsz, q_len), device = "cuda")
    mask = torch.ones(bsz, q_len, dtype = torch.long, device = "cuda")
    mask[1, :pad] = 0
    seen = {}
    hooks = [
        layer.self_attn.register_forward_pre_hook(
            (lambda i: lambda mod, args, kwargs: seen.__setitem__(i, (args, kwargs)))(i), with_kwargs = True)
        for i, layer in enumerate(model.model.layers)
    ]
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "0")
    with torch.no_grad():
        model(input_ids = ids, attention_mask = mask, use_cache = False)
    for hook in hooks:
        hook.remove()
    weight = torch.randn(bsz, q_len, 256, device = "cuda", generator = torch.Generator("cuda").manual_seed(1))
    return model, seen, mask, weight * mask.float()[..., None]


def _run_layer(monkeypatch, attn, args, kwargs, dtype, flex, weight):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1" if flex else "0")
    attn.zero_grad(set_to_none = True)
    hidden = _to(args[0] if args else kwargs["hidden_states"], dtype).detach().requires_grad_(True)
    kw = {k: _to(v, dtype) for k, v in kwargs.items() if k != "hidden_states"}
    kw["past_key_values"] = None  # training: no cache
    out = attn(hidden, **kw)[0]
    (out.float() * weight).sum().backward()
    grads = {n: p.grad.float() for n, p in attn.named_parameters() if p.grad is not None}
    grads["hidden"] = hidden.grad.float()
    return out.detach().float(), grads


@needs_flex_gpu
def test_layers_equal_eager_in_float32(monkeypatch):
    """Sliding, CSA and HCA layers at the real head count / head dim on a left-padded batch: with the
    dtype gate opened to fp32, flex and eager agree on the output and every gradient up to reduction
    order, so the block mask, compressed entries and sink are the same math."""
    model, seen, _, weight = _captured_layers(monkeypatch)
    real_applies = F4._flex_applies
    monkeypatch.setattr(F4, "_flex_applies", lambda module, query, *a: query.is_cuda and real_applies(
        module, query.to(torch.bfloat16), *a))
    for idx, layer in enumerate(model.model.layers):
        args, kwargs = seen[idx]
        F4.DEEPSEEK_V4_FLEX_STATS.update(flex = 0, eager = 0)
        o_eager, g_eager = _run_layer(monkeypatch, layer.self_attn, args, kwargs, torch.float32, False, weight)
        o_flex, g_flex = _run_layer(monkeypatch, layer.self_attn, args, kwargs, torch.float32, True, weight)
        # fp32 at head_dim 512: flex's backward needs ~200 KB of shared memory (A100: 163 KB). The backward probe
        # latches eager there instead of crashing loss.backward(); the parity check needs a card it fits (H100 / B200).
        oor = [e for e in F4._FAILED.values() if "out of resource" in e]
        if oor:
            assert F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 0, "eager": 2}
            pytest.skip(f"fp32 flex backward does not fit this card's shared memory: {oor[0][:300]}")
        assert F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 1, "eager": 1}, layer.self_attn.layer_type
        assert _rel(o_flex, o_eager) < 1e-5
        for name in g_eager:
            assert _rel(g_flex[name], g_eager[name]) < 1e-5, (layer.self_attn.layer_type, name)


@needs_flex_gpu
def test_layers_in_bf16_stay_close_to_an_fp32_oracle(monkeypatch):
    """The bf16 flex layer is about as far from an fp32 eager oracle as transformers' own bf16 eager
    layer (a wrong mask or sink would be off by O(1))."""
    model, seen, mask, weight = _captured_layers(monkeypatch)
    keep = mask.bool()
    for idx, layer in enumerate(model.model.layers):
        args, kwargs = seen[idx]
        attn32 = layer.self_attn
        attn16 = copy.deepcopy(attn32).to(torch.bfloat16)
        o_ref, g_ref = _run_layer(monkeypatch, attn32, args, kwargs, torch.float32, False, weight)
        F4.DEEPSEEK_V4_FLEX_STATS.update(flex = 0, eager = 0)
        o_eager, g_eager = _run_layer(monkeypatch, attn16, args, kwargs, torch.bfloat16, False, weight)
        o_flex, g_flex = _run_layer(monkeypatch, attn16, args, kwargs, torch.bfloat16, True, weight)
        assert F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 1, "eager": 1}, attn32.layer_type
        assert _rel(o_flex[keep], o_ref[keep]) <= 2 * _rel(o_eager[keep], o_ref[keep]) + 2e-3
        for name in g_ref:
            assert _rel(g_flex[name], g_ref[name]) <= 2 * _rel(g_eager[name], g_ref[name]) + 2e-3, (attn32.layer_type, name)


@needs_flex_gpu
def test_prefill_on_flex_then_decode_through_the_cache(monkeypatch):
    """Prefill with a cache engages flex; the following decode steps (q_len 1) stay eager and see the
    same cache, so logits track the all-eager run."""
    model = _tiny_model().to(torch.bfloat16)
    ids = torch.randint(0, 512, (1, 600), device = "cuda")

    def run(flex):
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1" if flex else "0")
        with torch.no_grad():
            out = model(input_ids = ids, use_cache = True)
            logits = [out.logits[:, -1].float()]
            cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
            for _ in range(5):
                out = model(input_ids = token, past_key_values = cache, use_cache = True)
                logits.append(out.logits[:, -1].float())
                cache = out.past_key_values
                token = torch.randint(0, 512, (1, 1), device = "cuda", generator = torch.Generator("cuda").manual_seed(len(logits)))
        return torch.stack(logits)

    eager = run(False)
    F4.DEEPSEEK_V4_FLEX_STATS.update(flex = 0, eager = 0)
    flex = run(True)
    assert F4.DEEPSEEK_V4_FLEX_STATS["flex"] == 4  # one prefill call per layer
    assert F4.DEEPSEEK_V4_FLEX_STATS["eager"] == 4 * 5  # every decode step stays eager
    assert _rel(flex, eager) < 0.02


def test_older_torch_keeps_eager(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.9.1+cu128")
    assert not F4._torch_tested()
    monkeypatch.setattr(torch, "__version__", "2.10.0+cu128")
    assert F4._torch_tested()


@needs_flex_gpu
@pytest.mark.parametrize("rate", [0, 4, 128])
def test_flex_graphs_stay_bounded_over_lengths(rate):
    """Sliding (rate 0), CSA (4), HCA (128) masks over varying lengths (incl. non multiples of 128) and
    batch 1 / 2, fwd + bwd: at most four graphs (the first length static, then dynamic, batch 1 vs > 1), none
    added per new length after that."""
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    def call(S, B):
        comp = S // rate if rate else 0
        q = torch.randn(B, 8, S, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        kv = torch.randn(B, 1, S + comp, 128, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        mask = _mask(B, S, 128, comp, max(rate, 1), 16, torch.bfloat16, "cuda")
        F4.deepseek_v4_flex_attention(_Attn(8, "cuda").to(torch.bfloat16), q, kv, mask, 128 ** -0.5).float().sum().backward()

    for B in (1, 2):
        for S in (300, 384, 513, 777):
            call(S, B)
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    for S in (260, 640, 1000, 1025):
        call(S, 2)
        call(S, 1)
    assert graphs <= 4, graphs
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == graphs


@needs_flex_gpu
def test_packed_sequences_match_upstream(monkeypatch):
    """Padding-free packing (position_ids reset per document, no attention_mask). The compressor windows and
    the compressed-entry causality follow absolute slots, so upstream itself mixes documents: flex must
    match upstream's packed result (the additive mask is read as is). Then each document alone, fwd + bwd."""
    model = _tiny_model().to(torch.bfloat16)
    docs = [300, 100, 250]
    ids = torch.randint(0, 512, (1, sum(docs)), device = "cuda", generator = torch.Generator("cuda").manual_seed(0))
    pos = torch.cat([torch.arange(n, device = "cuda") for n in docs])[None]
    starts = [0, 300, 400, 650]

    def run(flex, packed):
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_FLEX", "1" if flex else "0")
        model.zero_grad(set_to_none = True)
        pieces = [(0, 650)] if packed else list(zip(starts[:-1], starts[1:]))
        logits = []
        for a, b in pieces:
            out = model(input_ids = ids[:, a:b], position_ids = pos[:, a:b] if packed else None, use_cache = False).logits
            out.float().square().mean().backward()
            logits.append(out.detach().float())
        return torch.cat(logits, 1)

    F4.DEEPSEEK_V4_FLEX_STATS.update(flex = 0, eager = 0)
    flex = run(True, True)
    assert F4.DEEPSEEK_V4_FLEX_STATS["flex"] == 4
    eager = run(False, True)
    assert _rel(flex, eager) < 0.02, _rel(flex, eager)
    flex_docs = run(True, False)
    assert F4.DEEPSEEK_V4_FLEX_STATS["flex"] == 4 + 12
    eager_docs = run(False, False)
    assert _rel(flex_docs, eager_docs) < 0.02, _rel(flex_docs, eager_docs)


_SHORT_DOC_WORKER = """
import sys, torch
sys.path.insert(0, sys.argv[1])
import unsloth_zoo.temporary_patches
import test_deepseek_v4_sparse_flex as T
T.F4._MIN_KEYS = {8: {None: 16}, 9: {None: 16}, 10: {None: 16}}
model = T._tiny_model().to(torch.bfloat16)
ids = torch.randint(0, 512, (1, 650), device = "cuda", generator = torch.Generator("cuda").manual_seed(0))
pos = torch.cat([torch.arange(n, device = "cuda") for n in (300, 100, 250)])[None]
model(input_ids = ids, position_ids = pos, use_cache = False).logits.float().square().mean().backward()
for a, b in ((0, 300), (300, 400), (400, 650)):
    model(input_ids = ids[:, a:b], use_cache = False).logits.float().square().mean().backward()
assert T.F4.DEEPSEEK_V4_FLEX_STATS["flex"] == 16, T.F4.DEEPSEEK_V4_FLEX_STATS
print("SHORT_DOC_OK")
"""


@needs_flex_gpu
def test_short_document_after_longer_ones():
    """A 100 token document (under one 128 tile) after longer ones, fwd + bwd, fresh process: torch 2.10 sent
    it to the flex decoding kernel once the compiled flex had gone dynamic, and that failed to lower
    (S <= BLOCK_M on the 32-wide tiles)."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_SPARSE_")}
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run([sys.executable, "-c", _SHORT_DOC_WORKER, here], capture_output = True, text = True, env = env, timeout = 600)
    assert "SHORT_DOC_OK" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-3000:]


@needs_flex_gpu
def test_device_probe_traces_without_graph_break():
    """The first flex check can run inside a compiled graph (empty device cache): the probe must trace
    whole. importlib's find_spec is skipped by dynamo and broke the graph on torch 2.10."""
    import torch._dynamo
    torch._dynamo.reset()
    F4._DEVICE_OK.clear()
    attn = _Attn(4, "cuda")
    device = torch.device("cuda", torch.cuda.current_device())

    def probe(x):
        return x + F4._min_keys(attn, device, True)

    assert torch.compile(probe, fullgraph = True)(torch.zeros(1, device = "cuda")).item() == 16


def _capture(fn):
    """fn() warmed up on a side stream, captured into a CUDA graph and replayed: (replayed output, eager output)."""
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
    return out, eager


@needs_flex_gpu
@pytest.mark.parametrize("finite_bias", [False, True])
def test_cuda_graph_capture_matches_eager(finite_bias):
    """The finite-bias check is a host sync; under stream capture it invalidated the capture
    (cudaErrorStreamCaptureInvalidated). Capturing keeps the bias instead: the replay matches the eager call."""
    torch.manual_seed(0)
    attn = _Attn(4, "cuda").to(torch.bfloat16)
    q = torch.randn(1, 4, 256, 64, device = "cuda", dtype = torch.bfloat16)
    kv = torch.randn(1, 1, 256 + 64, 64, device = "cuda", dtype = torch.bfloat16)
    mask = _mask(1, 256, 128, 64, 4, 16, torch.bfloat16, "cuda")
    if finite_bias:
        mask = mask + torch.rand_like(mask).mul(0.5)
    with torch.no_grad():
        out, eager = _capture(lambda: modeling.eager_attention_forward(attn, q, kv, kv, mask, 0.125)[0])
    assert F4.DEEPSEEK_V4_FLEX_STATS["flex"] == 4 and F4.DEEPSEEK_V4_FLEX_STATS["eager"] == 0
    assert _rel(out, eager) < 1e-2, _rel(out, eager)


@needs_flex_gpu
def test_cold_fullgraph_compile_is_one_graph(monkeypatch):
    """Right after install, a fullgraph dynamic compile of the dispatcher is one graph: the compiled flex
    callable used to be set lazily inside the trace, and its `_FLEX is None` guard recompiled the next call."""
    import torch._dynamo
    monkeypatch.setattr(modeling, "eager_attention_forward", ORIGINAL)
    monkeypatch.setattr(F4, "_FLEX", None)
    F4.patch_deepseek_v4_sparse_flex()
    attn = _Attn(4, "cuda").to(torch.bfloat16)
    q = torch.randn(1, 4, 256, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    kv = torch.randn(1, 1, 320, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    mask = _mask(1, 256, 128, 64, 4, 16, torch.bfloat16, "cuda")
    forward = modeling.eager_attention_forward

    def fn(q, kv, mask):
        return forward(attn, q, kv, kv, mask, 0.125)[0]

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(fn, fullgraph = True, dynamic = True)
    for _ in range(2):
        compiled(q, kv, mask).float().sum().backward()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1, dict(torch._dynamo.utils.counters["stats"])


class _BackwardFails(torch.autograd.Function):
    """Stands in for a flex kernel whose forward runs and whose backward fails (shared-memory OutOfResources)."""

    @staticmethod
    def forward(ctx, query):
        return query.transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("out of resource: shared memory, Required: 262144, Hardware limit: 101376")


def test_flex_backward_failure_latches_eager(monkeypatch):
    """flex's backward runs in loss.backward(): the first training call of a shape class probes it, so a failing
    backward latches eager for this call on instead of ending training."""
    calls = []

    def flex(module, query, key, attention_mask, scaling):
        calls.append(1)
        return _BackwardFails.apply(query)

    # Every gate open on CPU except the latch.
    monkeypatch.setattr(F4, "_flex_applies", lambda *args: not getattr(F4, "_FAILED", {}))
    monkeypatch.setattr(F4, "deepseek_v4_flex_attention", flex)
    attn, q, kv, mask = _gate_inputs("cpu", dtype = torch.float32)
    q.requires_grad_()
    out, _ = modeling.eager_attention_forward(attn, q, kv, kv, mask, 0.125)
    out.sum().backward()  # eager's backward
    ref, _ = ORIGINAL(attn, q.detach(), kv, kv, mask, 0.125)
    torch.testing.assert_close(out.detach(), ref)
    assert len(F4._FAILED) == 1 and F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 0, "eager": 1}
    modeling.eager_attention_forward(attn, q, kv, kv, mask, 0.125)
    assert len(calls) == 1  # latched: not tried again (the failing probe was the only flex call)


def test_flex_forward_failure_latches_eager(monkeypatch):
    def flex(module, query, key, attention_mask, scaling):
        raise RuntimeError("triton compile error")

    monkeypatch.setattr(F4, "_flex_applies", lambda *args: True)
    monkeypatch.setattr(F4, "deepseek_v4_flex_attention", flex)
    attn, q, kv, mask = _gate_inputs("cpu", dtype = torch.float32)
    with torch.no_grad():
        out, _ = modeling.eager_attention_forward(attn, q, kv, kv, mask, 0.125)
    torch.testing.assert_close(out, ORIGINAL(attn, q, kv, kv, mask, 0.125)[0])
    assert len(F4._FAILED) == 1 and F4.DEEPSEEK_V4_FLEX_STATS == {"flex": 0, "eager": 1}
