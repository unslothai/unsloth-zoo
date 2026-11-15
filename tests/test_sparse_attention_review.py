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

"""Regression tests for the sparse attention review findings (unsloth-zoo #2509): dispatch latches, input
validation, the no-sink form, empty / -inf rows, layouts, the Qwen4Exp patch guards. CUDA tests skip on CPU."""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo import sparse_attention as sa  # noqa: E402

_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not _CUDA, reason = "CUDA kernels")


def _cpu_inputs(b = 1, s = 9, h = 2, d = 16, n = 11, k = 4, seed = 0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, s, h, d, generator = g)
    kv = torch.randn(b, n, d, generator = g)
    sink = torch.randn(h, generator = g)
    idx = torch.randint(-1, n, (b, s, k), generator = g).int()
    return q, kv, sink, idx


@pytest.fixture
def clean_dispatch(monkeypatch):
    monkeypatch.setattr(sa, "_FAILED", {})
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_BACKEND", raising = False)


# 1: an automatic window gather that failed once is latched (not retried every call).
def test_window_gather_latch_is_consulted(monkeypatch, clean_dispatch):
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("refused launch")

    monkeypatch.setattr(sa, "_gather_eligible", lambda q, window = 0: True)
    monkeypatch.setattr(sa, "_sparse_attn_window_gather", failing)
    q, kv, sink, idx = _cpu_inputs()
    win = torch.randn(1, 9, 16)
    for _ in range(2):
        sa.sparse_attn_window(q, win, kv, idx, sink, 0.25, 4)
    assert len(calls) == 1


# 25: a forced window gather that fails raises; it is not latched and rerun through the indexed path.
def test_forced_window_gather_failure_raises(monkeypatch, clean_dispatch):
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(sa, "_sparse_attn_window_gather", failing)
    q, kv, sink, idx = _cpu_inputs()
    before = sa.SPARSE_ATTN_STATS["fallback"]
    with pytest.raises(RuntimeError, match = "boom"):
        sa.sparse_attn_window(q, torch.randn(1, 9, 16), kv, idx, sink, 0.25, 4, backend = "gather")
    assert len(calls) == 1 and sa.SPARSE_ATTN_STATS["fallback"] == before


# 2: an inference-mode mask has no version counter; the geometry scan must not read it.
def test_geometry_of_an_inference_mask():
    with torch.inference_mode():
        mask = torch.tril(torch.ones(12, 12, dtype = torch.bool)).view(1, 1, 12, 12)
        result = sa._qsa_geometry(mask, 4)
    assert result is not None


# 3: without Triton nothing imports the Triton module; the CPU paths still run.
def test_no_triton_never_imports_the_kernel_module(monkeypatch, clean_dispatch):
    monkeypatch.setitem(sys.modules, "unsloth_zoo.sparse_mqa_kernel", None)   # any import raises ImportError
    monkeypatch.setattr(sa, "_HAS_TRITON", False)
    q, kv, sink, idx = _cpu_inputs()
    assert sa._triton_eligible(q) is False
    out = sa.sparse_attn(q, kv, sink, idx, 0.25)
    assert out.shape == q.shape
    query, key = torch.randn(1, 4, 9, 8), torch.randn(1, 2, 9, 8)
    assert sa.sparse_gqa_fast_path(query, key, key) is False
    assert sa.sparse_gqa_attn(query, key, key, idx[..., :3], 0.3, backend = "torch").shape == (1, 9, 4, 8)


# 6: the window entry validates its operands like sparse_attn.
@pytest.mark.parametrize("bad", ["window_width", "compressed_width", "short_sink", "index_rows"])
def test_window_entry_validates_operands(bad):
    q, kv, sink, idx = _cpu_inputs()
    win = torch.randn(1, 9, 16)
    if bad == "window_width":
        win = torch.randn(1, 9, 8)
    elif bad == "compressed_width":
        kv = torch.randn(1, 11, 8)
    elif bad == "short_sink":
        sink = sink[:1]
    else:
        idx = idx[:, :5]
    with pytest.raises(ValueError):
        sa.sparse_attn_window(q, win, kv, idx, sink, 0.25, 4, backend = "torch")


# 35: windows must be positive.
@pytest.mark.parametrize("window", [0, -3])
def test_window_must_be_positive(window):
    q, kv, sink, idx = _cpu_inputs()
    with pytest.raises(ValueError):
        sa.sparse_attn_window(q, torch.randn(1, 9, 16), kv, idx, sink, 0.25, window, backend = "torch")


def _capture_gather(monkeypatch):
    seen = {}

    def fake(query, local_kv, sparse_kv, idxs, sink, **kwargs):
        seen.update(local = None if local_kv is None else tuple(local_kv.shape), sparse = tuple(sparse_kv.shape),
                    sink = sink.clone())
        return query.new_zeros(query.shape)

    monkeypatch.setattr(sa, "_GATHER", fake)
    return seen


# 7: gather receives one KV head ([b, 1, n, d]) so it records share_kv and keeps the shared backward.
def test_gather_gets_a_single_shared_kv_head(monkeypatch):
    seen = _capture_gather(monkeypatch)
    q, kv, sink, idx = _cpu_inputs(h = 4)
    out = sa._gather_call(q, torch.randn(1, 9, 16), kv, idx, sink, 0.25, 4)
    assert seen["sparse"] == (1, 1, 11, 16) and seen["local"] == (1, 1, 9, 16)
    assert out.is_contiguous()   # 58


# 15-18: the no-sink form (attn_sink None) through _check, the reference (incl. recompute) and the gather adapter.
def test_no_sink_form(monkeypatch, clean_dispatch):
    q, kv, _, idx = _cpu_inputs()
    no_sink = torch.full((2,), float("-inf"))
    want = sa.sparse_attn_reference(q.double(), kv.double(), no_sink.double(), idx, 0.25)
    torch.testing.assert_close(sa.sparse_attn(q, kv, None, idx, 0.25).double(), want, rtol = 1e-5, atol = 1e-5)
    monkeypatch.setattr(sa, "_TORCH_CHUNK_ELEMS", 16)   # forces the checkpointed chunks (17: recompute gate)
    qg = q.clone().requires_grad_()
    out = sa.sparse_attn_reference(qg, kv, None, idx, 0.25)
    out.sum().backward()
    assert torch.isfinite(qg.grad).all()
    seen = _capture_gather(monkeypatch)
    sa._gather_call(q, None, kv, idx, None, 0.25, 0)
    assert torch.isneginf(seen["sink"]).all()


# 33 / 34: an empty selection stays connected to every input (zero gradients, not an error).
@pytest.mark.parametrize("fn", ["public", "reference"])
def test_empty_selection_is_connected_to_every_input(fn, clean_dispatch):
    q, kv, sink, _ = _cpu_inputs()
    kv, sink = kv.requires_grad_(), torch.full((2,), float("-inf")).requires_grad_()
    idx = torch.empty(1, 9, 0, dtype = torch.int32)
    call = sa.sparse_attn if fn == "public" else sa.sparse_attn_reference
    out = call(q, kv, sink, idx, 0.25)
    assert out.requires_grad and not out.any()
    d_kv, d_sink = torch.autograd.grad(out.sum(), (kv, sink))
    assert not d_kv.any() and not d_sink.isnan().any()


# 10 / 48: GQA validates the index rows and the head grouping before folding.
def test_gqa_validation(clean_dispatch):
    query, key = torch.randn(1, 4, 9, 8), torch.randn(1, 2, 9, 8)
    with pytest.raises(ValueError):
        sa.sparse_gqa_attn(query, key, key, torch.zeros(2, 9, 3, dtype = torch.int32), 0.3, backend = "torch")
    with pytest.raises(ValueError):
        sa.sparse_gqa_attn(torch.randn(1, 3, 9, 8), key, key, torch.zeros(1, 9, 3, dtype = torch.int32), 0.3,
                           backend = "torch")


# 11 / 12: unsupported mask types and ranks decline instead of crashing; per-head masks too.
@pytest.mark.parametrize("mask", ["none", "2d", "3d", "per_head"])
def test_geometry_and_selector_decline_unsupported_masks(mask):
    s = 12
    causal = torch.tril(torch.ones(s, s, dtype = torch.bool))
    m = {"none": None, "2d": causal, "3d": causal[None], "per_head": causal.expand(1, 2, s, s)}[mask]
    assert sa._qsa_geometry(m, 4) is None
    indexer = types.SimpleNamespace(compress_ratio = 4, block_topk = 2, index_head_dim = 8)
    assert sa.qsa_select_tokens(indexer, torch.randn(1, s, 16), (None, None), m, None) is None


# 36: every key inside a row's interval must share the row's first key, not only the endpoints.
def test_interval_rows_check_the_interior():
    s = 8
    visible = torch.tril(torch.ones(s, s, dtype = torch.bool))
    visible[3] = False
    visible[3, 3] = True             # row 3 sees only itself: key 3 starts a different "document"
    assert sa._contiguous_rows(visible[None]) is None
    assert sa._contiguous_rows(torch.tril(torch.ones(1, s, s, dtype = torch.bool))) is not None


# 20: the shared-memory helper answers 0 off CUDA.
def test_shared_memory_off_cuda():
    assert sa._shared_memory(torch.device("cpu")) == 0


# 26: the environment can force hd512 for the window form only.
def test_env_backend_parser_accepts_hd512_for_windows(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "hd512")
    assert sa._forced_backend(window = True) == "hd512"
    assert sa._forced_backend() is None


# 30: the fast-path probe follows the execution policy (no flex inside traced regions).
def test_fast_path_probe_agrees_with_compiled_execution(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_BACKEND", raising = False)
    monkeypatch.setattr(sa, "_gather_eligible", lambda q, window = 0: False)
    monkeypatch.setattr(sa, "_triton_eligible", lambda q, forced = False, kv = None: False)
    monkeypatch.setattr(sa, "_flex_eligible", lambda q, kv_len = None: True)
    query, key = torch.randn(1, 4, 9, 8), torch.randn(1, 2, 9, 8)
    assert sa.sparse_gqa_fast_path(query, key, key)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not sa.sparse_gqa_fast_path(query, key, key)


# 46: a differentiable finite sink never takes flex on a torch release where flex's LSE gradient is unverified.
def test_flex_with_sink_gated_below_the_verified_torch(monkeypatch, clean_dispatch):
    monkeypatch.setattr(sa, "_torch_version", lambda: (2, 6))
    monkeypatch.setattr(sa, "_order", lambda q, k, backend = None, kv_len = None: ("flex", "torch"))
    monkeypatch.setattr(sa, "_sparse_attn_flex", lambda *a: (_ for _ in ()).throw(AssertionError("flex ran")))
    q, kv, sink, idx = _cpu_inputs()
    before = sa.SPARSE_ATTN_STATS["torch"]
    sa.sparse_attn(q, kv, sink.requires_grad_(), idx, 0.25)
    assert sa.SPARSE_ATTN_STATS["torch"] == before + 1
    monkeypatch.setattr(sa, "_torch_version", lambda: (2, 14))
    assert sa._flex_sink_ok(q, kv, sink)


# 32: an untimed tuner selection launches once on a cold key (the selection's launch is the call's launch).
def test_tuner_launches_once_on_a_cold_untimed_call(monkeypatch):
    pytest.importorskip("triton")
    tune = pytest.importorskip("unsloth_zoo._vendored.attn_gym._backends.triton.tune")
    import triton
    monkeypatch.setenv("UNSLOTH_ATTN_GYM_AUTOTUNE", "0")
    launches = []

    class Kernel:
        arg_names = ["x"]

        def __getitem__(self, grid):
            return lambda **kwargs: launches.append(kwargs["x"])

    tuner = tune.TritonTuner(Kernel(), [triton.Config({"BLOCK": 16}), triton.Config({"BLOCK": 32})], key = lambda a: (1,))
    tuner[(1,)](7)
    assert launches == [7]
    tuner[(1,)](8)
    assert launches == [7, 8]


# 45: the large-head loader also searches namespace / editable package locations (spec.origin is None).
def test_large_head_loader_searches_package_locations(monkeypatch, tmp_path):
    pytest.importorskip("triton")
    from unsloth_zoo import sparse_mqa_kernel as mk
    kernels = tmp_path / "unsloth" / "kernels"
    kernels.mkdir(parents = True)
    (kernels / "large_head_attention.py").write_text("def _fwd_impl(*a): pass\ndef _bwd_impl(*a): pass\n")
    spec = types.SimpleNamespace(origin = None, submodule_search_locations = [str(tmp_path / "unsloth")])
    real_find = mk.importlib.util.find_spec
    monkeypatch.setattr(mk.importlib.util, "find_spec", lambda name, *a: spec if name == "unsloth" else real_find(name, *a))
    monkeypatch.setattr(mk, "_LH", None)
    monkeypatch.delitem(sys.modules, mk._LH_NAME, raising = False)
    module = mk._large_head_module()
    assert module is not None and module.__file__ == str(kernels / "large_head_attention.py")
    sys.modules.pop(mk._LH_NAME, None)


# ----------------------------------------------------------------------------------------------- Qwen4Exp patch
modeling = None
try:
    import transformers.models.qwen4_exp.modeling_qwen4_exp as modeling  # noqa: F811
except Exception:
    pass
qwen_only = pytest.mark.skipif(modeling is None, reason = "transformers without qwen4_exp")


def _qwen(dtype = torch.float64, seed = 0, attn = "eager"):
    from unsloth_zoo.temporary_patches import qwen4_exp as zq
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    config = Qwen4ExpTextConfig(
        hidden_size = 64, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
        indexer_budget = 16, indexer_compress_ratio = 4, indexer_head_dim = 16, indexer_n_heads = 2,
        indexer_kv_heads = 1, partial_rotary_factor = 0.25, num_hidden_layers = 4, vocab_size = 128,
        layer_types = ["linear_attention"] * 3 + ["full_attention"],
    )
    config._attn_implementation = attn
    torch.manual_seed(seed)
    attention = modeling.Qwen4ExpTextAttention(config, layer_idx = 3).to(dtype)
    rotary = modeling.Qwen4ExpTextRotaryEmbedding(config)
    return zq, config, attention, rotary


def _qwen_inputs(rotary, s, dtype = torch.float64, pe_dtype = torch.float64):
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(1, s, 64, generator = g).to(dtype)
    cos, sin = rotary(hidden.float(), torch.arange(s)[None])
    mask = torch.tril(torch.ones(s, s, dtype = torch.bool)).view(1, 1, s, s)
    return hidden, (cos.to(pe_dtype), sin.to(pe_dtype)), mask


# 13: the configured output_attentions default keeps the reference (which returns the weights).
@qwen_only
def test_config_output_attentions_keeps_the_reference(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    zq, config, attention, rotary = _qwen()
    hidden, pe, mask = _qwen_inputs(rotary, 40)
    assert zq._qwen4_exp_sparse_attention(attention, hidden, pe, mask, None, {}) is not None
    config.output_attentions = True
    assert zq._qwen4_exp_sparse_attention(attention, hidden, pe, mask, None, {}) is None


# 14: a per-head mask keeps the reference (one shared selection cannot carry per-head restrictions).
@qwen_only
def test_per_head_mask_keeps_the_reference(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    zq, config, attention, rotary = _qwen()
    hidden, pe, mask = _qwen_inputs(rotary, 40)
    per_head = mask.expand(1, 4, 40, 40).clone()
    per_head[:, 1, :, :8] = False
    assert zq._qwen4_exp_sparse_attention(attention, hidden, pe, per_head, None, {}) is None


# 39: the fast-path probe sees the real [b, h, s, d] shapes (flex's table budget is b * s * s).
@qwen_only
def test_fast_path_probe_shapes(monkeypatch):
    zq, config, attention, rotary = _qwen()
    hidden, pe, mask = _qwen_inputs(rotary, 40)
    seen = []
    monkeypatch.setattr(sa, "sparse_gqa_fast_path", lambda q, k, v: seen.append((tuple(q.shape), tuple(k.shape))) or False)
    assert zq._qwen4_exp_sparse_attention(attention, hidden, pe, mask, None, {}) is None
    assert seen == [((1, 4, 40, 32), (1, 2, 40, 32))]


# 47: an attention subclass defined in another module still finds transformers' RoPE.
@qwen_only
def test_subclass_finds_rope(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    zq, config, attention, rotary = _qwen()
    subclass = type("MyAttention", (modeling.Qwen4ExpTextAttention,), {"__module__": __name__})
    attention.__class__ = subclass
    assert not hasattr(sys.modules[__name__], "apply_rotary_pos_emb")
    hidden, pe, mask = _qwen_inputs(rotary, 40)
    assert zq._qwen4_exp_sparse_attention(attention, hidden, pe, mask, None, {}) is not None


# 23 / 24: bf16 projections with fp32 RoPE tables (4-bit / mixed precision): both the sparse path and the reference
# fallback (short sequences) align Q / K / V dtypes after RoPE, so the fused kernels / SDPA see one dtype; a compiled
# step runs. On CUDA, SDPA rejects mixed dtypes and the sparse kernels need bf16 Q / K, so the base fails there.
@qwen_only
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("seq", [12, 40], ids = ["reference_fallback", "sparse"])
def test_post_rope_dtype_alignment(monkeypatch, seq, device):
    if device == "cuda" and not (_CUDA and torch.cuda.get_device_capability()[0] >= 8):
        pytest.skip("CUDA bf16")
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_BACKEND", raising = False)
    if device == "cpu":
        monkeypatch.setenv("UNSLOTH_SPARSE_ATTN_BACKEND", "torch")
    monkeypatch.setattr(sa, "_FAILED", {})
    # eager: transformers' P.V matmul rejects fp32 probabilities against bf16 V (SDPA's math path upcasts).
    zq, config, attention, rotary = _qwen(torch.bfloat16, attn = "eager" if seq == 12 else "sdpa")
    zq.patch_qwen4_exp()
    assert modeling.Qwen4ExpTextAttention.forward is zq.qwen4_exp_attention_forward

    class InputCast(torch.nn.Module):   # bitsandbytes Linear4bit takes any input dtype; nn.Linear does not
        def __init__(self, linear):
            super().__init__()
            self.linear = linear

        def forward(self, x):
            return self.linear(x.to(self.linear.weight.dtype))

    attention.o_proj = InputCast(attention.o_proj)
    attention = attention.to(device)
    hidden, pe, mask = _qwen_inputs(rotary, seq, torch.bfloat16, torch.float32)
    hidden, pe, mask = hidden.to(device), tuple(x.to(device) for x in pe), mask.to(device)
    before = dict(sa.SPARSE_ATTN_STATS)
    out = attention(hidden, pe, mask)[0]
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    if device == "cuda" and seq == 40:   # the sparse path ran on a kernel, not the fp32 torch fallback
        ran = {k: sa.SPARSE_ATTN_STATS[k] - before[k] for k in before if sa.SPARSE_ATTN_STATS[k] != before[k]}
        assert set(ran) <= {"flex", "triton", "gather"} and ran, ran
    compiled = torch.compile(lambda h: attention(h, pe, mask)[0], backend = "aot_eager")
    torch.testing.assert_close(compiled(hidden).float(), out.float(), rtol = 2e-2, atol = 2e-2)


# ------------------------------------------------------------------------------------------------- CUDA kernels
# 4 / 5: a strided sink view gives the same result as its contiguous copy.
@cuda_only
@pytest.mark.parametrize("form", ["index", "window"])
def test_strided_sink(form):
    from unsloth_zoo import sparse_mqa_kernel as mk
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, 20, 4, 64, generator = g).cuda().to(dtype)
    kv = torch.randn(1, 30, 64, generator = g).cuda().to(dtype)
    idx = torch.randint(-1, 30, (1, 20, 8), generator = g).int().cuda()
    strided = torch.randn(8, generator = g).cuda()[::2]
    assert not strided.is_contiguous()
    if form == "index":
        got, want = (mk.sparse_mqa_attention(q, kv, sk, idx, 0.125) for sk in (strided, strided.contiguous()))
    else:
        win = torch.randn(1, 20, 64, generator = g).cuda().to(dtype)
        got, want = (mk.sparse_mqa_window_attention(q, win, kv, idx, sk, 0.125, 5) for sk in (strided, strided.contiguous()))
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)


# 19: the native support predicate checks the KV geometry.
@cuda_only
def test_native_support_checks_kv_geometry():
    from unsloth_zoo import sparse_mqa_kernel as mk
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    q = torch.empty(1, 4, 2, 64, device = "cuda", dtype = dtype)
    assert mk.sparse_mqa_supported(q, torch.empty(1, 7, 64, device = "cuda", dtype = dtype))
    for kv in (torch.empty(1, 7, 32, device = "cuda", dtype = dtype), torch.empty(2, 7, 64, device = "cuda", dtype = dtype),
               torch.empty(1, 1, 7, 64, device = "cuda", dtype = dtype)):
        assert not mk.sparse_mqa_supported(q, kv)


# 10 (kernel path): a malformed index batch never reaches the GQA Triton fold.
@cuda_only
def test_gqa_index_batch_rejected_before_the_kernel(clean_dispatch):
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    query = torch.randn(1, 4, 9, 64, device = "cuda", dtype = dtype)
    key = torch.randn(1, 2, 9, 64, device = "cuda", dtype = dtype)
    with pytest.raises(ValueError):
        sa.sparse_gqa_attn(query, key, key, torch.zeros(2, 9, 3, dtype = torch.int32, device = "cuda"), 0.1,
                           backend = "triton")


# 9: flex sink scaling on empty rows and disabled (-inf) sink heads gives finite gradients.
@cuda_only
def test_flex_sink_scaling_empty_rows_finite(clean_dispatch):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("flex is sm80+")
    q = torch.randn(1, 130, 4, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    kv = torch.randn(1, 140, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    sink = torch.tensor([0.5, float("-inf"), 1.0, float("-inf")], device = "cuda", requires_grad = True)
    idx = torch.randint(0, 140, (1, 130, 16), device = "cuda").int()
    idx[0, :5] = -1                      # empty rows
    out = sa.sparse_attn(q, kv, sink, idx, 0.125, backend = "flex")
    grads = torch.autograd.grad(out.float().sum(), (q, kv, sink))
    assert torch.isfinite(out).all() and all(torch.isfinite(g).all() for g in grads)
    assert not out[0, :5].any()


# 21 / 22 / 27 / 28: hd512 ops on empty inputs, and on empty intervals with a disabled sink (no NaN).
@cuda_only
def test_hd512_empty_inputs_and_neg_inf_rows():
    from unsloth_zoo import sparse_mqa_kernel as mk
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    q = torch.empty(1, 0, 16, 512, device = "cuda", dtype = dtype)
    kv = torch.empty(1, 0, 512, device = "cuda", dtype = dtype)
    if not mk.window_hd512_supported(torch.empty(1, 4, 16, 512, device = "cuda", dtype = dtype),
                                     torch.empty(1, 4, 512, device = "cuda", dtype = dtype)):
        pytest.skip("Unsloth's head_dim 512 kernel is not importable or does not fit here")
    sink = torch.zeros(16, device = "cuda")
    out, total, lse2 = mk._hfwd_op(q, kv, sink, 0.05, 4, True)
    assert out.shape == q.shape and total.shape == (1, 0, 16) and lse2.shape == (1, 16, 0)
    dq, dkv = mk._hbwd_op(q, q, kv, q, lse2, 0.05, 4)
    assert dq.shape == q.shape and dkv.shape == kv.shape
    # window 0: every interval is empty; with a -inf sink the share must be 0, not NaN.
    q = torch.randn(1, 20, 16, 512, device = "cuda", dtype = dtype, requires_grad = True)
    kv = torch.randn(1, 20, 512, device = "cuda", dtype = dtype, requires_grad = True)
    sink = torch.full((16,), float("-inf"), device = "cuda", requires_grad = True)
    out = mk._hfwd_op(q, kv, sink, 0.05, 0, True)[0]
    grads = torch.autograd.grad(out.float().sum(), (q, kv, sink))
    assert torch.isfinite(out).all() and not out.any()
    assert all(torch.isfinite(g).all() for g in grads)


_BIG_STRIDE = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from unsloth_zoo.sparse_mqa_kernel import sparse_mqa_attention
s, h, d = 4, 16, 128
stride_h = (2**31) // (h - 1) // d * d + d          # (h - 1) * stride_h > 2**31: overflows int32
storage = torch.empty((h - 1) * stride_h + s * d, device = "cuda", dtype = torch.bfloat16)
q = storage.as_strided((1, s, h, d), (0, d, stride_h, 1))
for head in range(h):
    q[0, :, head] = torch.randn(s, d, device = "cuda")
kv = torch.randn(1, 6, d, device = "cuda", dtype = torch.bfloat16)
idx = torch.arange(6, device = "cuda", dtype = torch.int32).expand(1, s, 6).contiguous()
got = sparse_mqa_attention(q, kv, None, idx, d ** -0.5)
want = sparse_mqa_attention(q.contiguous(), kv, None, idx, d ** -0.5)
torch.cuda.synchronize()
print("MAXDIFF", (got.float() - want.float()).abs().max().item())
"""


# 29: head offsets are 64-bit: a query view whose head stride times the head index passes 2**31.
@cuda_only
def test_head_stride_past_int32():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16")
    free = torch.cuda.mem_get_info()[0]
    if free < 6 * 2**30:
        pytest.skip("needs ~4.5 GB of device memory")
    root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, "-c", _BIG_STRIDE, root], capture_output = True, text = True, timeout = 600,
                          env = dict(os.environ, PYTHONPATH = root + os.pathsep + os.environ.get("PYTHONPATH", "")))
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("MAXDIFF")), None)
    assert line is not None, proc.stderr[-2000:]
    assert float(line.split()[1]) == 0.0, line
