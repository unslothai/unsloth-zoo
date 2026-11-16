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

"""CachedAutotuner compiles a cold key's configs on a thread pool before benchmarking them, keeps
the serial results (chosen config valid, outputs exact, failing configs skipped), and
UNSLOTH_FLA_PARALLEL_COMPILE=0 restores one-at-a-time compiles."""

import importlib.util
import pathlib
import threading

import pytest
from packaging import version

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if version.parse(triton.__version__.split("+")[0]) < version.parse("3.5"):
    pytest.skip("AsyncCompileMode needs triton>=3.5", allow_module_level = True)
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level = True)
tl = triton.language

_CACHE_PATH = pathlib.Path(__file__).resolve().parents[1] / "unsloth_zoo/_vendored/fla/ops/utils/cache.py"


@pytest.fixture
def cache_mod():
    spec = importlib.util.spec_from_file_location("fla_cache_parallel_under_test", _CACHE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scale_kernel(x, y, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    # BLOCK 4096 fails to compile: the autotuner must skip it, serial or parallel.
    tl.static_assert(BLOCK <= 1024)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y + offs, tl.load(x + offs, mask = mask) * SCALE, mask = mask)


def _make(cache_mod, monkeypatch, threads_seen, blocks = (128, 256, 512, 4096), warps = (2, 4)):
    """A fresh JITFunction per test (its binder picks up the recording compile on first launch)."""
    import triton.compiler as C

    jit_fn = triton.jit(_scale_kernel)
    original = C.compile

    def compile(src, *a, **k):
        if getattr(src, "fn", None) is jit_fn:
            threads_seen.append(threading.current_thread().name)
        return original(src, *a, **k)

    monkeypatch.setattr(C, "compile", compile)
    configs = [triton.Config({"BLOCK": b}, num_warps = w) for b in blocks for w in warps]
    return cache_mod.fla_cache_autotune(configs = configs, key = ["n"])(jit_fn)


def _launch(tuner, n, scale):
    x = torch.randn(n, device = "cuda")
    y = torch.empty_like(x)
    tuner[lambda meta: (triton.cdiv(n, meta["BLOCK"]),)](x, y, n, SCALE = scale)
    torch.cuda.synchronize()
    assert torch.equal(y, x * scale)
    return tuner.best_config


def test_cold_key_compiles_on_pool_and_matches_serial(cache_mod, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_mod, "_parallel_compile_threads", lambda: 4)
    seen = []
    tuner = _make(cache_mod, monkeypatch, seen)
    best = _launch(tuner, 3000, 2.0)
    pooled = [t for t in seen if t.startswith("fla_compile")]
    # All 8 configs (2 of them failing) compiled on the pool, none again on the calling thread.
    assert len(pooled) == 8, seen
    assert len(seen) == 8, seen
    assert best.kwargs["BLOCK"] <= 1024
    # Warm key: no compiles at all.
    seen.clear()
    _launch(tuner, 3000, 2.0)
    assert seen == []
    # A new key (another round right after the first) is compiled on the pool again.
    _launch(tuner, 4096, 2.0)  # % 16 == 0: a new specialization, so real compiles
    assert len([t for t in seen if t.startswith("fla_compile")]) == 8, seen


def test_kill_switch_compiles_serially(cache_mod, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_FLA_PARALLEL_COMPILE", "0")
    seen = []
    tuner = _make(cache_mod, monkeypatch, seen)
    best = _launch(tuner, 3000, 3.0)
    assert seen and not any(t.startswith("fla_compile") for t in seen), seen
    assert best.kwargs["BLOCK"] <= 1024


def test_single_config_round_never_uses_pool(cache_mod, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache_mod, "_parallel_compile_threads", lambda: 4)
    seen = []
    tuner = _make(cache_mod, monkeypatch, seen, blocks = (256,), warps = (4,))
    _launch(tuner, 1000, 1.5)
    assert seen and not any(t.startswith("fla_compile") for t in seen), seen


# --- configs pruned before compiling: num_stages twins before sm80, l2norm on head dim <= 128 ---


@triton.jit
def _pipelined_kernel(a, b, c, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype = tl.float32)
    for k in range(0, K, BK):
        x = tl.load(a + rm[:, None] * K + (k + rk)[None, :])
        y = tl.load(b + (k + rk)[:, None] * BN + rn[None, :])
        acc += tl.dot(x, y)
    tl.store(c + rm[:, None] * BN + rn[None, :], acc.to(tl.float16))


def _ptx(capability, num_stages):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(
        fn = _pipelined_kernel,
        signature = {"a": "*fp16", "b": "*fp16", "c": "*fp16", "K": "i32", "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
        constexprs = {"BM": 64, "BN": 64, "BK": 32},
        # 16-byte aligned pointers and K, as Triton specializes real launches: lets sm80 use cp.async.
        attrs = {(i,): [["tt.divisibility", 16]] for i in range(4)},
    )
    kernel = triton.compile(src, target = GPUTarget("cuda", capability, 32), options = {"num_warps": 4, "num_stages": num_stages})
    return kernel.asm["ptx"]


def test_num_stages_changes_nothing_before_sm80():
    # The premise of dropping num_stages twins: identical PTX on sm75, and the control (sm80, where
    # Triton pipelines the loop with cp.async) shows the comparison tells the kernels apart.
    assert _ptx(75, 2) == _ptx(75, 3) == _ptx(75, 4)
    assert len({_ptx(80, 2), _ptx(80, 3), _ptx(80, 4)}) == 3
    assert "cp.async" in _ptx(80, 3) and "cp.async" not in _ptx(75, 3)


def test_stage_twins_dropped_only_before_sm80(cache_mod, monkeypatch):
    import types

    configs = [triton.Config({"BK": bk}, num_warps = w, num_stages = s) for bk in (32, 64) for w in (2, 4) for s in (2, 3, 4)]
    x = torch.zeros(1, device = "cuda")
    tuner = types.SimpleNamespace(nargs = {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (7, 5))
    kept = cache_mod._drop_stage_only_twins(tuner, configs, {"x": x})
    assert [(c.kwargs["BK"], c.num_warps, c.num_stages) for c in kept] == [(32, 2, 2), (32, 4, 2), (64, 2, 2), (64, 4, 2)]
    monkeypatch.setenv("UNSLOTH_FLA_FULL_AUTOTUNE", "1")
    assert cache_mod._drop_stage_only_twins(tuner, configs, {"x": x}) == configs
    monkeypatch.delenv("UNSLOTH_FLA_FULL_AUTOTUNE")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (8, 9))
    assert cache_mod._drop_stage_only_twins(tuner, configs, {"x": x}) == configs
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (7, 5))
    assert cache_mod._drop_stage_only_twins(tuner, configs, {"x": x}) == configs


def test_l2norm_keeps_two_configs_up_to_head_dim_128(monkeypatch):
    import os
    import sys

    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")
    fla_vendor = pytest.importorskip("unsloth_zoo.temporary_patches.fla_vendor")
    if not fla_vendor._vendored_injection_supported():
        pytest.skip("vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3")
    injected, _ = fla_vendor._inject_vendored_fla()
    if not injected:
        pytest.skip("vendored fla injection failed on this host")
    L = sys.modules.get("fla.modules.l2norm") or __import__("fla.modules.l2norm", fromlist = ["x"])
    for tuner in (L.l2norm_fwd_kernel, L.l2norm_bwd_kernel):
        tuner.nargs = {}  # set by Autotuner.run before it prunes
        kept = tuner.prune_configs({"D": 128})
        assert {(c.kwargs["BT"], c.num_warps) for c in kept} == {(16, 4), (32, 4)}
        assert len(tuner.prune_configs({"D": 256})) == len(tuner.configs) == 25
        monkeypatch.setenv("UNSLOTH_FLA_FULL_AUTOTUNE", "1")
        assert len(tuner.prune_configs({"D": 128})) == 25
        monkeypatch.delenv("UNSLOTH_FLA_FULL_AUTOTUNE")
    # Still exact through the pruned autotune.
    x = torch.randn(4096, 128, device = "cuda", dtype = torch.float16)
    y, rstd = L.l2norm_fwd(x)
    ref = x.float() / (x.float().square().sum(-1, keepdim = True) + 1e-6).sqrt()
    torch.testing.assert_close(y.float(), ref, atol = 2e-3, rtol = 2e-3)
    assert (L.l2norm_fwd_kernel.best_config.kwargs["BT"], L.l2norm_fwd_kernel.best_config.num_warps) in ((16, 4), (32, 4))
