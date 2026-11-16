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
