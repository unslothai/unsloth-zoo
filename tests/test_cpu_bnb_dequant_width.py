# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes <= 0.50.2 CPU NF4 dequant takes absmax per row on its AVX512 / NEON path, so layers whose
in_features is not a multiple of the 64-value block (80, 1040, 4304) dequantize with the wrong scales.
Once the CPU 4-bit gate says yes, dequant must match a torch dequant of the same packed weight."""

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
try:
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    from bitsandbytes.backends.cpu import ops as _cpu_ops  # noqa: F401
except Exception as _e:  # pragma: no cover - no bitsandbytes CPU backend
    pytest.skip(f"bitsandbytes CPU backend unavailable: {_e}", allow_module_level = True)

from unsloth_zoo.cpu import runtime


def _reference(q, st, shape):
    """Flat-block NF4 dequant (high nibble first), written out here so the test owns its oracle."""
    code = F.get_4bit_type("nf4", device = "cpu").float()
    data = q.reshape(-1).view(torch.uint8)
    idx = torch.stack([data >> 4, data & 0xF], dim = -1).reshape(-1)[: shape[0] * shape[1]].long()
    scale = st.absmax.float().repeat_interleave(st.blocksize)[: idx.numel()]
    return (code[idx] * scale).reshape(shape)


def _max_error(n, dtype, rows = 64):
    torch.manual_seed(0)
    W = torch.randn(rows, n) * torch.linspace(0.01, 3, n)
    q, st = F.quantize_4bit(W.to(dtype), blocksize = 64, quant_type = "nf4", compress_statistics = False)
    got = F.dequantize_4bit(q, st).float()
    want = _reference(q, st, (rows, n)).to(dtype).float()
    return float((got - want).abs().max())


def test_dequant_is_exact_once_the_cpu_4bit_gate_passes():
    runtime.cpu_bnb_4bit_supported.cache_clear()
    if not runtime.cpu_bnb_4bit_supported():
        pytest.skip("bitsandbytes 4-bit is not used on this CPU")
    for n in (80, 1040, 4304, 128, 4096):
        for dtype in (torch.float32, torch.bfloat16):
            assert _max_error(n, dtype) == 0.0, (n, dtype)


def test_flat_dequant_chunks_cover_every_block(monkeypatch):
    if not runtime.patch_bnb_cpu_dequant_4bit():
        pytest.skip("bitsandbytes here does not need the dequant fix")
    # Chunks of 2 * 1024 values: 80-wide rows end mid-chunk and mid-block, the last chunk is a partial block.
    monkeypatch.setattr(runtime, "_FLAT_DEQUANT_CHUNK", 2048)
    for rows, n in ((101, 80), (33, 1040), (7, 4304)):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            assert _max_error(n, dtype, rows = rows) == 0.0, (rows, n, dtype)


def test_patch_is_version_gated_and_idempotent(monkeypatch):
    from bitsandbytes.backends.cpu import ops as cpu_ops
    real = cpu_ops.lib
    base = getattr(real, "_unsloth_lib", real)
    if not any(hasattr(base, name) for name in runtime._BNB_CPU_DEQUANT_4BIT):
        pytest.skip("no native CPU 4-bit dequant in this bitsandbytes")
    try:
        for version, installed in (("0.51.0", False), ("0.50.3", False), ("0.50.3.dev0", False),
                                   ("0.50.2", True), ("0.49.1", True)):
            cpu_ops.lib = base
            monkeypatch.setattr(bnb, "__version__", version)
            assert runtime.patch_bnb_cpu_dequant_4bit() is installed, version
            assert isinstance(cpu_ops.lib, runtime._FlatDequantLib) is installed, version
        assert runtime.patch_bnb_cpu_dequant_4bit() is True
        assert cpu_ops.lib._unsloth_lib is base  # never wrapped twice
        # Everything but the six dequant symbols is the native library itself.
        name = next(n for n in ("cquantize_blockwise_cpu_fp32", "cdequantize_blockwise_cpu_fp32") if hasattr(base, n))
        assert getattr(cpu_ops.lib, name) is getattr(base, name)
    finally:
        cpu_ops.lib = real


def test_flat_dequant_passes_whole_block_rows_through():
    import ctypes as ct
    calls = []
    wrapped = runtime._flat_dequant_4bit(lambda *a: calls.append(a), 2)
    args = (ct.c_void_p(4096), ct.c_void_p(8192), ct.c_void_p(16384), ct.c_longlong(64), ct.c_longlong(3), ct.c_longlong(128))
    wrapped(*args)
    assert calls == [args]
    calls.clear()
    wrapped(ct.c_void_p(4096), ct.c_void_p(8192), ct.c_void_p(16384), ct.c_longlong(64), ct.c_longlong(3), ct.c_longlong(80))
    (A, absmax, out, bs, m, n), = calls
    assert (A.value, absmax.value, out.value, bs.value, m.value, n.value) == (4096, 8192, 16384, 64, 1, 240)


_PROBE_CHILD = textwrap.dedent(
    """
    import sys, torch, bitsandbytes as bnb
    from unsloth_zoo.cpu import runtime
    assert runtime.__file__.startswith(sys.argv[1]), runtime.__file__
    dtype = torch.bfloat16 if runtime.cpu_bf16_supported() else torch.float32
    print("UNPATCHED_PROBE", runtime._bnb_cpu_dequant_4bit_correct(torch, bnb, dtype))
    print("PATCHED", runtime.patch_bnb_cpu_dequant_4bit())
    print("PATCHED_PROBE", runtime._bnb_cpu_dequant_4bit_correct(torch, bnb, dtype))
    """
)


def test_width_80_probe_sees_the_wrong_scales():
    # A fresh process: bitsandbytes there has not been patched yet.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _PROBE_CHILD, str(ZOO_ROOT)], env = env,
                          capture_output = True, text = True, timeout = 600)
    lines = dict(l.split(" ", 1) for l in proc.stdout.splitlines() if l.split(" ", 1)[0].isupper() and " " in l)
    assert "PATCHED_PROBE" in lines, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"
    assert lines["PATCHED_PROBE"] == "True"
    if lines["PATCHED"] == "True" and "avx512f" in runtime.cpu_flags():
        # Affected release on an AVX512 / NEON host: the probe must have caught the unpatched dequant.
        assert lines["UNPATCHED_PROBE"] == "False", proc.stdout
