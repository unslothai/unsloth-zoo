# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Regression tests for the MLX GPTQ/AWQ pre-quantized rescope.

mlx-lm's native packed-quant path (mlx_lm.utils._transform_awq_weights, >=0.30.4)
assumes the AutoAWQ tensor layout: it raises on any ``.g_idx`` tensor and unpacks
``.qweight`` as AWQ ``[in, out//pack]``. Standard AutoGPTQ ships a ``.g_idx``
(even for desc_act=False) and packs ``.qweight`` as ``[in//pack, out]``, so no
released mlx-lm can load AutoGPTQ natively. These tests pin that GPTQ is always
dequantized locally (with correct math, including when g_idx is omitted) while
standard AWQ is deferred to mlx-lm.
"""

from __future__ import annotations

import json
import os
import glob
import tempfile

import numpy as np
import pytest


def _ensure_bitsandbytes_importable():
    """unsloth_zoo/__init__ imports temporary_patches -> bitsandbytes, which
    raises on this CPU/CUDA-mismatched host. Install a minimal stub (only the
    import-time surface: nn.Params4bit, functional.dequantize_4bit) so importing
    unsloth_zoo.mlx.loader works in isolation, mirroring the full-suite ordering
    that already makes these MLX tests importable."""
    import sys
    import types
    try:
        import bitsandbytes  # noqa: F401
        return
    except Exception:
        pass
    bnb = types.ModuleType("bitsandbytes")
    nn = types.ModuleType("bitsandbytes.nn")
    functional = types.ModuleType("bitsandbytes.functional")

    class Params4bit:  # minimal placeholder
        pass

    nn.Params4bit = Params4bit
    functional.dequantize_4bit = lambda *a, **k: None
    bnb.nn = nn
    bnb.functional = functional
    sys.modules["bitsandbytes"] = bnb
    sys.modules["bitsandbytes.nn"] = nn
    sys.modules["bitsandbytes.functional"] = functional


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    _ensure_bitsandbytes_importable()
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# ---------------------------------------------------------------------------
# AutoGPTQ 4-bit packing helpers (sequential nibble order, [in//8, out]).
# ---------------------------------------------------------------------------

def _pack_qweight_gptq(intmat):
    inn, out = intmat.shape
    q = np.zeros((inn // 8, out), dtype=np.uint32)
    for r in range(inn // 8):
        for k in range(8):
            q[r] |= (intmat[8 * r + k].astype(np.uint32) & 0xF) << (4 * k)
    return q.astype(np.int32)


def _pack_qzeros_gptq(zmat):
    g, out = zmat.shape
    q = np.zeros((g, out // 8), dtype=np.uint32)
    for c in range(out // 8):
        for k in range(8):
            q[:, c] |= (zmat[:, 8 * c + k].astype(np.uint32) & 0xF) << (4 * k)
    return q.astype(np.int32)


def _make_gptq_tensors(inn=16, out=8, gs=8, seed=0, desc_act=False):
    rng = np.random.default_rng(seed)
    groups = inn // gs
    q = rng.integers(0, 16, size=(inn, out)).astype(np.int64)
    stored_zero = rng.integers(0, 15, size=(groups, out)).astype(np.int64)
    scales = rng.random((groups, out)).astype(np.float32) + 0.1
    if desc_act:
        perm = rng.permutation(inn)
        g_idx = (np.arange(inn) // gs)[perm].astype(np.int32)
    else:
        g_idx = (np.arange(inn) // gs).astype(np.int32)
    ref = np.zeros((out, inn), dtype=np.float32)
    for i in range(inn):
        g = int(g_idx[i])
        ref[:, i] = (q[i] - (stored_zero[g] + 1)) * scales[g]
    return q, stored_zero, scales, g_idx, ref


# ---------------------------------------------------------------------------
# FIX 1a: GPTQ is always rejected (dequantized locally); AWQ GEMM defers.
# ---------------------------------------------------------------------------

def test_reject_prequant_gptq_always_true_on_supported_mlx_lm(monkeypatch):
    import unsloth_zoo.mlx.loader as ml
    monkeypatch.setattr(ml, "_mlx_lm_supports_native_prequant", lambda: True)
    # Even a plain desc_act=False config must be rejected -> dequantized here,
    # because mlx-lm raises on the trivial g_idx AutoGPTQ still ships.
    assert ml._mlx_lm_would_reject_prequant(
        "/nonexistent", "gptq", {"bits": 4, "group_size": 128, "desc_act": False}
    ) is True
    assert ml._mlx_lm_would_reject_prequant(
        "/nonexistent", "gptq", {"bits": 4, "group_size": 128, "desc_act": True}
    ) is True


def test_reject_prequant_awq_defers_on_supported_mlx_lm(monkeypatch):
    import unsloth_zoo.mlx.loader as ml
    monkeypatch.setattr(ml, "_mlx_lm_supports_native_prequant", lambda: True)
    assert ml._mlx_lm_would_reject_prequant(
        "/nonexistent", "awq", {"bits": 4, "group_size": 128}
    ) is False


@pytest.mark.parametrize("group_size", [-1, 0])
def test_reject_prequant_full_group_awq_dequants_locally(monkeypatch, group_size):
    # Full-group / per-column GEMM AWQ (q_group_size <= 0) cannot be deferred to
    # mlx-lm: its native _transform_awq_weights forwards the raw non-positive
    # group size into nn.quantize, which MLX's affine kernels reject. It must be
    # routed to the local dequantizer instead (which the round-4 fix handles).
    import unsloth_zoo.mlx.loader as ml
    monkeypatch.setattr(ml, "_mlx_lm_supports_native_prequant", lambda: True)
    assert ml._mlx_lm_would_reject_prequant(
        "/nonexistent", "awq", {"bits": 4, "group_size": group_size}
    ) is True
    # The alternate AutoAWQ key spelling is honoured too.
    assert ml._mlx_lm_would_reject_prequant(
        "/nonexistent", "awq", {"bits": 4, "q_group_size": group_size}
    ) is True


def test_awq_group_size_is_full_predicate():
    import unsloth_zoo.mlx.loader as ml
    # Non-positive group sizes are full-group / per-column; positive ones grouped.
    assert ml._awq_group_size_is_full({"group_size": -1}) is True
    assert ml._awq_group_size_is_full({"group_size": 0}) is True
    assert ml._awq_group_size_is_full({"q_group_size": -1}) is True
    assert ml._awq_group_size_is_full({"group_size": 128}) is False
    assert ml._awq_group_size_is_full({"group_size": 64}) is False
    # A missing / unparseable group size defers to the grouped native path.
    assert ml._awq_group_size_is_full({}) is False
    assert ml._awq_group_size_is_full({"group_size": None}) is False
    assert ml._awq_group_size_is_full({"group_size": "bad"}) is False
    assert ml._awq_group_size_is_full(None) is False


def test_reject_prequant_old_mlx_lm_dequants_everything(monkeypatch):
    import unsloth_zoo.mlx.loader as ml
    monkeypatch.setattr(ml, "_mlx_lm_supports_native_prequant", lambda: False)
    assert ml._mlx_lm_would_reject_prequant("/x", "awq", {"bits": 4}) is True
    assert ml._mlx_lm_would_reject_prequant("/x", "gptq", {"bits": 4}) is True


def test_dead_gptq_g_idx_helper_removed():
    import unsloth_zoo.mlx.loader as ml
    assert not hasattr(ml, "_gptq_g_idx_is_permuted")


# ---------------------------------------------------------------------------
# GPTQ dequant math round-trips (desc_act False and True).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("desc_act", [False, True])
def test_gptq_dequantize_weight_roundtrip(desc_act):
    import mlx.core as mx
    import unsloth_zoo.mlx.loader as ml
    q, zero, scales, g_idx, ref = _make_gptq_tensors(desc_act=desc_act, seed=1)
    dense = ml._gptq_dequantize_weight(
        mx.array(_pack_qweight_gptq(q)),
        mx.array(_pack_qzeros_gptq(zero)),
        mx.array(scales),
        mx.array(g_idx),
        bits=4,
    )
    got = np.asarray(dense).astype(np.float32)
    assert np.allclose(got, ref, atol=1e-3), np.abs(got - ref).max()


# ---------------------------------------------------------------------------
# FIX 1b: materialize handles a GPTQ checkpoint that omits g_idx (synthesis),
# and strips the HF quantization metadata.
# ---------------------------------------------------------------------------

def _write_gptq_repo(tmp, with_g_idx, gs=8, seed=2):
    import mlx.core as mx
    q, zero, scales, g_idx, ref = _make_gptq_tensors(gs=gs, seed=seed)
    name = "model.layers.0.self_attn.q_proj"
    tensors = {
        name + ".qweight": mx.array(_pack_qweight_gptq(q)),
        name + ".qzeros": mx.array(_pack_qzeros_gptq(zero)),
        name + ".scales": mx.array(scales),
    }
    if with_g_idx:
        tensors[name + ".g_idx"] = mx.array(g_idx)
    mx.save_safetensors(os.path.join(tmp, "model.safetensors"), tensors)
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({"model_type": "llama"}, f)
    return name, ref


@pytest.mark.parametrize("with_g_idx", [True, False])
def test_materialize_gptq_with_and_without_g_idx(with_g_idx):
    import mlx.core as mx
    import unsloth_zoo.mlx.loader as ml
    gs = 8
    tmp = tempfile.mkdtemp(prefix="rescope_gptq_")
    name, ref = _write_gptq_repo(tmp, with_g_idx=with_g_idx, gs=gs)
    quant_config = {"quant_method": "gptq", "bits": 4, "group_size": gs}
    cfg = {"model_type": "llama", "quantization_config": quant_config}
    out_dir, new_cfg = ml._materialize_dequantized_hf_checkpoint(
        tmp, cfg, "gptq", quant_config,
    )
    # HF quant metadata stripped.
    assert "quantization_config" not in new_cfg
    # Dense weight produced and numerically correct (synthesized g_idx path too).
    weights = mx.load(glob.glob(os.path.join(out_dir, "*.safetensors"))[0])
    assert name + ".weight" in weights
    assert name + ".qweight" not in weights
    got = np.asarray(weights[name + ".weight"]).astype(np.float32)
    assert np.allclose(got, ref, atol=1e-2), np.abs(got - ref).max()


def test_is_dropped_dequant_sidecar_predicate():
    import unsloth_zoo.mlx.loader as ml
    # Packed weights, their shard index, and the quant sidecars are dropped.
    for dropped in (
        "model.safetensors", "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "quantize_config.json", "quant_config.json",
    ):
        assert ml._is_dropped_dequant_sidecar(dropped) is True, dropped
    # Ordinary metadata the dense checkpoint still needs is kept.
    for kept in (
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "tokenizer.model", "special_tokens_map.json", "generation_config.json",
    ):
        assert ml._is_dropped_dequant_sidecar(kept) is False, kept


def test_materialize_gptq_drops_quant_sidecars():
    import unsloth_zoo.mlx.loader as ml
    gs = 8
    tmp = tempfile.mkdtemp(prefix="rescope_gptq_sidecar_")
    name, ref = _write_gptq_repo(tmp, with_g_idx=True, gs=gs)
    # A real GPTQ repo ships quantize_config.json (AWQ ships quant_config.json)
    # describing the packed tensors, plus ordinary tokenizer metadata.
    with open(os.path.join(tmp, "quantize_config.json"), "w") as f:
        json.dump({"bits": 4, "group_size": gs, "quant_method": "gptq"}, f)
    with open(os.path.join(tmp, "quant_config.json"), "w") as f:
        json.dump({"bits": 4, "q_group_size": gs, "version": "GEMM"}, f)
    with open(os.path.join(tmp, "tokenizer_config.json"), "w") as f:
        json.dump({"model_max_length": 2048}, f)
    quant_config = {"quant_method": "gptq", "bits": 4, "group_size": gs}
    cfg = {"model_type": "llama", "quantization_config": quant_config}
    out_dir, _ = ml._materialize_dequantized_hf_checkpoint(
        tmp, cfg, "gptq", quant_config,
    )
    present = set(os.listdir(out_dir))
    # The now-invalid quantization sidecars must not survive into the dense
    # checkpoint (they would let a loader mis-detect it as still packed).
    assert "quantize_config.json" not in present
    assert "quant_config.json" not in present
    # Ordinary metadata is preserved, and config.json is the stripped dense one.
    assert "tokenizer_config.json" in present
    assert "config.json" in present
    with open(os.path.join(out_dir, "config.json")) as f:
        assert "quantization_config" not in json.load(f)


# ---------------------------------------------------------------------------
# FIX 3: dense / 16-bit / full-finetuning requests resolve to a disabled spec,
# which the loader uses to force the fp16 dequant path (never a quantized base).
# The default load_in_4bit=True stays enabled so standard AWQ defers to mlx-lm.
# ---------------------------------------------------------------------------

def _resolve(**overrides):
    import unsloth_zoo.mlx.loader as ml
    kwargs = dict(
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=False,
        load_in_fp8=False, load_in_mxfp4=False, load_in_nvfp4=False,
        full_finetuning=False, q_bits=None, q_group_size=None, q_mode=None,
        mlx_quantization_config=None, quantization_config=None,
        quant_predicate=None, quantize_modules=None, force_requantize=False,
    )
    kwargs.update(overrides)
    return ml._resolve_mlx_quantization_spec(**kwargs)


def test_spec_disabled_for_16bit_and_full_finetuning():
    assert _resolve(load_in_16bit=True).enabled is False
    assert _resolve(full_finetuning=True).enabled is False


def test_spec_enabled_for_default_4bit():
    assert _resolve(load_in_4bit=True).enabled is True


# ---------------------------------------------------------------------------
# Integration helpers to drive FastMLXModel.from_pretrained with mocks.
# ---------------------------------------------------------------------------

class _FakeGroup:
    def __init__(self, size=2, rank=0, name="group"):
        self._size, self._rank, self.name = size, rank, name

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _write_repo(path, quant_method="gptq", adapter_base=None):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump({
            "model_type": "llama",
            "quantization_config": {
                "quant_method": quant_method, "bits": 4, "group_size": 128,
            },
        }, f)
    if adapter_base is not None:
        with open(os.path.join(path, "adapter_config.json"), "w") as f:
            json.dump(
                {"base_model_name_or_path": adapter_base, "peft_type": "LORA"}, f,
            )
    return path


# ---------------------------------------------------------------------------
# FIX 4: a distributed GPTQ/AWQ load that would need dequant is rejected with a
# clear message (the metadata-only distributed snapshot has no packed weights),
# instead of the raw "no .safetensors weights found" crash.
# ---------------------------------------------------------------------------

def test_distributed_gptq_dequant_rejected_clearly(monkeypatch, tmp_path):
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import FastMLXModel

    repo = _write_repo(str(tmp_path / "gptq"))
    monkeypatch.setattr(mlx_lm_utils, "_download", lambda *a, **k: repo)
    monkeypatch.setattr(
        loader, "_materialize_dequantized_hf_checkpoint",
        lambda *a, **k: pytest.fail("must reject before dequantizing"),
    )
    with pytest.raises(NotImplementedError, match=r"distributed MLX loading of the GPTQ"):
        FastMLXModel.from_pretrained(
            repo, text_only=True, tensor_group=_FakeGroup(name="tensor"),
        )


# ---------------------------------------------------------------------------
# FIX 2: a LoRA adapter dir carrying a copied GPTQ base config.json must NOT be
# dequantized as a full quantized checkpoint (there are no packed weights in it);
# it takes the adapter branch and dequantizes the recursively-loaded base only.
# ---------------------------------------------------------------------------

def test_adapter_dir_with_gptq_base_config_not_dequantized(monkeypatch, tmp_path):
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import FastMLXModel

    adapter_dir = _write_repo(str(tmp_path / "adapter"), adapter_base="fake/base")
    base_dir = _write_repo(str(tmp_path / "base"))

    def _fake_download(name, *a, **k):
        return base_dir if "base" in str(name) else adapter_dir
    monkeypatch.setattr(mlx_lm_utils, "_download", _fake_download)

    seen = []

    def _fake_materialize(local_path, *a, **k):
        # Record which checkpoint got dequantized, then stop the load. The
        # adapter branch is a declared LoRA, so this surfaces as a clean
        # "failed to load the LoRA adapter" error (no network fallback).
        seen.append(str(local_path))
        raise RuntimeError("stop-after-dequant")
    monkeypatch.setattr(
        loader, "_materialize_dequantized_hf_checkpoint", _fake_materialize,
    )

    with pytest.raises(RuntimeError, match="failed to load the LoRA adapter"):
        FastMLXModel.from_pretrained(adapter_dir, text_only=True)

    # Only the recursively-loaded base was dequantized; the adapter dir itself
    # was routed to the adapter branch (never materialized).
    assert seen == [str(base_dir)]


# ---------------------------------------------------------------------------
# AutoAWQ GEMM 4-bit packing helper ([in, out//8], interleave 0,4,1,5,2,6,3,7).
# ---------------------------------------------------------------------------

# Natural output column k is stored at packed nibble slot _AWQ_ORDER[k]; this is
# the inverse of loader._AWQ_REVERSE_ORDER used on unpack.
_AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def _pack_awq(intmat):
    rows, cols = intmat.shape
    q = np.zeros((rows, cols // 8), dtype=np.uint32)
    for b in range(cols // 8):
        for k in range(8):
            q[:, b] |= (intmat[:, 8 * b + k].astype(np.uint32) & 0xF) << (4 * _AWQ_ORDER[k])
    return q.astype(np.int32)


def _make_awq_tensors(inn=8, out=16, gs=4, seed=3):
    rng = np.random.default_rng(seed)
    groups = 1 if gs <= 0 else inn // gs
    w_int = rng.integers(0, 16, size=(inn, out)).astype(np.int64)   # [in, out]
    z_int = rng.integers(0, 16, size=(groups, out)).astype(np.int64)
    scales = rng.random((groups, out)).astype(np.float32) + 0.1
    eff_gs = inn if gs <= 0 else gs
    ref = np.zeros((out, inn), dtype=np.float32)                    # [out, in]
    for i in range(inn):
        g = i // eff_gs
        ref[:, i] = (w_int[i] - z_int[g]) * scales[g]
    return _pack_awq(w_int), _pack_awq(z_int), scales, ref


# ---------------------------------------------------------------------------
# AWQ layout guards: full-group (group_size <= 0) dequant math, and rejection of
# non-GEMM AWQ variants that neither mlx-lm nor the local dequant can decode.
# ---------------------------------------------------------------------------

def test_awq_dequantize_weight_grouped_roundtrip():
    # Standard grouped AWQ (group_size > 0) still round-trips; this also pins the
    # test packer against the loader's unpack.
    import mlx.core as mx
    import unsloth_zoo.mlx.loader as ml
    qw, qz, scales, ref = _make_awq_tensors(inn=8, out=16, gs=4, seed=5)
    dense = ml._awq_dequantize_weight(mx.array(qw), mx.array(qz), mx.array(scales), 4, bits=4)
    got = np.asarray(dense).astype(np.float32)
    assert np.allclose(got, ref, atol=1e-2), np.abs(got - ref).max()


@pytest.mark.parametrize("group_size", [-1, 0])
def test_awq_dequantize_weight_full_group(group_size):
    # AutoAWQ full-group / per-column (q_group_size <= 0): a single group spans
    # the whole input dim. Without the guard, arange//-1 mis-gathers the single
    # row of scales/zeros and reconstructs wrong weights (or indexes negatively).
    import mlx.core as mx
    import unsloth_zoo.mlx.loader as ml
    qw, qz, scales, ref = _make_awq_tensors(inn=8, out=16, gs=-1, seed=6)
    dense = ml._awq_dequantize_weight(
        mx.array(qw), mx.array(qz), mx.array(scales), group_size, bits=4,
    )
    got = np.asarray(dense).astype(np.float32)
    assert np.allclose(got, ref, atol=1e-2), np.abs(got - ref).max()


def test_awq_quant_config_is_gemm_predicate():
    import unsloth_zoo.mlx.loader as ml
    # GEMM (default) and a blank/missing version are the native layout.
    assert ml._awq_quant_config_is_gemm({"version": "GEMM"}) is True
    assert ml._awq_quant_config_is_gemm({"version": "gemm"}) is True
    assert ml._awq_quant_config_is_gemm({}) is True
    assert ml._awq_quant_config_is_gemm({"version": ""}) is True
    # Non-GEMM variants pack differently and must be rejected upstream.
    for bad in ("GEMV", "gemv", "gemv_fast", "marlin", "exllama", "ipex"):
        assert ml._awq_quant_config_is_gemm({"version": bad}) is False, bad


def test_non_gemm_awq_rejected_before_native_load(monkeypatch, tmp_path):
    # A GEMV-packed AWQ checkpoint must fail loud instead of silently deferring
    # to mlx-lm (whose loader assumes GEMM) or being locally mis-dequantized.
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import FastMLXModel

    repo = str(tmp_path / "awq_gemv")
    os.makedirs(repo, exist_ok=True)
    with open(os.path.join(repo, "config.json"), "w") as f:
        json.dump({
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "awq", "bits": 4, "group_size": 128,
                "version": "GEMV",
            },
        }, f)
    monkeypatch.setattr(mlx_lm_utils, "_download", lambda *a, **k: repo)
    monkeypatch.setattr(
        loader, "_materialize_dequantized_hf_checkpoint",
        lambda *a, **k: pytest.fail("non-GEMM AWQ must not be locally dequantized"),
    )
    with pytest.raises(NotImplementedError, match=r"'GEMV' layout"):
        FastMLXModel.from_pretrained(repo, text_only=True)


def test_full_group_gemm_awq_routes_to_local_dequant(monkeypatch, tmp_path):
    # A full-group / per-column GEMM AWQ base (q_group_size == -1) on a default
    # 4-bit LoRA load must NOT be deferred to mlx-lm's native path (which cannot
    # apply the group_size <= 0 normalization) but instead be dequantized
    # locally. Prove the load reaches _materialize_dequantized_hf_checkpoint.
    import mlx_lm.utils as mlx_lm_utils
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import FastMLXModel

    repo = str(tmp_path / "awq_full_group")
    os.makedirs(repo, exist_ok=True)
    with open(os.path.join(repo, "config.json"), "w") as f:
        json.dump({
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "awq", "bits": 4, "group_size": -1,
                "version": "GEMM",
            },
        }, f)
    monkeypatch.setattr(mlx_lm_utils, "_download", lambda *a, **k: repo)
    monkeypatch.setattr(loader, "_mlx_lm_supports_native_prequant", lambda: True)

    class _ReachedLocalDequant(Exception):
        pass

    def _sentinel(*a, **k):
        raise _ReachedLocalDequant

    monkeypatch.setattr(
        loader, "_materialize_dequantized_hf_checkpoint", _sentinel,
    )
    # Default (4-bit) load keeps quantization_spec.enabled True, so this is not
    # the _force_dense_dequant path -- reaching local dequant proves the AWQ
    # full-group routing, not the dense-load fallback.
    with pytest.raises(_ReachedLocalDequant):
        FastMLXModel.from_pretrained(repo, text_only=True)
