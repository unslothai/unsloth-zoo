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

"""Tests UNSLOTH_GDN_BACKEND=flashqla opt-in routing on CPU: fla and flash_qla are stubbed, checking routing, the kernel-hub signature contract and the fallback."""

import inspect
import os
import sys
import types

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    from unsloth_zoo.temporary_patches import fla_vendor as fv
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)


def _fla_chunk(
    q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
    use_qk_l2norm_in_kernel = False, use_beta_sigmoid_in_kernel = False,
    allow_neg_eigval = False, state_v_first = False, cu_seqlens = None,
    cu_seqlens_cpu = None, cp_context = None, **kwargs,
):
    return ("fla", kwargs)


@pytest.fixture
def fake_env(monkeypatch):
    gdr = types.ModuleType("fla.ops.gated_delta_rule")
    gdr.chunk_gated_delta_rule = _fla_chunk
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", gdr)
    modeling = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    modeling.chunk_gated_delta_rule = _fla_chunk
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)

    calls = []

    def flash_chunk(**kwargs):
        calls.append(kwargs)
        return ("flashqla", kwargs)

    fake_flash = types.SimpleNamespace(chunk_gated_delta_rule = flash_chunk)
    state = dict(fv._flashqla_state, skipped = set())
    monkeypatch.setattr(fv, "_flashqla_state", state)
    monkeypatch.setattr(fv, "_load_flashqla", lambda: state.update(module = fake_flash) or fake_flash)
    return gdr, modeling, calls


def _supported_only_when(monkeypatch, predicate):
    monkeypatch.setattr(
        fv, "_flashqla_unsupported",
        lambda q, *a: None if predicate(q) else "unsupported",
    )


def test_unset_leaves_fla_untouched(monkeypatch, fake_env):
    gdr, modeling, _ = fake_env
    monkeypatch.delenv("UNSLOTH_GDN_BACKEND", raising = False)
    assert fv._install_gdn_backend() is False
    assert gdr.chunk_gated_delta_rule is _fla_chunk
    assert modeling.chunk_gated_delta_rule is _fla_chunk


def test_flashqla_routes_eligible_calls_and_falls_back(monkeypatch, fake_env):
    gdr, modeling, calls = fake_env
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")
    _supported_only_when(monkeypatch, lambda q: q == "eligible")
    assert fv._install_gdn_backend() is True
    wrapped = gdr.chunk_gated_delta_rule
    assert wrapped is not _fla_chunk
    # Pre-#47630 modeling globals are rebound too.
    assert modeling.chunk_gated_delta_rule is wrapped
    # transformers filters kwargs by the implementation's signature.
    assert list(inspect.signature(wrapped).parameters) == list(inspect.signature(_fla_chunk).parameters)

    out = wrapped("eligible", "k", "v", "g", "beta", initial_state = "h0",
                  output_final_state = True, use_qk_l2norm_in_kernel = True, cu_seqlens = "cu")
    assert out[0] == "flashqla"
    sent = calls[-1]
    assert (sent["q"], sent["initial_state"], sent["cu_seqlens"]) == ("eligible", "h0", "cu")
    assert sent["output_final_state"] is True and sent["use_qk_l2norm_in_kernel"] is True
    assert fv._flashqla_state["calls"] == 1

    out = wrapped("other", "k", "v", "g", "beta", use_gate_in_kernel = True)
    assert out == ("fla", {"use_gate_in_kernel": True})
    assert fv._flashqla_state["calls"] == 1

    # Idempotent: a second patch run does not wrap the wrapper.
    assert fv._install_gdn_backend() is True
    assert gdr.chunk_gated_delta_rule is wrapped


def test_flashqla_failure_falls_back_once_per_dtype(monkeypatch, fake_env):
    # fp16 on SM100 can pass the verifier yet fail to compile (TileLang 0.1.12, nvcc 13.1); that call and later
    # same-dtype calls fall back to fla with one warning.
    gdr, _, calls = fake_env
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")
    _supported_only_when(monkeypatch, lambda q: True)
    built = []

    def flash_chunk(**kwargs):
        built.append(kwargs["q"].dtype)
        if kwargs["q"].dtype == "float16":
            raise RuntimeError("Compilation error:\nno suitable user-defined conversion")
        return ("flashqla", kwargs)

    fv._flashqla_state["module"] = types.SimpleNamespace(chunk_gated_delta_rule = flash_chunk)
    monkeypatch.setattr(fv, "_load_flashqla", lambda: fv._flashqla_state["module"])
    warnings = []
    monkeypatch.setattr(fv.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    assert fv._install_gdn_backend() is True
    wrapped = gdr.chunk_gated_delta_rule

    fp16 = types.SimpleNamespace(dtype = "float16")
    bf16 = types.SimpleNamespace(dtype = "bfloat16")
    assert wrapped(fp16, "k", "v", "g", "beta")[0] == "fla"
    assert wrapped(fp16, "k", "v", "g", "beta")[0] == "fla"
    assert built == ["float16"]
    assert len(warnings) == 1 and "float16" in warnings[0] and "Compilation error" in warnings[0]
    assert wrapped(bf16, "k", "v", "g", "beta")[0] == "flashqla"
    assert built == ["float16", "bfloat16"]


def test_flashqla_failure_releases_its_buffers_before_fla(monkeypatch, fake_env):
    # The fallback must run outside the except block, since the traceback would keep the failed backend's scratch alive
    # while fla allocates, causing an avoidable OOM.
    import gc
    import weakref
    torch = pytest.importorskip("torch")
    gdr, _, _ = fake_env
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")
    _supported_only_when(monkeypatch, lambda q: True)
    scratch = []

    def flash_chunk(**kwargs):
        buffer = torch.empty(16)
        scratch.append(weakref.ref(buffer))
        raise RuntimeError("Compilation error")

    alive_in_fla = []

    def fla_chunk(
        q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
        use_qk_l2norm_in_kernel = False, use_beta_sigmoid_in_kernel = False,
        allow_neg_eigval = False, state_v_first = False, cu_seqlens = None,
        cu_seqlens_cpu = None, cp_context = None, **kwargs,
    ):
        alive_in_fla.append(scratch[-1]() is not None)
        return ("fla", kwargs)

    monkeypatch.setattr(gdr, "chunk_gated_delta_rule", fla_chunk)
    fv._flashqla_state["module"] = types.SimpleNamespace(chunk_gated_delta_rule = flash_chunk)
    monkeypatch.setattr(fv, "_load_flashqla", lambda: fv._flashqla_state["module"])
    monkeypatch.setattr(fv.logger, "warning", lambda msg, *a, **k: None)
    assert fv._install_gdn_backend() is True
    gc.disable()
    try:
        out = gdr.chunk_gated_delta_rule(types.SimpleNamespace(dtype = "float16"), "k", "v", "g", "beta")
    finally:
        gc.enable()
    assert out[0] == "fla"
    assert alive_in_fla == [False]


def test_flashqla_oom_is_not_swallowed(monkeypatch, fake_env):
    torch = pytest.importorskip("torch")
    gdr, _, _ = fake_env
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")
    _supported_only_when(monkeypatch, lambda q: True)

    def flash_chunk(**kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    fv._flashqla_state["module"] = types.SimpleNamespace(chunk_gated_delta_rule = flash_chunk)
    monkeypatch.setattr(fv, "_load_flashqla", lambda: fv._flashqla_state["module"])
    assert fv._install_gdn_backend() is True
    with pytest.raises(torch.cuda.OutOfMemoryError):
        gdr.chunk_gated_delta_rule(types.SimpleNamespace(dtype = "float16"), "k", "v", "g", "beta")
    assert not fv._flashqla_state.get("failed_calls")


def test_flashqla_unavailable_keeps_fla(monkeypatch, fake_env):
    gdr, _, _ = fake_env
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")
    monkeypatch.setattr(fv, "_load_flashqla", lambda: None)
    assert fv._install_gdn_backend() is False
    assert gdr.chunk_gated_delta_rule is _fla_chunk


def test_extra_flags_reach_the_verifier(monkeypatch, fake_env):
    gdr, _, _ = fake_env
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")
    seen = {}

    def verifier(q, k, v, g, beta, initial_state, cu_seqlens, cp_context, extra):
        seen.update(extra)
        return "unsupported"

    monkeypatch.setattr(fv, "_flashqla_unsupported", verifier)
    fv._install_gdn_backend()
    out = gdr.chunk_gated_delta_rule("q", "k", "v", "g", "beta", allow_neg_eigval = True)
    assert out[0] == "fla" and seen["allow_neg_eigval"] is True


def _fla_chunk_named(
    q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
    use_qk_l2norm_in_kernel = False, use_gate_in_kernel = False, head_first = False,
    cu_seqlens = None, **kwargs,
):
    return ("fla", kwargs)


@pytest.mark.parametrize("flag", ["use_gate_in_kernel", "head_first"])
def test_named_flags_of_the_original_reach_the_verifier(monkeypatch, fake_env, flag):
    gdr, _, calls = fake_env
    gdr.chunk_gated_delta_rule = _fla_chunk_named
    monkeypatch.setenv("UNSLOTH_GDN_BACKEND", "flashqla")
    seen = {}

    def verifier(q, k, v, g, beta, initial_state, cu_seqlens, cp_context, extra):
        seen.update(extra)
        return "flag" if extra.get(flag) else None

    monkeypatch.setattr(fv, "_flashqla_unsupported", verifier)
    fv._install_gdn_backend()
    out = gdr.chunk_gated_delta_rule("q", "k", "v", "g", "beta", **{flag: True})
    assert out[0] == "fla" and seen[flag] is True and calls == []


def test_verifier_rejects_what_flashqla_cannot_run():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    state = fv._flashqla_state
    previous = state["capability"]
    state["capability"] = tuple(torch.cuda.get_device_capability())
    try:
        def make(K = 128, dtype = torch.bfloat16):
            q = torch.zeros(1, 8, 2, K, device = "cuda", dtype = dtype)
            v = torch.zeros(1, 8, 2, 128, device = "cuda", dtype = dtype)
            g = torch.zeros(1, 8, 2, device = "cuda")
            return q, q, v, g, g
        assert fv._flashqla_unsupported(*make(), None, None, None, {}) is None
        assert fv._flashqla_unsupported(*make(K = 64), None, None, None, {}) is not None
        assert fv._flashqla_unsupported(*make(dtype = torch.float32), None, None, None, {}) is not None
        q, k, v, g, b = make()
        q2 = q.expand(2, -1, -1, -1)
        assert fv._flashqla_unsupported(q2, q2, v.expand(2, -1, -1, -1), g, b, None, "cu", None, {}) is not None
        assert fv._flashqla_unsupported(q, k, v, g, b, None, None, object(), {}) is not None
        assert fv._flashqla_unsupported(q, k, v, g, b, None, None, None, {"use_gate_in_kernel": True}) is not None
    finally:
        state["capability"] = previous


def _qla_chunk_010(
    q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
    use_qk_l2norm_in_kernel = False, cu_seqlens = None, head_first = False,
):
    return "flashqla-0.1.0"


def _qla_chunk_011(
    q, k, v, g, beta, scale = None, initial_state = None, output_final_state = False,
    use_qk_l2norm_in_kernel = False, cu_seqlens = None, head_first = False,
    state_v_first = False,
):
    return "flashqla-0.1.1"


@pytest.mark.parametrize("chunk, version, accepted", [
    (_qla_chunk_010, None, False), (_qla_chunk_011, None, False),
    (_qla_chunk_011, "0.1.1", False), (_qla_chunk_011, "0.1.2", True), (_qla_chunk_011, "0.1.3", True),
])
def test_load_rejects_flashqla_before_0_1_2(monkeypatch, chunk, version, accepted):
    import torch
    monkeypatch.setattr(fv, "_flashqla_version", lambda module: version)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    monkeypatch.setitem(sys.modules, "flash_qla", types.SimpleNamespace(chunk_gated_delta_rule = chunk))
    monkeypatch.setattr(fv, "_flashqla_state", dict(fv._flashqla_state, module = None, failed = False))
    loaded = fv._load_flashqla()
    assert (loaded is not None) == accepted
    assert fv._flashqla_state["failed"] == (not accepted)


@pytest.mark.parametrize("capability, allow, accepted", [
    ((12, 0), None, False), ((12, 1), None, False), ((12, 0), "1", True), ((9, 0), None, True),
])
def test_load_keeps_fla_on_sm120_unless_allowed(monkeypatch, capability, allow, accepted):
    import torch
    monkeypatch.setattr(fv, "_flashqla_version", lambda module: "0.1.3")
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setitem(sys.modules, "flash_qla", types.SimpleNamespace(chunk_gated_delta_rule = _qla_chunk_011))
    monkeypatch.setattr(fv, "_flashqla_state", dict(fv._flashqla_state, module = None, failed = False))
    if allow is None: monkeypatch.delenv("UNSLOTH_FLASHQLA_ALLOW_SM120", raising = False)
    else: monkeypatch.setenv("UNSLOTH_FLASHQLA_ALLOW_SM120", allow)
    assert (fv._load_flashqla() is not None) == accepted


def test_flashqla_version_belongs_to_the_imported_module(tmp_path, monkeypatch):
    import importlib.metadata as md
    checkout = tmp_path / "flash_qla" / "__init__.py"
    checkout.parent.mkdir()
    checkout.write_text("")
    module = types.SimpleNamespace(__file__ = str(checkout))
    installed = tmp_path / "site" / "flash_qla" / "__init__.py"
    dist = types.SimpleNamespace(
        version = "0.1.3",
        files = ["flash_qla/__init__.py"],
        locate_file = lambda f: tmp_path / "site" / f,
    )
    monkeypatch.setattr(md, "distribution", lambda name: dist)
    # A shadowing checkout does not inherit the installed distribution's version.
    assert fv._flashqla_version(module) is None
    assert fv._flashqla_version(types.SimpleNamespace(__file__ = str(installed))) == "0.1.3"
    assert fv._flashqla_version(types.SimpleNamespace(__file__ = str(checkout), __version__ = "0.1.2")) == "0.1.2"
