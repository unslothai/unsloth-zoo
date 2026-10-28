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

# gpt-oss decode: fused residual add + RMSNorm and fused sink epilogue vs the eager op chains.
# Kernels run on CUDA when present, else under the Triton interpreter on CPU (subprocess).
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytest.importorskip("triton")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SCRIPT = textwrap.dedent('''
    import json, os, sys
    import torch
    from unsloth_zoo import gpt_oss_decode_fused as F
    dev = sys.argv[1]
    res = {}
    # The Triton interpreter truncates fp32 -> bf16 instead of rounding: bf16 only on a GPU.
    DTYPES = ("bfloat16", "float16", "float32") if dev == "cuda" else ("float16", "float32")

    def eager_rms(w, eps, x):
        # temporary_patches/gpt_oss.py rms_layernorm_forward
        h = x.to(torch.float32)
        var = h.square().mean(-1, keepdim = True) + eps
        h = h * torch.rsqrt(var)
        h = h * w.to(torch.float32)
        return h.to(x.dtype)

    def ulps(a, b):
        # Distance in units of the dtype's last place (bf16 / fp16 / fp32).
        eps = torch.finfo(a.dtype).eps
        a, b = a.float(), b.float()
        return float(((a - b).abs() / (b.abs().clamp_min(2.0 ** -10) * eps)).max())

    g = torch.Generator().manual_seed(0)
    for dtype in DTYPES:
        dt = getattr(torch, dtype)
        for N, rows in ((2880, 1), (2880, 4), (100, 3)):
            for w_dtype in (dt, torch.float32):
                x = (torch.randn(rows, 1, N, generator = g) * 3).to(dev, dt)
                r = (torch.randn(rows, 1, N, generator = g) * 3).to(dev, dt)
                w = (torch.rand(N, generator = g) + 0.5).to(dev, w_dtype)
                ref_sum = x + r
                ref = eager_rms(w, 1e-5, ref_sum)
                xx = x.clone()
                y, ss = F.add_rms_norm(xx, r, w, 1e-5)
                key = f"{dtype}_{N}_{rows}_{w_dtype}"
                res["sum_exact_" + key] = bool(torch.equal(ss, ref_sum))
                res["no_res_untouched_x_" + key] = bool(torch.equal(xx, x))
                res["norm_ulps_" + key] = ulps(y, ref) if dtype != "float32" else float((y - ref).abs().max())
                res["norm_mismatch_frac_" + key] = float((y != ref).float().mean())
                x0 = x.clone()
                y0, _ = F.add_rms_norm(x0, None, w, 1e-5)
                res["no_res_untouched_" + key] = bool(torch.equal(x0, x))
                res["no_res_ulps_" + key] = float((y0.float() - eager_rms(w, 1e-5, x).float()).abs().max())

    # Unsupported inputs fall back (None), nothing written.
    x = torch.randn(2, 8, device = dev)
    res["none_noncontig"] = F.add_rms_norm(x.t(), None, torch.ones(2, device = dev), 1e-5) is None
    res["none_bad_weight"] = F.add_rms_norm(x, None, torch.ones(4, device = dev), 1e-5) is None

    for dtype in DTYPES[:-1]:
        dt = getattr(torch, dtype)
        for B, H, Q, D in ((1, 64, 1, 64), (3, 8, 5, 64), (2, 4, 1, 48)):
            # flex returns [B, H, Q, D]; also a transposed (non-contiguous) layout.
            for layout in ("contig", "strided"):
                if layout == "contig":
                    a = torch.randn(B, H, Q, D, generator = g).to(dev, dt)
                else:
                    a = torch.randn(B, Q, H, D, generator = g).to(dev, dt).transpose(1, 2)
                lse = (torch.randn(B, H, Q, generator = g) * 4).to(dev)
                lse[0, 0, 0] = float("-inf")  # no visible key
                sinks = torch.randn(H, generator = g).to(dev, dt)
                scale = torch.sigmoid(lse.float() - sinks.float().unsqueeze(1))
                ref = (a * scale.unsqueeze(-1).to(a.dtype)).transpose(1, 2).contiguous()
                out = F.sink_scale_output(a, lse, sinks)
                key = f"{dtype}_{B}_{H}_{Q}_{D}_{layout}"
                res["sink_shape_" + key] = list(out.shape) == list(ref.shape) and out.is_contiguous()
                res["sink_ulps_" + key] = ulps(out, ref)
                res["sink_mismatch_frac_" + key] = float((out != ref).float().mean())
                res["sink_empty_row_zero_" + key] = bool((out[0, 0, 0] == 0).all())
    res["calls"] = dict(F.FUSED_DECODE_CALLS)
    print("RESULT " + json.dumps(res))
''')


def _run(tmp_path, script, *args, env_extra = None):
    path = tmp_path / "case.py"
    path.write_text(script)
    env = dict(os.environ)
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    env["PYTHONPATH"] = os.pathsep.join(x for x in (_ROOT, env.get("PYTHONPATH")) if x)
    for key, value in (env_extra or {}).items():
        # None unsets: CI exports switches (UNSLOTH_COMPILE_DISABLE=1) that skip the model patch under test.
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run([sys.executable, str(path), *args], capture_output = True, text = True, env = env,
                          timeout = 1500)
    line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert proc.returncode == 0 and line, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(line[-1][len("RESULT "):])


@pytest.fixture(scope = "module")
def kernels(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpt_oss_decode_fused")
    if torch.cuda.is_available():
        return _run(tmp, _SCRIPT, "cuda")
    return _run(tmp, _SCRIPT, "cpu", env_extra = {"TRITON_INTERPRET": "1"})


def test_residual_sum_is_bit_exact(kernels):
    keys = [k for k in kernels if k.startswith("sum_exact_")]
    assert keys and all(kernels[k] for k in keys), {k: kernels[k] for k in keys}


def test_norm_within_one_ulp_of_eager(kernels):
    for k, v in kernels.items():
        if k.startswith("norm_ulps_"):
            # bf16 / fp16: at most one ulp from the fp32 reduction order; fp32: absolute.
            limit = 1e-5 if "float32_" in k[len("norm_ulps_"):][:8] else 1.01
            assert v <= limit, (k, v)
        if k.startswith("norm_mismatch_frac_") and not k.startswith("norm_mismatch_frac_float32"):
            assert v <= 0.02, (k, v)


def test_norm_without_residual_leaves_input(kernels):
    for k, v in kernels.items():
        if k.startswith("no_res_untouched_"):
            assert v, k
        if k.startswith("no_res_ulps_"):
            assert v <= 0.02 * 8, (k, v)


def test_unsupported_inputs_fall_back(kernels):
    assert kernels["none_noncontig"] and kernels["none_bad_weight"]


def test_sink_epilogue_matches_eager(kernels):
    keys = [k[len("sink_ulps_"):] for k in kernels if k.startswith("sink_ulps_")]
    assert keys
    for key in keys:
        assert kernels["sink_shape_" + key], key
        if torch.cuda.is_available() and torch.version.cuda:
            # libdevice exp + an IEEE division: the same scale torch.sigmoid computes, bit for bit.
            assert kernels["sink_mismatch_frac_" + key] == 0, (key, kernels["sink_mismatch_frac_" + key])
        assert kernels["sink_ulps_" + key] <= 1.01, (key, kernels["sink_ulps_" + key])
        assert kernels["sink_mismatch_frac_" + key] <= 0.02, (key, kernels["sink_mismatch_frac_" + key])
        assert kernels["sink_empty_row_zero_" + key], key


def test_gates_cpu(monkeypatch):
    from unsloth_zoo import gpt_oss_decode_fused as F
    monkeypatch.setattr(F, "_ENABLED", True)
    assert not F.fused_decode_ok(torch.zeros(2))  # CPU tensor
    if torch.cuda.is_available():
        x = torch.zeros(2, device = "cuda")
        assert F.fused_decode_ok(x) == (not torch.is_grad_enabled())
        with torch.no_grad():
            assert F.fused_decode_ok(x)
        monkeypatch.setattr(F, "_ENABLED", False)
        with torch.no_grad():
            assert not F.fused_decode_ok(x)


# Decode loop wiring: the gpt-oss patched model, fused norms forced on (CPU: Triton interpreter) vs the
# kill switch. Same tokens and near-equal logits; the residual stream is never mutated behind the caller.
_LOOP = textwrap.dedent('''
    import json, os, sys
    os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
    import torch
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    import transformers.models.gpt_oss.modeling_gpt_oss as M
    dev, dtype = sys.argv[1], getattr(torch, sys.argv[2])
    from unsloth_zoo.temporary_patches import gpt_oss as G
    from unsloth_zoo import gpt_oss_decode_fused as F
    if dev == "cpu":
        # The fused path is CUDA only. On CPU the loop wiring runs against a torch stand-in with the
        # kernel's contract (sum written into x, normed output returned); the kernel itself is
        # checked against the eager ops above.
        def _stand_in(x, residual, weight, eps):
            if residual is not None:
                x = x + residual
            h = x.float()
            h = h * torch.rsqrt(h.square().mean(-1, keepdim = True) + eps) * weight.float()
            F.FUSED_DECODE_CALLS["add_rms_norm"] += 1
            return h.to(x.dtype), x
        G._fused_add_rms_norm = _stand_in
        G._fused_decode_ok = lambda x: F._ENABLED and not torch.is_grad_enabled()
        # Without a GPU flex_attention reports unavailable and exports no decode helpers; the model
        # patch only imports them (eval on CPU runs eager attention).
        import unsloth_zoo.flex_attention as FA
        for name in ("is_flex_attention_decoding", "flex_attention_with_sink_decoding"):
            if not hasattr(FA, name):
                setattr(FA, name, None)
    G.patch_GptOssAttention(); G.patch_GptOssModel()
    L = 4
    torch.manual_seed(0)
    c = GptOssConfig(hidden_size = 128, intermediate_size = 128, num_hidden_layers = L, num_attention_heads = 8,
                     num_key_value_heads = 2, head_dim = 32, num_local_experts = 4, num_experts_per_tok = 2,
                     vocab_size = 256, max_position_embeddings = 512, sliding_window = 6,
                     layer_types = ["sliding_attention", "full_attention"] * (L // 2))
    c._attn_implementation = "eager"
    model = M.GptOssForCausalLM(c)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.normal_(0, 0.5 if "sinks" in n else 0.06)
            if "norm" in n:
                p.add_(1.0)
    model = model.to(dev, dtype).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, 256, (2, 9), generator = g).to(dev)
    with torch.no_grad():
        out = model.generate(ids, attention_mask = torch.ones_like(ids), max_new_tokens = 24, do_sample = False,
                             pad_token_id = 0, output_scores = True, return_dict_in_generate = True)
    res = {"tokens": out.sequences.tolist(), "scores": torch.stack(out.scores, 1).float().tolist(),
           "calls": dict(F.FUSED_DECODE_CALLS)}
    print("RESULT " + json.dumps(res))
''')


@pytest.fixture(scope = "module")
def loop_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpt_oss_decode_loop")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if dev == "cuda" else "float32"
    return {
        arm: _run(
            tmp, _LOOP, dev, dtype,
            env_extra = {
                "UNSLOTH_GPT_OSS_FUSED_DECODE": flag,
                # patch_GptOssModel returns early under either, so neither arm would reach the decode loop.
                "UNSLOTH_COMPILE_DISABLE": None,
                "UNSLOTH_ENABLE_FLEX_ATTENTION": None,
            },
        )
        for arm, flag in (("fused", "1"), ("eager", "0"))
    }


def test_decode_loop_engages_fused_norms_and_kill_switch(loop_runs):
    L = 4
    calls = loop_runs["fused"]["calls"]["add_rms_norm"]
    # Two per layer plus the final norm, per decode step (23 of them).
    assert calls >= 23 * (2 * L + 1), calls
    assert loop_runs["eager"]["calls"]["add_rms_norm"] == 0


def test_decode_loop_tokens_and_scores_match_eager(loop_runs):
    fused, eager = loop_runs["fused"], loop_runs["eager"]
    assert fused["tokens"] == eager["tokens"]
    diff = (torch.tensor(fused["scores"]) - torch.tensor(eager["scores"])).abs().max()
    tol = 1e-4 if not torch.cuda.is_available() else 0.06
    assert diff <= tol, diff


def test_launch_failure_falls_back_and_disables(monkeypatch):
    # A Triton build that cannot compile the kernels must not break decoding: None (the caller runs the
    # eager ops on untouched inputs) and the fused path switches off.
    from unsloth_zoo import gpt_oss_decode_fused as F

    class _Broken:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                raise RuntimeError("no compiler")
            return launch

    monkeypatch.setattr(F, "_ENABLED", True)
    monkeypatch.setattr(F, "_add_rms_norm_kernel", _Broken())
    monkeypatch.setattr(F, "_sink_scale_kernel", _Broken())
    x, r = torch.randn(2, 8), torch.randn(2, 8)
    x0 = x.clone()
    assert F.add_rms_norm(x, r, torch.ones(8), 1e-5) is None
    assert torch.equal(x, x0)
    assert not F._ENABLED
    monkeypatch.setattr(F, "_ENABLED", True)
    assert F.sink_scale_output(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1), torch.randn(2)) is None
    assert not F._ENABLED


def test_disabled_or_tracing_never_launches(monkeypatch):
    from unsloth_zoo import gpt_oss_decode_fused as F
    launches = []

    class _Counting:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(grid)
                raise RuntimeError("no compiler")
            return launch

    monkeypatch.setattr(F, "_add_rms_norm_kernel", _Counting())
    monkeypatch.setattr(F, "_sink_scale_kernel", _Counting())
    monkeypatch.setattr(F, "_ENABLED", True)
    x, r, w = torch.randn(2, 8), torch.randn(2, 8), torch.ones(8)
    # One failed launch disables the path: later calls in the same step go straight to the eager ops.
    assert F.add_rms_norm(x, r, w, 1e-5) is None
    assert F.add_rms_norm(x, r, w, 1e-5) is None
    assert F.sink_scale_output(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1), torch.randn(2)) is None
    assert len(launches) == 1
    monkeypatch.setattr(F, "_ENABLED", True)
    monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    assert F.add_rms_norm(x, r, w, 1e-5) is None
    assert F.sink_scale_output(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1), torch.randn(2)) is None
    assert len(launches) == 1 and F._ENABLED
