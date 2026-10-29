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

"""rope_decode vs the compiled apply_rotary_pos_emb region (bitwise), its gates, and the decode wiring."""
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


def _run(tmp_path, script, *args, env_extra = None):
    path = tmp_path / "case.py"
    path.write_text(script)
    env = dict(os.environ)
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    env["PYTHONPATH"] = os.pathsep.join(x for x in (_ROOT, env.get("PYTHONPATH")) if x)
    for key, value in (env_extra or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run([sys.executable, str(path), *args], capture_output = True, text = True, env = env,
                          timeout = 1500)
    line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert proc.returncode == 0 and line, proc.stdout[-2000:] + proc.stderr[-3000:]
    return json.loads(line[-1][len("RESULT "):])


def _region(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rotate(x):
        first, second = torch.chunk(x, 2, dim = -1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim = -1)
    return rotate(q), rotate(k)


_KERNEL = textwrap.dedent('''
    import json, sys
    import torch
    from unsloth_zoo import gpt_oss_decode_fused as F
    dev = sys.argv[1]
    REGION = """__REGION__"""
    exec(REGION)
    compiled = torch.compile(_region, dynamic = True)
    # The Triton interpreter truncates fp32 -> bf16 instead of rounding: bf16 only on a GPU.
    DTYPES = ("bfloat16", "float16") if dev == "cuda" else ("float16",)
    g = torch.Generator().manual_seed(0)
    res = {}
    for dtype in DTYPES:
        dt = getattr(torch, dtype)
        for B, HQ, HK, D in ((1, 64, 8, 64), (4, 64, 8, 64), (3, 8, 2, 128), (2, 4, 4, 48)):
            # [B, 1, H, D].transpose(1, 2): the layout the attention projections hand over.
            q = (torch.randn(B, 1, HQ, D, generator = g) * 4).to(dev, dt).transpose(1, 2)
            k = (torch.randn(B, 1, HK, D, generator = g) * 4).to(dev, dt).transpose(1, 2)
            ang = torch.rand(B, 1, D // 2, generator = g) * 4096
            cos, sin = ang.cos().to(dev, dt), ang.sin().to(dev, dt)
            qo, ko = F._rope_launch(q, k, cos, sin)
            c32, s32 = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
            def fp32_once(x):
                a, b = x.float().chunk(2, -1)
                return torch.cat((a * c32 - b * s32, b * c32 + a * s32), -1).to(x.dtype)
            cq, ck = compiled(q, k, cos, sin)
            key = f"{dtype}_{B}_{HQ}_{HK}_{D}"
            res["fp32_once_" + key] = bool(torch.equal(qo, fp32_once(q)) and torch.equal(ko, fp32_once(k)))
            res["compiled_" + key] = bool(torch.equal(qo, cq) and torch.equal(ko, ck))
            res["strides_" + key] = [list(qo.stride()), list(cq.stride()), list(ko.stride()), list(ck.stride())]
    print("RESULT " + json.dumps(res))
''')


@pytest.fixture(scope = "module")
def kernel_results(tmp_path_factory):
    import inspect
    tmp = tmp_path_factory.mktemp("gpt_oss_rope_decode")
    script = _KERNEL.replace("__REGION__", textwrap.dedent(inspect.getsource(_region)))
    if torch.cuda.is_available() and torch.version.hip is None:
        return _run(tmp, script, "cuda")
    return _run(tmp, script, "cpu", env_extra = {"TRITON_INTERPRET": "1"})


def test_kernel_rounds_once_in_fp32(kernel_results):
    keys = [k for k in kernel_results if k.startswith("fp32_once_")]
    assert keys and all(kernel_results[k] for k in keys), kernel_results


def test_kernel_matches_the_compiled_region_bitwise_and_layout(kernel_results):
    keys = [k for k in kernel_results if k.startswith("compiled_")]
    assert keys and all(kernel_results[k] for k in keys), kernel_results
    for k in kernel_results:
        if k.startswith("strides_"):
            qo, cq, ko, ck = kernel_results[k]
            assert qo == cq and ko == ck, (k, kernel_results[k])


@pytest.fixture
def F(monkeypatch):
    from unsloth_zoo import gpt_oss_decode_fused as F
    monkeypatch.setattr(F, "_ENABLED", True)
    monkeypatch.setattr(F, "_ROPE_ENABLED", True)
    monkeypatch.setattr(F, "_INTERPRET", True)
    monkeypatch.setattr(F, "_ROPE_CHECKED", {})
    monkeypatch.setattr(F, "FUSED_DECODE_CALLS", dict(F.FUSED_DECODE_CALLS, rope_decode = 0))
    monkeypatch.setattr(F, "_capturing", lambda device: False)
    return F


def _inputs(B = 2, HQ = 4, HK = 2, D = 16, S = 1, dtype = torch.float16):
    q = torch.randn(B, S, HQ, D).to(dtype).transpose(1, 2)
    k = torch.randn(B, S, HK, D).to(dtype).transpose(1, 2)
    ang = torch.rand(B, S, D // 2)
    return q, k, ang.cos().to(dtype), ang.sin().to(dtype)


class _Counter:
    """A call-counting stand-in for a torch.compile'd region (rope_decode declines plain callables)."""
    def __init__(self, fn):
        self.fn, self.n = fn, 0
        self._torchdynamo_orig_callable = fn

    def __call__(self, *a):
        self.n += 1
        return self.fn(*a)


def _contiguous_region(*a):
    return tuple(t.contiguous() for t in _region(*a))


def test_first_call_checks_then_trusts(F, monkeypatch):
    launch = _Counter(_contiguous_region)
    region = _Counter(_contiguous_region)
    monkeypatch.setattr(F, "_rope_launch", launch)
    inputs = [_inputs() for _ in range(3)] + [_inputs(dtype = torch.bfloat16), _inputs(B = 3)]
    rng = torch.get_rng_state()
    for args in inputs[:3]:
        assert F.rope_decode(*args, region) is not None
    # One check on random inputs (kernel + region once each), then three real launches.
    assert launch.n == 4 and region.n == 1 and F.FUSED_DECODE_CALLS["rope_decode"] == 3
    for args in inputs[3:]:
        F.rope_decode(*args, region)
    assert region.n == 3 and len(F._ROPE_CHECKED) == 3 and set(F._ROPE_CHECKED.values()) == {True}
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("wrong", ["values", "layout"])
def test_a_mismatch_returns_the_region_and_stays_off(F, monkeypatch, wrong):
    if wrong == "values":
        bad = lambda *a: tuple(t + 1 for t in _contiguous_region(*a))  # noqa: E731
    else:
        bad = lambda *a: tuple(t.as_strided(t.shape, (t.stride(0), t.stride(1), 7, 1)) for t in _contiguous_region(*a))  # noqa: E731
    launch = _Counter(bad)
    region = _Counter(_contiguous_region)
    monkeypatch.setattr(F, "_rope_launch", launch)
    args = _inputs()
    assert F.rope_decode(*args, region) is None
    assert F.rope_decode(*args, region) is None and launch.n == 1 and region.n == 1
    assert F.FUSED_DECODE_CALLS["rope_decode"] == 0


@pytest.mark.parametrize("case", ["two_tokens", "fp32", "cos_fp32", "cos_shape", "k_batch", "odd_dim", "q_stride"])
def test_uncovered_inputs_return_none_without_launching(F, monkeypatch, case):
    launch = _Counter(_contiguous_region)
    monkeypatch.setattr(F, "_rope_launch", launch)
    q, k, cos, sin = _inputs(S = 2) if case == "two_tokens" else _inputs()
    if case == "fp32":
        q, k, cos, sin = q.float(), k.float(), cos.float(), sin.float()
    elif case == "cos_fp32":
        cos, sin = cos.float(), sin.float()
    elif case == "cos_shape":
        cos, sin = cos.expand(2, 1, 8)[:1], sin[:1]
    elif case == "k_batch":
        k = k[:1]
    elif case == "odd_dim":
        q, k = q[..., :15], k[..., :15]
    elif case == "q_stride":
        q = q.transpose(2, 3)
    assert F.rope_decode(q, k, cos, sin, _Counter(_contiguous_region)) is None and launch.n == 0


def test_kill_switch_cpu_tensors_and_capture(F, monkeypatch):
    launch = _Counter(_contiguous_region)
    region = _Counter(_contiguous_region)
    monkeypatch.setattr(F, "_rope_launch", launch)
    monkeypatch.setattr(F, "_ROPE_ENABLED", False)
    assert F.rope_decode(*_inputs(), region) is None
    monkeypatch.setattr(F, "_ROPE_ENABLED", True)
    monkeypatch.setattr(F, "_INTERPRET", False)
    assert F.rope_decode(*_inputs(), region) is None
    monkeypatch.setattr(F, "_INTERPRET", True)
    # Capture before the check: nothing launched or recorded, the region runs.
    monkeypatch.setattr(F, "_capturing", lambda device: True)
    assert F.rope_decode(*_inputs(), region) is None and not F._ROPE_CHECKED
    assert launch.n == 0 and region.n == 0


def test_eager_regions_are_never_replaced(F, monkeypatch):
    launch = _Counter(_contiguous_region)
    monkeypatch.setattr(F, "_rope_launch", launch)
    # Not torch.compile'd (UNSLOTH_COMPILE_DISABLE, or torch.compile refused to wrap it).
    assert F.rope_decode(*_inputs(), _contiguous_region) is None
    region = _Counter(_contiguous_region)
    assert F.rope_decode(*_inputs(), region) is not None
    n = launch.n
    # A compiled region that runs eagerly under another stance.
    with torch.compiler.set_stance("force_eager"):
        assert F.rope_decode(*_inputs(), region) is None
    assert launch.n == n


def test_launch_failure_disables_and_falls_back(F, monkeypatch):
    def broken(*a):
        raise RuntimeError("no compiler")
    monkeypatch.setattr(F, "_rope_launch", broken)
    assert F.rope_decode(*_inputs(), _Counter(_contiguous_region)) is None and not F._ENABLED


_CERT = textwrap.dedent('''
    import json, sys
    import torch
    from unsloth_zoo import gpt_oss_decode_fused as F
    dev = sys.argv[1]
    REGION = """__REGION__"""
    exec(REGION)
    dt = torch.bfloat16 if dev == "cuda" else torch.float16
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(2, 1, 8, 64, generator = g) * 4).to(dev, dt).transpose(1, 2)
    k = (torch.randn(2, 1, 2, 64, generator = g) * 4).to(dev, dt).transpose(1, 2)
    # Position 0: cos = 1, sin = 0, where any rounding gives the same answer.
    cos0, sin0 = torch.ones(2, 1, 32, dtype = dt, device = dev), torch.zeros(2, 1, 32, dtype = dt, device = dev)
    eager = lambda *a: tuple(t.contiguous() for t in _region(*a))
    eager._torchdynamo_orig_callable = _region  # a compiled region that runs eagerly (every op rounded)
    res = {"position0_agrees": all(bool(torch.equal(a, b)) for a, b in zip(F._rope_launch(q, k, cos0, sin0), eager(q, k, cos0, sin0)))}
    res["eager_region_declined"] = F.rope_decode(q, k, cos0, sin0, eager) is None
    ang = torch.rand(2, 1, 32, generator = g) * 4096
    cos, sin = ang.cos().to(dev, dt), ang.sin().to(dev, dt)
    res["eager_differs_later"] = not all(bool(torch.equal(a, b)) for a, b in zip(F._rope_launch(q, k, cos, sin), eager(q, k, cos, sin)))
    compiled = torch.compile(_region, dynamic = True)
    out = F.rope_decode(q, k, cos0, sin0, compiled)
    out2 = F.rope_decode(q, k, cos, sin, compiled)
    ref2 = compiled(q, k, cos, sin)
    res["compiled_accepted"] = out is not None and out2 is not None
    res["compiled_equal"] = out2 is not None and all(bool(torch.equal(a, b)) for a, b in zip(out2, ref2))
    print("RESULT " + json.dumps(res))
''')


def test_first_token_cannot_certify_an_eager_region(tmp_path):
    import inspect
    script = _CERT.replace("__REGION__", textwrap.dedent(inspect.getsource(_region)))
    if torch.cuda.is_available() and torch.version.hip is None:
        res = _run(tmp_path, script, "cuda")
    else:
        res = _run(tmp_path, script, "cpu", env_extra = {"TRITON_INTERPRET": "1"})
    assert res["position0_agrees"] and res["eager_differs_later"], res  # the premise
    assert res["eager_region_declined"] and res["compiled_accepted"] and res["compiled_equal"], res


_LOOP = textwrap.dedent('''
    import json, os, sys
    os.environ.setdefault("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-test")
    import torch
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    import transformers.models.gpt_oss.modeling_gpt_oss as M
    from unsloth_zoo.temporary_patches import gpt_oss as G
    import unsloth_zoo.flex_attention as FA
    # Without a GPU flex_attention exports no sink kernels and the attention patch would not install; eval
    # with a dense mask (CPU) runs the eager attention, so stand-ins that are never called are enough.
    def _unused(*a, **k):
        raise AssertionError("flex attention is not used by CPU eval")
    for name in ("flex_attention_with_sink", "flex_attention_with_sink_inference", "flex_attention_add_sinks"):
        if getattr(FA, name, None) is None:
            setattr(FA, name, _unused)
    for name in ("is_flex_attention_decoding", "flex_attention_with_sink_decoding"):
        if not hasattr(FA, name):
            setattr(FA, name, None)
    calls = {"rope": 0, "shapes": set()}
    use = sys.argv[1] == "on"
    def stand_in(q, k, cos, sin, region):
        calls["rope"] += 1
        calls["shapes"].add(q.shape[2])
        return region(q, k, cos, sin)
    G._rope_decode = stand_in
    G._fused_decode_ok = lambda x: use and not torch.is_grad_enabled()
    G._fused_add_rms_norm = lambda *a: None
    G.patch_GptOssAttention(); G.patch_GptOssModel()
    L = 2
    torch.manual_seed(0)
    c = GptOssConfig(hidden_size = 64, intermediate_size = 64, num_hidden_layers = L, num_attention_heads = 4,
                     num_key_value_heads = 2, head_dim = 16, num_local_experts = 4, num_experts_per_tok = 2,
                     vocab_size = 128, max_position_embeddings = 256, sliding_window = 6,
                     layer_types = ["sliding_attention", "full_attention"])
    c._attn_implementation = "eager"
    model = M.GptOssForCausalLM(c).eval()
    ids = torch.randint(1, 128, (2, 7), generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = model.generate(ids, attention_mask = torch.ones_like(ids), max_new_tokens = 6, do_sample = False,
                             pad_token_id = 0)
    print("RESULT " + json.dumps({"tokens": out.tolist(), "rope": calls["rope"], "q_lens": sorted(calls["shapes"]),
                                  "patched": M.GptOssAttention.forward.__module__}))
''')


def test_decode_loop_calls_rope_decode(tmp_path):
    env = {"UNSLOTH_COMPILE_DISABLE": None, "UNSLOTH_ENABLE_FLEX_ATTENTION": None}
    on = _run(tmp_path, _LOOP, "on", env_extra = env)
    off = _run(tmp_path, _LOOP, "off", env_extra = env)
    assert on["rope"] >= 6 * 2 and off["rope"] == 0, (on, off)
    assert 1 in on["q_lens"] and on["tokens"] == off["tokens"]
