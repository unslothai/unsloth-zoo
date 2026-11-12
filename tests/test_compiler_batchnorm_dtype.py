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

"""Compiled torch.nn.BatchNorm accepts an input whose dtype differs from its weight (float32 into bfloat16 under QLoRA);
runs in a fresh interpreter since the compiler mutates torch.nn."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("unsloth_zoo.compiler")

ROOT = Path(__file__).resolve().parents[1]


def test_compiled_batchnorm_casts_mismatched_input(tmp_path):
    script = textwrap.dedent(
        f"""
        import importlib, json, sys, torch
        sys.path.insert(0, {str(ROOT)!r})
        import unsloth_zoo.compiler as compiler
        importlib.import_module("transformers.models.llama.modeling_llama")
        compiler.unsloth_compile_transformers("llama", disable=False)
        out = {{}}
        for name in ("BatchNorm1d", "LayerNorm"):
            out[name + "_patched"] = bool(getattr(getattr(torch.nn, name).forward, "__unsloth_dtype_wrapped__", False))
        bn = torch.nn.BatchNorm1d(4).to(torch.bfloat16).train()
        x = torch.randn(2, 4, 3)
        try:
            y = bn(x)
            out["dtype"] = str(y.dtype)
        except RuntimeError as e:
            out["error"] = str(e)[:200]
        same = torch.nn.BatchNorm1d(4).to(torch.bfloat16).eval()
        xb = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        ref = torch.nn.functional.batch_norm(xb, same.running_mean, same.running_var, same.weight, same.bias, False, 0.1, same.eps)
        out["same_dtype_identical"] = bool(torch.equal(same(xb), ref))
        print("RESULT " + json.dumps(out))
        """
    )
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION=str(tmp_path / "cache"))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    out = json.loads(line[-1][len("RESULT "):])
    assert out["BatchNorm1d_patched"], out
    assert "error" not in out, out
    assert out["dtype"] == "torch.float32", out
    assert out["same_dtype_identical"], out


def test_compiled_batchnorm_keeps_native_mixed_precision(tmp_path):
    script = textwrap.dedent(
        f"""
        import importlib, json, sys, torch
        sys.path.insert(0, {str(ROOT)!r})
        import unsloth_zoo.compiler as compiler
        importlib.import_module("transformers.models.llama.modeling_llama")
        compiler.unsloth_compile_transformers("llama", disable=False)
        out = {{"patched": bool(getattr(torch.nn.BatchNorm1d.forward, "__unsloth_dtype_wrapped__", False))}}
        for dtype in (torch.bfloat16, torch.float16):
            for training in (True, False):
                torch.manual_seed(0)
                bn = torch.nn.BatchNorm1d(8).train(training).requires_grad_(False)
                ref = torch.nn.BatchNorm1d(8).train(training).requires_grad_(False)
                ref.load_state_dict(bn.state_dict())
                x = torch.randn(4, 8, 16, dtype=dtype, requires_grad=True)
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, tuple(t.shape))) or t, lambda t: t):
                    y = bn(x)
                y.float().square().sum().backward()
                xr = x.detach().clone().requires_grad_(True)
                yr = torch.nn.functional.batch_norm(xr, ref.running_mean, ref.running_var, ref.weight, ref.bias,
                                                    training, ref.momentum, ref.eps)
                yr.float().square().sum().backward()
                key = str(dtype) + ("_train" if training else "_eval")
                out[key] = dict(
                    out_dtype=str(y.dtype),
                    fp32_input_copies=sum(1 for d, s in saved if d == torch.float32 and s == tuple(x.shape)),
                    identical=bool(torch.equal(y, yr) and torch.equal(x.grad, xr.grad)
                                   and bn.weight.grad is None and ref.weight.grad is None
                                   and torch.equal(bn.running_mean, ref.running_mean)),
                )
        print("RESULT " + json.dumps(out))
        """
    )
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION=str(tmp_path / "cache"))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    out = json.loads(line[-1][len("RESULT "):])
    assert out.pop("patched"), out
    for key, row in out.items():
        assert row["out_dtype"] == key.split("_")[0], (key, row)
        assert row["fp32_input_copies"] == 0, (key, row)
        assert row["identical"], (key, row)


def test_compiled_batchnorm_trainable_params_keep_float32_cast(tmp_path):
    script = textwrap.dedent(
        f"""
        import importlib, json, sys, torch
        sys.path.insert(0, {str(ROOT)!r})
        import unsloth_zoo.compiler as compiler
        importlib.import_module("transformers.models.llama.modeling_llama")
        compiler.unsloth_compile_transformers("llama", disable=False)
        out = {{}}
        for dtype in (torch.bfloat16, torch.float16):
            for trainable in ("weight", "bias"):
                torch.manual_seed(0)
                bn = torch.nn.BatchNorm1d(8).train().requires_grad_(False)
                getattr(bn, trainable).requires_grad_(True)
                ref = torch.nn.BatchNorm1d(8).train().requires_grad_(False)
                getattr(ref, trainable).requires_grad_(True)
                ref.load_state_dict(bn.state_dict())
                x = torch.randn(4, 8, 16, dtype=dtype, requires_grad=True)
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, tuple(t.shape))) or t, lambda t: t):
                    y = bn(x)
                y.float().square().sum().backward()
                xr = x.detach().clone().requires_grad_(True)
                yr = torch.nn.functional.batch_norm(xr.float(), ref.running_mean, ref.running_var, ref.weight, ref.bias,
                                                    True, ref.momentum, ref.eps).to(dtype)
                yr.float().square().sum().backward()
                out[str(dtype) + "_" + trainable] = dict(
                    out_dtype=str(y.dtype),
                    fp32_input_copies=sum(1 for d, s in saved if d == torch.float32 and s == tuple(x.shape)),
                    identical=bool(torch.equal(y, yr) and torch.equal(x.grad, xr.grad)
                                   and torch.equal(getattr(bn, trainable).grad, getattr(ref, trainable).grad)),
                )
        print("RESULT " + json.dumps(out))
        """
    )
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION=str(tmp_path / "cache"))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    out = json.loads(line[-1][len("RESULT "):])
    for key, row in out.items():
        assert row["out_dtype"] == key.split("_")[0], (key, row)
        assert row["fp32_input_copies"] == 1, (key, row)
        assert row["identical"], (key, row)


def test_batchnorm_cast_survives_compile_mode_changes(tmp_path):
    # disable=True between disable=False loads rebuilds via the source-free wrapper; it must cast the same.
    script = textwrap.dedent(
        f"""
        import importlib, json, sys, torch
        sys.path.insert(0, {str(ROOT)!r})
        import unsloth_zoo.compiler as compiler
        rows = []
        for model_type, disable in (("llama", False), ("bert", True), ("qwen2", False)):
            importlib.import_module(f"transformers.models.{{model_type}}.modeling_{{model_type}}")
            compiler.unsloth_compile_transformers(model_type, disable=disable)
            row = {{"step": model_type}}
            for name, wd, xd, frozen in (("canary", torch.bfloat16, torch.float32, False),
                                         ("trainable", torch.float32, torch.bfloat16, False),
                                         ("frozen", torch.float32, torch.bfloat16, True)):
                bn = torch.nn.BatchNorm1d(4).to(wd).train()
                if frozen:
                    bn.requires_grad_(False)
                x = torch.randn(2, 4, 3, dtype=xd, requires_grad=True)
                saved = []
                try:
                    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append((t.dtype, tuple(t.shape))) or t, lambda t: t):
                        y = bn(x)
                    y.float().sum().backward()
                    row[name] = dict(out=str(y.dtype), copies=sum(1 for d, s in saved if d == torch.float32 and s == tuple(x.shape)))
                except RuntimeError as e:
                    row[name] = dict(error=str(e)[:200])
            rows.append(row)
        print("RESULT " + json.dumps(rows))
        """
    )
    env = dict(os.environ, UNSLOTH_COMPILE_LOCATION=str(tmp_path / "cache"))
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env.setdefault("UNSLOTH_ALLOW_CPU", "1")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900, env=env)
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, proc.stdout[-2000:] + proc.stderr[-3000:]
    rows = json.loads(line[-1][len("RESULT "):])
    assert [row["step"] for row in rows] == ["llama", "bert", "qwen2"], rows
    for row in rows:
        assert row["canary"].get("out") == "torch.float32", row
        assert row["trainable"] == {"out": "torch.bfloat16", "copies": 1}, row
        assert row["frozen"] == {"out": "torch.bfloat16", "copies": 1 if row["step"] == "bert" else 0}, row
