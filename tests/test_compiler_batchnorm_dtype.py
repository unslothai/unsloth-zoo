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

"""The compiled torch.nn.BatchNorm forward Unsloth installs accepts an input whose dtype differs from
its weight (Canary's conformer under QLoRA feeds float32 into a bfloat16 BatchNorm1d).

Runs in a fresh interpreter: the compiler mutates torch.nn process-wide."""

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
    # bfloat16 / float16 activations against FROZEN float32 BatchNorm parameters (AMP over float32 weights,
    # LoRA) are native mixed precision BatchNorm: no float32 copy of the input may be saved for backward.
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
    # Trainable float32 BatchNorm parameters keep the input cast: on CUDA the native mixed kernel returns
    # weight / bias gradients at input-dtype precision, so skipping it would cost full fine-tuning accuracy.
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
