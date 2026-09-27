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

"""DeepSeek-V4's rope must stay eager in the compiled cache: Inductor with dynamic = True
reads out of bounds in its backward (pytorch#198553), so LoRA training went NaN."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")


_CHILD = r'''
import os, sys, json, io, contextlib, importlib.util
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
import torch
from unsloth_zoo.compiler import unsloth_compile_transformers
import transformers.models.deepseek_v4.modeling_deepseek_v4 as modeling

# Taken before compiling: the compiler swaps its emitted functions back into `modeling`.
eager_rope = modeling.apply_rotary_pos_emb
out = {}
for mt in ("deepseek_v4", "llama"):
    with contextlib.redirect_stdout(io.StringIO()):
        unsloth_compile_transformers(
            model_type = mt, fast_lora_forwards = False, fullgraph = True,
            import_from_cache = False, disable = False, supports_sdpa = [None],
        )
    path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], f"unsloth_compiled_module_{mt}.py")
    generated = open(path, encoding = "utf-8").read()
    where = generated.find("def apply_rotary_pos_emb(")
    out[mt] = generated[:where].rsplit("\n@", 1)[-1].strip() if where != -1 else None
    out[mt + "_path"] = path

if torch.cuda.is_available():
    spec = importlib.util.spec_from_file_location("dsv4_cache", out["deepseek_v4_path"])
    cache = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache)
    torch.manual_seed(0)
    # Strided like the attention call: q is (B, S, H, D) transposed, cos / sin half sized.
    x = torch.randn(2, 128, 4, 512, device = "cuda").transpose(1, 2)
    cos = torch.randn(1, 32, 128, device = "cuda").transpose(1, 2)
    sin = torch.randn(1, 32, 128, device = "cuda").transpose(1, 2)
    grad_out = torch.randn(2, 4, 128, 512, device = "cuda")
    errors = []
    for _ in range(3):
        grads = []
        for fn in (eager_rope, cache.apply_rotary_pos_emb):
            leaf = x.detach().clone(memory_format = torch.preserve_format).requires_grad_()
            (grad,) = torch.autograd.grad(fn(leaf, cos, sin), leaf, grad_out)
            grads.append(grad)
        errors.append(((grads[1] - grads[0]).norm() / grads[0].norm()).item())
    out["grad_rel_err"] = max(errors)
print("@@@" + json.dumps(out))
'''


def _run_child(tmp_path):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], cwd = str(tmp_path), env = env,
        capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    for line in result.stdout.splitlines():
        if line.startswith("@@@"):
            return json.loads(line[3:])
    raise AssertionError(result.stdout[-4000:])


@pytest.fixture(scope = "module")
def child(tmp_path_factory):
    return _run_child(tmp_path_factory.mktemp("dsv4_rope"))


def test_deepseek_v4_rope_is_left_eager(child):
    assert child["deepseek_v4"] == "torch.compiler.disable(recursive = False)", child


def test_other_models_keep_their_rope_decorator(child):
    # Scoped to deepseek_v4: the same name is a different function elsewhere.
    assert child["llama"] is not None and "compiler.disable" not in child["llama"], child


def test_cached_rope_gradient_matches_transformers(child):
    if "grad_rel_err" not in child:
        pytest.skip("needs CUDA: the out-of-bounds read is in the Triton backward kernel")
    assert child["grad_rel_err"] < 1e-5, child
