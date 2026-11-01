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

"""Checks that UNSLOTH_COMPILE_DISABLE set to '1' or 'partial' only switches off torch.compile: rewritten classes, fast
LoRA forward and SDPA mask fix stay on, eager, with no Dynamo frame traced; probed in a subprocess since the flag is
read at import"""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("peft")
pytest.importorskip("transformers")

_CHILD = r'''
import os, io, json, contextlib, re
os.environ["UNSLOTH_COMPILE_LOCATION"] = os.path.abspath("unsloth_compiled_cache")
import torch
import torch._dynamo.utils as dynamo_utils
from unsloth_zoo.compiler import unsloth_compile_transformers
from transformers.models.llama import modeling_llama
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = "llama", fast_lora_forwards = True, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [True],
    )
generated = open(os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], "unsloth_compiled_module_llama.py")).read()
import peft.tuners.lora.layer as lora_layer
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
from unsloth_zoo.temporary_patches.misc import patch_sdpa_bool_causal_mask
import transformers.integrations.sdpa_attention as sdpa_mod
patch_sdpa_bool_causal_mask()
from unsloth_zoo.temporary_patches.misc import patch_transformers_masks
import transformers.masking_utils as masking_utils
patch_transformers_masks()
# The wrapped mask builder must not be torch.compiler.disable'd: a fullgraph user compile would refuse it.
_mask_cells = [c.cell_contents for c in (masking_utils.create_causal_mask.__closure__ or ()) if callable(c.cell_contents)]
def plain(x): return x + 1
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_fused_ce_loss
torch.manual_seed(0)
hidden = torch.randn(2, 16, 32, requires_grad = True)
weight = torch.randn(64, 32, requires_grad = True)
labels = torch.randint(0, 64, (2, 16))
# target_gb takes the chunked torch.func route, the one torch_compile applies to.
loss = unsloth_fused_ce_loss(None, hidden, weight, None, labels, target_gb = 1)
loss.backward()
print("RESULT " + json.dumps({
    "mlp_module"      : modeling_llama.LlamaMLP.__module__,
    "lora_forward"    : lora_layer.Linear.forward.__name__,
    # Model functions only: the shared prepend keeps its decorator, a no-op under the flag.
    "compile_decos"   : len(re.findall(r"@torch_compile_with_fallback\([^\n]*\)\n(?:@[^\n]*\n)*def Llama", generated)),
    "fallback_is_id"  : torch_compile_with_fallback(fullgraph = True, dynamic = True)(plain) is plain,
    "sdpa_fix"        : bool(getattr(sdpa_mod.sdpa_attention_forward, "__unsloth_bool_causal_mask_fix__", False)),
    "mask_disabled"   : any(bool(getattr(c, "_torchdynamo_disable", False)) for c in _mask_cells),
    "loss"            : float(loss),
    "frames"          : int(dynamo_utils.counters["frames"]["total"]),
}))
'''


def _probe(tmp_path, value):
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    if value is not None:
        env["UNSLOTH_COMPILE_DISABLE"] = value
    proc = subprocess.run([sys.executable, "-c", _CHILD], cwd = tmp_path, capture_output = True, text = True,
                          timeout = 900, env = env)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(line[len("RESULT "):])


@pytest.mark.parametrize("value", [None, "1", "partial"], ids = ["unset", "1", "partial"])
def test_compile_disable_keeps_optimizations(tmp_path, value):
    r = _probe(tmp_path, value)
    assert r["mlp_module"].startswith("unsloth_compiled_module_llama"), r
    assert r["lora_forward"] == "unsloth_forward", r
    assert r["sdpa_fix"] is True, r
    assert r["mask_disabled"] is False, r
    if value is None:
        assert r["compile_decos"] > 0 and r["fallback_is_id"] is False, r
        return
    assert r["compile_decos"] == 0, r
    assert r["fallback_is_id"] is True, r
    assert r["frames"] == 0, r


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
