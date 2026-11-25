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

"""The installed fused forward replaces the class forward of every transformers *ForCausalLM, so a model
loaded with plain transformers in the same process ran it too and returned empty logits with labels.
Once require_model_marker() is on, only models passed to mark_model fuse; the rest return the logits
and loss of an unpatched forward. Without it every model fuses, as before."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = textwrap.dedent(
    """
    import json, os, sys
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    def build():
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                             num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 64)
        return LlamaForCausalLM(config).eval()

    ids = torch.randint(0, 128, (2, 12), generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = build()(input_ids = ids, labels = ids)

    from unsloth_zoo.fused_losses import forward_install as fi
    assert fi.install_for_class(LlamaForCausalLM), fi.audit()
    if sys.argv[1] == "marker":
        fi.require_model_marker()
    plain, marked = build(), build()
    fi.mark_model(marked)
    out = {}
    with torch.no_grad():
        for name, model in (("plain", plain), ("marked", marked)):
            o = model(input_ids = ids, labels = ids)
            out[name] = dict(
                numel = o.logits.numel(),
                logits_equal = o.logits.numel() > 0 and bool(torch.equal(o.logits, ref.logits)),
                loss_diff = abs(o.loss.item() - ref.loss.item()),
            )
    print("RESULT " + json.dumps(out))
    """
)


def _run(arm):
    env = dict(os.environ, PYTHONPATH = ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""), CUDA_VISIBLE_DEVICES = "")
    env.pop("UNSLOTH_RETURN_LOGITS", None)
    env.pop("UNSLOTH_FUSED_FORWARD", None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, arm], capture_output = True, text = True, env = env, cwd = ROOT, timeout = 600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        if "install_for_class" in proc.stderr and "AssertionError" in proc.stderr:
            pytest.skip("this transformers LlamaForCausalLM forward is not rewritten by the installer")
        raise AssertionError(proc.stdout[-2000:] + proc.stderr[-4000:])
    return json.loads(lines[-1][len("RESULT "):])


def test_marker_scopes_fused_branch_to_marked_models():
    out = _run("marker")
    # A plain model keeps its logits, bitwise those of the unpatched forward.
    assert out["plain"]["numel"] > 0
    assert out["plain"]["logits_equal"]
    assert out["plain"]["loss_diff"] == 0.0
    # A marked model still fuses: no logits, same loss to fp32 rounding.
    assert out["marked"]["numel"] == 0
    assert out["marked"]["loss_diff"] < 1e-4


def test_without_marker_requirement_every_model_fuses():
    out = _run("legacy")
    assert out["plain"]["numel"] == 0
    assert out["marked"]["numel"] == 0
    assert out["plain"]["loss_diff"] < 1e-4
