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

"""A compiled eager-attention decoder (T5, Bloom, GPT-J) must stay causal: the causal-mask removal is SDPA only."""

import json
import os
import subprocess
import sys

import pytest

_CHILD = r'''
import contextlib, copy, importlib, io, json, sys
import torch
from transformers import AutoConfig
from unsloth_zoo.compiler import unsloth_compile_transformers

model_type, class_name = sys.argv[1], sys.argv[2]
modeling = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
if not any(hasattr(c, "_update_causal_mask") for c in vars(modeling).values() if isinstance(c, type)):
    print("@@@" + json.dumps({"skip": "no _update_causal_mask hook on this transformers"}))
    sys.exit(0)

cfg = AutoConfig.for_model(model_type)
for key, value in dict(
    vocab_size = 128, hidden_size = 32, n_embd = 32, n_layer = 2, n_head = 4, n_inner = 64, rotary_dim = 4,
    d_model = 32, d_ff = 64, d_kv = 8, num_layers = 2, num_decoder_layers = 2, num_heads = 4,
).items():
    if hasattr(cfg, key):
        setattr(cfg, key, value)
for key in list(vars(cfg)):
    if "drop" in key and isinstance(getattr(cfg, key), float):
        setattr(cfg, key, 0.0)

torch.manual_seed(0)
model = getattr(modeling, class_name)(copy.deepcopy(cfg)).eval()
state = {k: v.clone() for k, v in model.state_dict().items()}
ids = torch.randint(3, 120, (1, 9), generator = torch.Generator().manual_seed(1))
changed = ids.clone()
changed[0, 8] = (changed[0, 8] + 7) % 120 + 3


def future_influence(m):
    """Largest change in the logits of positions 0..7 when only token 8 is edited."""
    keys = ("decoder_input_ids",) if "ForConditionalGeneration" in class_name else ("input_ids",)
    extra = {"input_ids": ids[:, :5]} if "ForConditionalGeneration" in class_name else {}
    with torch.no_grad():
        a = m(**{keys[0]: ids}, **extra).logits
        b = m(**{keys[0]: changed}, **extra).logits
    return (a[:, :8] - b[:, :8]).abs().max().item()


stock = future_influence(model)
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = model_type, fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
compiled_model = getattr(modeling, class_name)(copy.deepcopy(cfg)).eval()
compiled_model.load_state_dict(state)
print("@@@" + json.dumps({"stock": stock, "compiled": future_influence(compiled_model)}))
'''


@pytest.mark.parametrize(
    ("model_type", "class_name"),
    [
        pytest.param("t5", "T5ForConditionalGeneration", id = "t5"),
        pytest.param("bloom", "BloomForCausalLM", id = "bloom"),
        pytest.param("gptj", "GPTJForCausalLM", id = "gptj"),
    ],
)
def test_compiled_eager_decoder_ignores_future_tokens(tmp_path, model_type, class_name):
    pytest.importorskip(f"transformers.models.{model_type}.modeling_{model_type}")
    env = dict(os.environ)
    env.update(
        UNSLOTH_ALLOW_CPU = "1", UNSLOTH_ZOO_DISABLE_GPU_INIT = "1",
        UNSLOTH_COMPILE_DISABLE = "1", UNSLOTH_COMPILE_LOCATION = "unsloth_compiled_cache",
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, model_type, class_name],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith("@@@")]
    assert line, result.stdout[-4000:] + result.stderr[-4000:]
    payload = json.loads(line[-1][3:])
    if "skip" in payload:
        pytest.skip(payload["skip"])
    assert payload["stock"] < 1e-5, payload
    assert payload["compiled"] < 1e-5, payload
