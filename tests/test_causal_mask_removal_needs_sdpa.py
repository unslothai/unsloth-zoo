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

"""Compiled eager-attention decoders must stay causal: the causal-mask removal applies only under SDPA / flash."""

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

model_type, class_name, attn = sys.argv[1:4]
modeling = importlib.import_module(f"transformers.models.{model_type}.modeling_{model_type}")
hooks = [c for c in vars(modeling).values() if isinstance(c, type) and "_update_causal_mask" in vars(c)]
if not hooks:
    print("@@@" + json.dumps({"skip": "no _update_causal_mask hook on this transformers"}))
    sys.exit(0)

cfg = AutoConfig.for_model(model_type)
for key, value in dict(
    vocab_size = 128, hidden_size = 32, num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 4,
    intermediate_size = 64, ffn_dim = 64, word_embed_proj_dim = 32, n_layer = 2, n_head = 4, n_embd = 32,
).items():
    if hasattr(cfg, key):
        setattr(cfg, key, value)
for key in list(vars(cfg)):
    if "drop" in key and isinstance(getattr(cfg, key), float):
        setattr(cfg, key, 0.0)
cfg._attn_implementation = attn

torch.manual_seed(0)
model = getattr(modeling, class_name)(copy.deepcopy(cfg)).eval()
state = {k: v.clone() for k, v in model.state_dict().items()}
ids = torch.randint(3, 120, (1, 9), generator = torch.Generator().manual_seed(1))
changed = ids.clone()
changed[0, 8] = (changed[0, 8] + 7) % 120 + 3


def future_influence(m):
    with torch.no_grad():
        return (m(input_ids = ids).logits[:, :8] - m(input_ids = changed).logits[:, :8]).abs().max().item()


stock = future_influence(model)
originals = [c._update_causal_mask for c in hooks]
with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = model_type, fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
compiled_model = getattr(modeling, class_name)(copy.deepcopy(cfg)).eval()
compiled_model.load_state_dict(state)
print("@@@" + json.dumps({
    "stock": stock,
    "compiled": future_influence(compiled_model),
    "patched": any(c._update_causal_mask is not o for c, o in zip(hooks, originals)),
}))
'''


@pytest.mark.parametrize(
    ("model_type", "class_name", "attn"),
    [
        pytest.param("bloom", "BloomForCausalLM", "eager", id = "bloom-eager"),
        pytest.param("opt", "OPTForCausalLM", "eager", id = "opt-eager"),
        pytest.param("stablelm", "StableLmForCausalLM", "eager", id = "stablelm-eager"),
        pytest.param("opt", "OPTForCausalLM", "sdpa", id = "opt-sdpa"),
    ],
)
def test_compiled_decoder_ignores_future_tokens(tmp_path, model_type, class_name, attn):
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
        [sys.executable, "-c", _CHILD, model_type, class_name, attn],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith("@@@")]
    assert line, result.stdout[-4000:] + result.stderr[-4000:]
    payload = json.loads(line[-1][3:])
    if "skip" in payload:
        pytest.skip(reason = f"transformers 5.x dropped this hook: {payload['skip']}")
    assert payload["stock"] < 1e-5, payload
    assert payload["compiled"] < 1e-5, payload
    if attn == "sdpa":
        # Removal must still engage so SDPA keeps the memory saving.
        assert payload["patched"], payload
