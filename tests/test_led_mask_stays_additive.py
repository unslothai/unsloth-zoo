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

"""LED's additive encoder mask must not be rewritten to a boolean mask; its Longformer attention is not SDPA."""

import json
import os
import subprocess
import sys

import pytest

_CHILD = r'''
import contextlib, copy, io, json, os
import torch
import transformers.models.led.modeling_led as led
from transformers import AutoConfig
from unsloth_zoo.compiler import unsloth_compile_transformers

cfg = AutoConfig.for_model("led")
cfg.d_model, cfg.vocab_size, cfg.attention_window = 32, 128, [4, 4]
cfg.encoder_layers = cfg.decoder_layers = 2
if hasattr(cfg, "num_hidden_layers"):
    cfg.num_hidden_layers = 2
cfg.encoder_attention_heads = cfg.decoder_attention_heads = 4
cfg.encoder_ffn_dim = cfg.decoder_ffn_dim = 64
for key in list(vars(cfg)):
    if "drop" in key and isinstance(getattr(cfg, key), float):
        setattr(cfg, key, 0.0)

torch.manual_seed(0)
stock = led.LEDForConditionalGeneration(copy.deepcopy(cfg)).eval()
state = {k: v.clone() for k, v in stock.state_dict().items()}
ids = torch.randint(3, 120, (2, 9), generator = torch.Generator().manual_seed(1))
mask = torch.ones_like(ids)
mask[1, 6:] = 0
with torch.no_grad():
    expected = stock(input_ids = ids, attention_mask = mask).encoder_last_hidden_state

with contextlib.redirect_stdout(io.StringIO()):
    unsloth_compile_transformers(
        model_type = "led", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
compiled = led.LEDForConditionalGeneration(copy.deepcopy(cfg)).eval()
compiled.load_state_dict(state)
with torch.no_grad():
    actual = compiled(input_ids = ids, attention_mask = mask).encoder_last_hidden_state
print("@@@" + json.dumps({
    "replaced": type(compiled.led.encoder.layers[0].self_attn.longformer_self_attn).__module__,
    "max_diff": (expected - actual).abs().max().item(),
}))
'''


def test_compiled_led_encoder_matches_stock(tmp_path):
    pytest.importorskip("transformers.models.led.modeling_led")
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
        [sys.executable, "-c", _CHILD],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    line = [row for row in result.stdout.splitlines() if row.startswith("@@@")]
    assert line, result.stdout[-4000:] + result.stderr[-4000:]
    payload = json.loads(line[-1][3:])
    assert payload["replaced"].startswith("unsloth_compiled_module_led"), payload
    assert payload["max_diff"] < 1e-5, payload
