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

import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")


def test_new_patches_are_registered():
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    from unsloth_zoo.temporary_patches.moe_experts_interface import patch_experts_interface
    from unsloth_zoo.temporary_patches.llama4_moe import patch_llama4_moe
    from unsloth_zoo.temporary_patches.inkling import patch_inkling_text_config
    from unsloth_zoo.temporary_patches.compiled_model_identity import patch_compiled_model_is_custom_code
    from unsloth_zoo.temporary_patches.bitsandbytes_large_tensors import patch_bitsandbytes_large_tensors
    for patch in (
        patch_experts_interface,
        patch_llama4_moe,
        patch_inkling_text_config,
        patch_compiled_model_is_custom_code,
        patch_bitsandbytes_large_tensors,
    ):
        assert patch in TEMPORARY_PATCHES, patch.__name__


def test_tuple_returning_linear_is_skipped_under_any_name():
    import re
    from unsloth_zoo.peft_utils import get_peft_regex

    class Scorer(nn.Linear):
        def forward(self, x):
            logits = super().forward(x)
            return torch.sigmoid(logits), logits

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.gate_scorer = Scorer(8, 4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("C", (), {"model_type": "x", "architectures": [], "_name_or_path": "x"})()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
            for layer in self.model.layers:
                layer.self_attn = Block()
                layer.mlp = Block()

    model = Model()
    regex = get_peft_regex(model)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    matched = {n for n in names if re.fullmatch(regex, n)}
    assert not any("gate_scorer" in n for n in matched), matched
    assert any(n.endswith("q_proj") for n in matched)


def test_quantizer_leaves_experts_it_cannot_route_unpacked():
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit
    if not moe_utils_bnb4bit.HAS_BNB:
        pytest.skip("bitsandbytes 4-bit is not usable here, so no 4-bit load reaches the quantizer")
    if not moe_utils_bnb4bit.is_transformers_v5_moe_quantization_available():
        pytest.skip("transformers has no v5 MoE quantization hooks")
    from bitsandbytes.nn import Params4bit
    from transformers import BitsAndBytesConfig
    from unsloth_zoo.temporary_patches.moe_utils import forward_moe_backend
    replace_expert_params_with_bnb_params = moe_utils_bnb4bit.replace_expert_params_with_bnb_params

    def own_forward(self, hidden_states):
        return torch.bmm(hidden_states, self.gate_up_proj)

    Handled = type("HandledExperts", (nn.Module,), {"forward": forward_moe_backend})
    Unhandled = type("UnhandledExperts", (nn.Module,), {"forward": own_forward})

    model = nn.Module()
    model.handled = Handled()
    model.unhandled = Unhandled()
    for m in (model.handled, model.unhandled):
        m.gate_up_proj = nn.Parameter(torch.zeros(2, 8, 4))
        m.down_proj = nn.Parameter(torch.zeros(2, 4, 4))

    config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_type = "nf4")
    replace_expert_params_with_bnb_params(model, quantization_config = config)
    assert isinstance(model.handled.gate_up_proj, Params4bit)
    assert isinstance(model.handled.down_proj, Params4bit)
    assert type(model.unhandled.gate_up_proj) is nn.Parameter
    assert type(model.unhandled.down_proj) is nn.Parameter


_INKLING_COMPILE_CHILD = r'''
import os, io, json, contextlib
os.environ["UNSLOTH_COMPILE_LOCATION"] = "unsloth_compiled_cache"
from unsloth_zoo.compiler import unsloth_compile_transformers
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    unsloth_compile_transformers(
        model_type = "inkling", fast_lora_forwards = False, fullgraph = True,
        import_from_cache = False, disable = False, supports_sdpa = [None],
    )
generated = open(os.path.join("unsloth_compiled_cache", "unsloth_compiled_module_inkling.py")).read()
where = generated.find("def plan_out_scales(")
line_above = generated[:where].rsplit("\n", 2)[-2] if where != -1 else None
print("@@@" + json.dumps({"found": where != -1, "line_above": line_above}))
'''


def test_compiler_emits_scalar_only_helpers_uncompiled(tmp_path):
    pytest.importorskip("transformers.models.inkling.modeling_inkling")
    env = dict(os.environ)
    env["UNSLOTH_ALLOW_CPU"] = "1"
    env["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
    env.pop("UNSLOTH_COMPILE_DISABLE", None)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    result = subprocess.run(
        [sys.executable, "-c", _INKLING_COMPILE_CHILD],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 1800,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    payload = next(json.loads(l[3:]) for l in result.stdout.splitlines() if l.startswith("@@@"))
    assert payload["found"], "plan_out_scales is not in the generated module, so this proves nothing"
    assert "torch_compile" not in payload["line_above"] and "torch.compile" not in payload["line_above"], payload
