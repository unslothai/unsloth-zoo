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

"""An adapter Unsloth saves must reload in plain PEFT (no Unsloth patches loaded).

Two target_modules shapes broke that:
- Gemma 4: the LoRA sits on Gemma4ClippableLinear's inner `.linear`, but the saved leaf
  name ("q_proj") makes plain PEFT pick the wrapper and raise "not supported".
- Fused MoE experts: leaf names that wrapped nothing (gate_proj, up_proj) make PEFT's
  transformers v5 MoE conversion double the rank of gate_up_proj, so the load fails.

CPU only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

from unsloth_zoo.temporary_patches import moe_utils as MU  # noqa: E402


from _portable_targets_models import TwoTowers  # noqa: E402

HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_portable_targets_models.py")


def _reload_in_plain_peft(adapter_dir, state, x):
    """Load in a fresh interpreter that never imports Unsloth, whose patches would let
    this process load what plain PEFT cannot. Returns (returncode, stderr, output)."""
    torch.save(state, str(adapter_dir / "state.pt"))
    torch.save(x, str(adapter_dir / "x.pt"))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(
        [sys.executable, HELPER, str(adapter_dir), str(adapter_dir / "state.pt"), str(adapter_dir / "x.pt")],
        env = env, capture_output = True, text = True, timeout = 300,
    )
    out = torch.load(str(adapter_dir / "x.pt.out")) if r.returncode == 0 else None
    return r.returncode, r.stderr, out


def _seed_lora_b(model):
    g = torch.Generator().manual_seed(0)
    for module in model.modules():
        lora_B = getattr(module, "lora_B", None)
        if lora_B is not None and hasattr(lora_B, "keys"):
            for k in lora_B.keys():
                with torch.no_grad():
                    lora_B[k].weight.copy_(torch.randn(lora_B[k].weight.shape, generator = g))


def _unsloth_style_gemma4_adapter(tmp_path):
    """What Unsloth leaves on disk: LoRA on text q_proj and vision q_proj.linear, and
    target_modules = ["q_proj"] in adapter_config.json."""
    torch.manual_seed(0)
    base = TwoTowers()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    config = LoraConfig(r = 2, lora_alpha = 4, target_modules = r".*(?:text\.\d+\.q_proj|vision\.\d+\.q_proj\.linear)")
    model = get_peft_model(base, config)
    _seed_lora_b(model)
    model.peft_config["default"].target_modules = {"q_proj"}
    model.save_pretrained(str(tmp_path))
    # Pin the pre-fix file whether or not this process has the save hook installed.
    path = tmp_path / "adapter_config.json"
    config = json.loads(path.read_text())
    config["target_modules"] = ["q_proj"]
    path.write_text(json.dumps(config))
    return model, state


def test_plain_peft_rejects_the_leaf_name_for_a_wrapped_linear(tmp_path):
    # The bug being fixed: without the rewrite plain PEFT picks the wrapper.
    _, state = _unsloth_style_gemma4_adapter(tmp_path)
    code, err, _ = _reload_in_plain_peft(tmp_path, state, torch.randn(3, 6))
    assert code != 0 and "is not supported" in err


def test_gemma4_style_adapter_reloads_in_plain_peft(tmp_path):
    model, state = _unsloth_style_gemma4_adapter(tmp_path)
    written = MU.write_portable_target_modules(model, str(tmp_path))
    assert written == [str(tmp_path / "adapter_config.json")]
    saved = json.loads((tmp_path / "adapter_config.json").read_text())["target_modules"]
    assert isinstance(saved, str)

    x = torch.randn(3, 6)
    code, err, out = _reload_in_plain_peft(tmp_path, state, x)
    assert code == 0, err[-2000:]
    with torch.no_grad():
        assert torch.allclose(model(x), out)


def test_compact_regex_selects_nothing_extra():
    all_names = ["a.0.q_proj", "a.1.q_proj", "a.10.q_proj", "b.0.q_proj"]
    regex = MU._target_regex_for(["a.0.q_proj", "a.1.q_proj", "a.10.q_proj"], all_names)
    assert {n for n in all_names if re.fullmatch(regex, n)} == {"a.0.q_proj", "a.1.q_proj", "a.10.q_proj"}
    # Only some layers wrapped: the collapsed form would widen, so the names stay literal.
    regex = MU._target_regex_for(["a.0.q_proj"], all_names)
    assert {n for n in all_names if re.fullmatch(regex, n)} == {"a.0.q_proj"}


class Experts(nn.Module):
    def __init__(self, e = 4, h = 6, i = 5):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.randn(e, h, i))


class MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(6, 6, bias = False)
        self.experts = Experts()


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MoEBlock() for _ in range(2)])
        self.config = types.SimpleNamespace(model_type = "qwen3_moe")


def _fused_expert_model():
    config = LoraConfig(
        r = 2, lora_alpha = 4,
        target_modules = ["q_proj", "gate_proj", "up_proj", "down_proj"],
        target_parameters = ["experts.gate_up_proj", "experts.down_proj"],
    )
    return get_peft_model(TinyMoE(), config)


def test_fused_expert_adapter_drops_leaf_names_that_wrapped_nothing():
    model = _fused_expert_model()
    out = MU.portable_lora_target_modules(
        model, "default", ["q_proj", "gate_proj", "up_proj", "down_proj"],
    )
    assert out == ["q_proj"]
    # Nothing to drop: left alone.
    assert MU.portable_lora_target_modules(model, "default", ["q_proj"]) is None
    # A regex is never rewritten.
    assert MU.portable_lora_target_modules(model, "default", ".*q_proj") is None


def test_dropped_names_stop_pefts_v5_moe_conversion_doubling_the_rank():
    conversion = pytest.importorskip("peft.utils.transformers_weight_conversion")
    convert = getattr(conversion, "_convert_peft_config_moe", None)
    if convert is None or "qwen3_moe" not in getattr(conversion, "_MODEL_TO_CONVERSION_PATTERN", {}):
        pytest.skip(reason = "this PEFT has no transformers v5 MoE conversion for qwen3_moe")
    # Unsloth's import_fixes wraps it to skip explicit targets; plain PEFT runs the original.
    while hasattr(convert, "__wrapped__"):
        convert = convert.__wrapped__

    def converted(target_modules):
        config = LoraConfig(
            r = 2, lora_alpha = 4, target_modules = list(target_modules),
            target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        )
        convert(config, TinyMoE())
        return config.rank_pattern

    assert converted(["q_proj", "gate_proj", "up_proj"])  # the bug: rank doubled
    model = _fused_expert_model()
    kept = MU.portable_lora_target_modules(model, "default", ["q_proj", "gate_proj", "up_proj"])
    assert converted(kept) == {}


def test_dense_adapter_is_left_alone():
    model = get_peft_model(TwoTowers(), LoraConfig(r = 2, target_modules = ["linear"]))
    assert MU.portable_lora_target_modules(model, "default", ["linear"]) is None


def test_save_pretrained_hook_writes_the_portable_targets(tmp_path, monkeypatch):
    from peft import PeftModel
    monkeypatch.setattr(PeftModel, "save_pretrained", PeftModel.save_pretrained)
    assert MU._patch_peft_save_pretrained_for_moe_layout()
    model = _fused_expert_model()
    model.base_model.model.config = {"model_type": "qwen3_moe"}
    model.save_pretrained(str(tmp_path))
    saved = json.loads((tmp_path / "adapter_config.json").read_text())["target_modules"]
    assert saved == ["q_proj"]
