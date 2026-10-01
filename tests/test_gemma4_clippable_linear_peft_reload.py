import sys
import types

import torch


def test_gemma4_clippable_linear_peft_reload_patch(monkeypatch):
    from peft.tuners.lora.model import LoraModel
    from unsloth_zoo.temporary_patches.gemma4 import patch_Gemma4ClippableLinear_peft_reload

    class Gemma4ClippableLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    fake_module = types.ModuleType("transformers.models.gemma4.modeling_gemma4")
    fake_module.Gemma4ClippableLinear = Gemma4ClippableLinear
    monkeypatch.setitem(sys.modules, "transformers.models.gemma4.modeling_gemma4", fake_module)

    calls = []

    def original(self, peft_config, adapter_name, target, target_name, parent, current_key=None, **kwargs):
        calls.append((target, target_name, parent, current_key, kwargs))
        return "created"

    monkeypatch.setattr(LoraModel, "_create_and_replace", original)

    patch_Gemma4ClippableLinear_peft_reload()
    patched = LoraModel._create_and_replace

    target = Gemma4ClippableLinear()
    assert patched(None, None, "default", target, "q_proj", object(), current_key="model.q_proj") == "created"
    assert calls[-1][0] is target.linear
    assert calls[-1][1] == "linear"
    assert calls[-1][2] is target
    assert calls[-1][3] == "model.q_proj"

    ordinary = torch.nn.Linear(2, 2)
    parent = object()
    assert patched(None, None, "default", ordinary, "q_proj", parent, current_key="model.q_proj") == "created"
    assert calls[-1][0] is ordinary
    assert calls[-1][1] == "q_proj"
    assert calls[-1][2] is parent
