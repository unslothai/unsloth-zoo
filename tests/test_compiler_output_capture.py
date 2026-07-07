# Tests for the transformers v5 output capture helpers in unsloth_zoo.compiler:
# patch_output_capture_targets and calls_output_capture_target.

import types

import pytest

from unsloth_zoo.compiler import (
    calls_output_capture_target,
    patch_output_capture_targets,
)

output_capturing = pytest.importorskip(
    "transformers.utils.output_capturing",
    reason="output capture targets exist on transformers >= 5.2 only",
)
OutputRecorder = output_capturing.OutputRecorder


class FakeRouter:
    pass


class FakeAttention:
    pass


class FakeDecoderLayer:
    pass


class ReplacementRouter:
    __name__ = "FakeRouter"


class ReplacementAttention:
    __name__ = "FakeAttention"


def make_modeling_file():
    mod = types.ModuleType("fake_modeling")

    class FakeModel:
        _can_record_outputs = {
            "router_logits": OutputRecorder(FakeRouter, index=0),
            "hidden_states": FakeDecoderLayer,
            "attentions": [OutputRecorder(FakeAttention, index=1)],
            "extras": (FakeAttention,),
        }

    mod.FakeModel = FakeModel
    mod.FakeRouter = FakeRouter
    mod.FakeAttention = FakeAttention
    mod.FakeDecoderLayer = FakeDecoderLayer
    return mod


def test_collects_target_names_without_replacements():
    mod = make_modeling_file()
    names = patch_output_capture_targets(mod)
    assert names == {"FakeRouter", "FakeAttention", "FakeDecoderLayer"}
    flags = mod.FakeModel._can_record_outputs
    assert flags["router_logits"].target_class is FakeRouter
    assert flags["hidden_states"] is FakeDecoderLayer


def test_retargets_specs_to_replacement_classes():
    mod = make_modeling_file()
    replacements = {
        "FakeRouter": ReplacementRouter,
        "FakeAttention": ReplacementAttention,
    }
    patch_output_capture_targets(mod, replacements)
    flags = mod.FakeModel._can_record_outputs

    rec = flags["router_logits"]
    assert rec.target_class is ReplacementRouter
    assert rec.index == 0

    # Bare classes without a replacement stay untouched.
    assert flags["hidden_states"] is FakeDecoderLayer

    # List and tuple specs keep their container type and are retargeted.
    assert isinstance(flags["attentions"], list)
    assert flags["attentions"][0].target_class is ReplacementAttention
    assert flags["attentions"][0].index == 1
    assert isinstance(flags["extras"], tuple)
    assert flags["extras"][0] is ReplacementAttention


def test_returns_empty_set_for_module_without_specs():
    mod = types.ModuleType("empty_modeling")
    assert patch_output_capture_targets(mod) == set()


def test_detects_direct_call_to_capture_target():
    init = "def __init__(self, config):\n    self.gate = FakeRouter(config)\n"
    source = "def forward(self, x):\n    return self.gate(x)\n"
    assert calls_output_capture_target(init, source, {"FakeRouter"})
    assert not calls_output_capture_target(init, source, {"OtherTarget"})


def test_ignores_assigned_but_uncalled_target():
    init = "def __init__(self, config):\n    self.gate = FakeRouter(config)\n"
    source = "def forward(self, x):\n    return x\n"
    assert not calls_output_capture_target(init, source, {"FakeRouter"})


def test_detects_capture_outputs_decorated_forward():
    # Targets built indirectly (e.g. Zamba2 get_layers) never match the
    # assignment regex, but the decorator alone means hooks run inside.
    init = "def __init__(self, config):\n    self.layers = self.get_layers(config)\n"
    source = "@capture_outputs\ndef forward(self, x):\n    return x\n"
    assert calls_output_capture_target(init, source, {"FakeRouter"})
