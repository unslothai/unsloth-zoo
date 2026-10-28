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

"""Generated LoRA forward with several active adapters (`set_adapter(["a", "b"])`)."""

from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo import compiler

peft = pytest.importorskip("peft")


# peft/tuners/lora/layer.py, class Linear.
LAYER_LINEAR = """\
    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        for active_adapter in self.active_adapters:
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            if active_adapter not in self.lora_variant:  # vanilla LoRA
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                result = self.lora_variant[active_adapter].forward(
                    self, active_adapter=active_adapter, x=x, result=result,
                )

        result = result.to(torch_result_dtype)
        return result
"""

# peft/tuners/lora/bnb.py, class Linear4bit.
BNB_LINEAR4BIT = """\
    def forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype

            if active_adapter not in self.lora_variant:  # vanilla LoRA
                output = lora_B(lora_A(dropout(x))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result = result + output
            else:
                result = self.lora_variant[active_adapter].forward(
                    self, active_adapter=active_adapter, x=x, result=result,
                )

        return result
"""

UPDATE_OUTSIDE_LOOP = """\
    def forward(self, x):
        result = self.base_layer(x)
        lora_A = self.lora_A["a"]
        lora_B = self.lora_B["a"]
        dropout = self.lora_dropout["a"]
        scaling = self.scaling["a"]
        result = result + lora_B(lora_A(dropout(x))) * scaling
        return result
"""


def _dedent_like_compiler(source):
    spaces = source.find("def")
    return "\n".join(line[spaces:] for line in source.split("\n"))


def _run_generated(source, layer, x):
    namespace = {"torch": torch}
    exec(textwrap.dedent(compiler.COMPILED_LORA_FORWARD), namespace)
    exec(source.replace("def forward", "def unsloth_forward", 1), namespace)
    return namespace["unsloth_forward"](layer, x)


def _layer(adapters):
    layer = SimpleNamespace(
        base_layer = torch.nn.Linear(6, 6),
        active_adapters = list(adapters),
        lora_A = {}, lora_B = {}, lora_dropout = {}, scaling = {}, lora_variant = {},
    )
    generator = torch.Generator().manual_seed(0)
    for name, rank, scaling in (("a", 2, 2.0), ("b", 3, 0.5)):
        A = torch.nn.Linear(6, rank, bias = False)
        B = torch.nn.Linear(rank, 6, bias = False)
        for module in (A, B):
            module.weight.data = torch.randn(module.weight.shape, generator = generator)
        layer.lora_A[name], layer.lora_B[name] = A, B
        layer.lora_dropout[name] = torch.nn.Identity()
        layer.scaling[name] = scaling
    return layer


def _expected(layer, x, adapters):
    out = layer.base_layer(x)
    for name in adapters:
        out = out + layer.lora_B[name](layer.lora_A[name](x)) * layer.scaling[name]
    return out


@pytest.mark.parametrize("shape", [LAYER_LINEAR, BNB_LINEAR4BIT])
@pytest.mark.parametrize("adapters", [("a",), ("b",), ("a", "b"), ("b", "a")])
def test_every_active_adapter_contributes(shape, adapters):
    source = compiler._patch_lora_update(_dedent_like_compiler(shape))
    assert "lora_forward(" in source
    layer = _layer(adapters)
    x = torch.randn(3, 6, generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = _run_generated(source, layer, x)
        expected = _expected(layer, x, adapters)
    torch.testing.assert_close(out, expected, rtol = 1e-5, atol = 1e-5)


def test_update_outside_the_adapter_loop_still_compiles():
    source = compiler._patch_lora_update(_dedent_like_compiler(UPDATE_OUTSIDE_LOOP))
    compile(source, "<test>", "exec")
    assert "return lora_forward(" in source
    layer = _layer(("a",))
    x = torch.randn(3, 6, generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        out = _run_generated(source, layer, x)
        expected = _expected(layer, x, ("a",))
    torch.testing.assert_close(out, expected, rtol = 1e-5, atol = 1e-5)


def test_source_without_the_vanilla_update_is_untouched():
    source = _dedent_like_compiler(LAYER_LINEAR).replace(
        "result = result + lora_B(lora_A(dropout(x))) * scaling", "pass"
    )
    assert compiler._patch_lora_update(source) == source


@pytest.fixture
def unsloth_lora_forwards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = [(cls, cls.__dict__.get("forward")) for cls, _, _ in compiler.get_lora_layer_modules()]
    compiler.patch_lora_forwards({})
    try:
        yield
    finally:
        for cls, forward in saved:
            if forward is not None:
                cls.forward = forward
            elif "forward" in cls.__dict__:
                del cls.forward


def _peft_two_adapters():
    from peft import LoraConfig, get_peft_model

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    # init_lora_weights = False gives lora_B nonzero values, so each adapter changes the output.
    model = get_peft_model(
        _Wrap(),
        LoraConfig(r = 4, lora_alpha = 8, target_modules = ["fc"], init_lora_weights = False),
        adapter_name = "first",
    )
    model.add_adapter(
        "second",
        LoraConfig(r = 2, lora_alpha = 2, target_modules = ["fc"], init_lora_weights = False),
    )
    return model


def test_patched_peft_linear_keeps_all_active_adapters(unsloth_lora_forwards):
    model = _peft_two_adapters()
    layer = model.base_model.model.fc
    assert type(layer).forward.__name__ == "unsloth_forward"
    x = torch.randn(3, 8, generator = torch.Generator().manual_seed(3))

    def expected(names):
        out = layer.base_layer(x)
        for name in names:
            out = out + layer.lora_B[name](layer.lora_A[name](x)) * layer.scaling[name]
        return out

    with torch.no_grad():
        for names in (["first"], ["second"], ["first", "second"]):
            model.base_model.set_adapter(names)
            assert layer.active_adapters == names
            torch.testing.assert_close(model(x), expected(names), rtol = 1e-5, atol = 1e-5)
        both, first = expected(["first", "second"]), expected(["first"])
        assert not torch.allclose(both, first), "second adapter must change the output"


def test_patched_peft_linear_trains_every_active_adapter(unsloth_lora_forwards):
    model = _peft_two_adapters()
    model.base_model.set_adapter(["first", "second"])
    x = torch.randn(3, 8, generator = torch.Generator().manual_seed(4))
    model(x).sum().backward()
    layer = model.base_model.model.fc
    for name in ("first", "second"):
        grad = layer.lora_B[name].weight.grad
        assert grad is not None and grad.abs().max() > 0, f"adapter {name} got no gradient"


def test_patched_peft_linear_accumulates_adapters_before_the_dtype_cast(unsloth_lora_forwards):
    from peft import LoraConfig, get_peft_model

    # bf16 base, fp32 adapters: each 0.003 update rounds away in bf16, their sum does not.
    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(1, 1, bias = False, dtype = torch.bfloat16)),
        LoraConfig(r = 1, lora_alpha = 1, target_modules = ["0"]),
        adapter_name = "a",
    )
    model.add_adapter("b", LoraConfig(r = 1, lora_alpha = 1, target_modules = ["0"]))
    layer = model.base_model.model[0]
    with torch.no_grad():
        layer.base_layer.weight.fill_(1.0)
        for name in ("a", "b"):
            layer.lora_A[name].weight.fill_(1.0)
            layer.lora_B[name].weight.fill_(0.003)
    assert layer.lora_A["a"].weight.dtype == torch.float32
    model.base_model.set_adapter(["a", "b"])
    with torch.no_grad():
        out = model(torch.tensor([[1.0]], dtype = torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert out.item() == 1.0078125
