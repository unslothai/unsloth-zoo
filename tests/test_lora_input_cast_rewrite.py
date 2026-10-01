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

"""Source-level tests for the LoRA input cast rewrite in compiler.py (#4127).

Deleting PEFT's cast outright under UNSLOTH_FORCE_FLOAT32 is safe for vanilla LoRA, which
is rewritten into the self-casting `lora_forward`, but handed variants (DoRA, QALoRA,
aLoRA) an uncast `x` and killed DoRA with "c10::Half != float".
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from types import SimpleNamespace

import pytest
import torch

from unsloth_zoo.compiler import (
    _LORA_INPUT_CASTS,
    _patch_lora_input_cast,
    patch_lora_forwards,
)


# peft/tuners/lora/layer.py, class Linear: the cast sits directly in the adapter loop.
LAYER_LINEAR = """\
    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                        **kwargs,
                    )

            result = result.to(torch_result_dtype)

        return result
"""

# peft/tuners/lora/bnb.py, class Linear4bit: the cast is the LAST statement of an
# `if requires_conversion:` block, so removing the line would leave it empty.
BNB_LINEAR4BIT = """\
    def forward(self, x, *args, **kwargs):
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            result = result.clone()

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)

                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    output = lora_B(lora_A(dropout(x))) * scaling
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    result = result + output
                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                        **kwargs,
                    )

        return result
"""

# The bare `.to()` form, still what peft/tuners/lora/variants.py uses.
LEGACY_TO_SPELLING = LAYER_LINEAR.replace(
    "x = self._cast_input_dtype(x, lora_A.weight.dtype)",
    "x = x.to(lora_A.weight.dtype)",
)

SHAPES = {
    "layer.Linear": LAYER_LINEAR,
    "bnb.Linear4bit": BNB_LINEAR4BIT,
    "legacy .to() spelling": LEGACY_TO_SPELLING,
}

CAST_STATEMENTS = tuple(statement for statement, _ in _LORA_INPUT_CASTS)


def _dedent_like_compiler(source: str) -> str:
    spaces = source.find("def")
    return "\n".join(line[spaces:] for line in source.split("\n"))


def _assigns_x(node: ast.Assign) -> bool:
    for target in node.targets:
        if isinstance(target, ast.Name) and target.id == "x":
            return True
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                if isinstance(element, ast.Name) and element.id == "x":
                    return True
    return False


def _cast_assignments_in_adapter_loop(source: str) -> list[str]:
    """Casts of `x` to the LoRA dtype inside a `for active_adapter` loop. Outside one,
    the guard compiles and then raises NameError at the first forward."""
    tree = ast.parse(source)
    found: list[str] = []

    def walk(node: ast.AST, loop_targets: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            targets = loop_targets
            if isinstance(child, ast.For):
                targets = targets + (ast.unparse(child.target),)
            if (
                isinstance(child, ast.Assign)
                and "active_adapter" in targets
                and _assigns_x(child)
                and "lora_A.weight.dtype" in ast.unparse(child)
            ):
                found.append(ast.unparse(child))
            walk(child, targets)

    walk(tree, ())
    return found


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("force_float32", ["0", "1"])
def test_cast_survives_for_every_shape_and_switch(shape, force_float32, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", force_float32)
    source = _dedent_like_compiler(SHAPES[shape])
    assert any(statement in source for statement in CAST_STATEMENTS), (
        f"fixture {shape} no longer carries a cast spelling; fix the fixture"
    )

    rewritten = _patch_lora_input_cast(source)
    rewritten = rewritten.replace("def forward", "def unsloth_forward", 1)
    ast.parse(rewritten)

    assert "lora_A.weight.dtype" in rewritten, (
        f"{shape} under UNSLOTH_FORCE_FLOAT32={force_float32}: the cast to the "
        "LoRA dtype was dropped, which is the #4127 regression"
    )
    casts = _cast_assignments_in_adapter_loop(rewritten)
    assert casts, (
        f"{shape} under UNSLOTH_FORCE_FLOAT32={force_float32}: no cast of x to "
        "the LoRA dtype inside the `for active_adapter` loop"
    )


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_forced_float32_guards_the_cast_on_the_variant(shape, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    rewritten = _patch_lora_input_cast(_dedent_like_compiler(SHAPES[shape]))
    casts = _cast_assignments_in_adapter_loop(rewritten)
    assert len(casts) == 1, casts
    assert "lora_variant" in casts[0], (
        "the cast must be gated on lora_variant so the vanilla LoRA path, which "
        f"casts inside lora_forward, is untouched: {casts[0]}"
    )
    assert "active_adapter in" in casts[0], casts[0]
    for statement in CAST_STATEMENTS:
        assert statement not in rewritten or "lora_variant" in rewritten


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_unforced_path_is_unchanged_by_the_fix(shape, monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising=False)
    source = _dedent_like_compiler(SHAPES[shape])
    rewritten = _patch_lora_input_cast(source)
    if "torch.is_autocast_enabled()" in source:
        assert rewritten == source
    else:
        assert "if not torch.is_autocast_enabled(): result, x = " in rewritten
        assert "lora_variant" not in _cast_assignments_in_adapter_loop(rewritten)[0]


class _Recorder:
    def __init__(self, dtype):
        self.weight = SimpleNamespace(dtype=dtype)
        self.seen: list[torch.dtype] = []

    def __call__(self, value):
        self.seen.append(value.dtype)
        return value.to(self.weight.dtype)


class _RecordingVariant:
    def __init__(self):
        self.seen: list[torch.dtype] = []

    def forward(self, module, active_adapter, x, result, **kwargs):
        self.seen.append(x.dtype)
        return result


def _run_rewritten_forward(source: str, *, variant: bool):
    namespace: dict = {"torch": torch}
    exec(textwrap.dedent(source), namespace)
    forward = namespace["unsloth_forward"]

    lora_A = _Recorder(torch.float32)
    lora_B = _Recorder(torch.float32)
    recording_variant = _RecordingVariant()
    layer = SimpleNamespace(
        disable_adapters=False,
        merged=False,
        base_layer=lambda value, *args, **kwargs: value,
        active_adapters=["default"],
        lora_A={"default": lora_A},
        lora_B={"default": lora_B},
        lora_dropout={"default": lambda value: value},
        scaling={"default": 1.0},
        lora_variant={"default": recording_variant} if variant else {},
        _cast_input_dtype=lambda value, dtype: value.to(dtype),
        _check_forward_args=lambda *args, **kwargs: None,
    )
    forward(layer, torch.zeros(2, 2, dtype=torch.float16))
    return lora_A.seen, recording_variant.seen


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_variant_branch_receives_a_cast_activation(shape, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    rewritten = _patch_lora_input_cast(_dedent_like_compiler(SHAPES[shape]))
    rewritten = rewritten.replace("def forward", "def unsloth_forward", 1)

    vanilla_seen, variant_seen = _run_rewritten_forward(rewritten, variant=True)
    assert variant_seen == [torch.float32], (
        f"{shape}: the LoRA variant was handed {variant_seen}, so DoRA still "
        "meets float32 LoRA weights with a float16 activation (#4127)"
    )


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_vanilla_branch_is_not_cast_under_forced_float32(shape, monkeypatch):
    """The vanilla path keeps the pre-fix behaviour: lora_forward does its own cast."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    rewritten = _patch_lora_input_cast(_dedent_like_compiler(SHAPES[shape]))
    rewritten = rewritten.replace("def forward", "def unsloth_forward", 1)

    vanilla_seen, variant_seen = _run_rewritten_forward(rewritten, variant=False)
    assert vanilla_seen == [torch.float16], (
        f"{shape}: the vanilla LoRA branch was handed {vanilla_seen}; the guard "
        "must leave it exactly as it was, since lora_forward casts internally"
    )
    assert variant_seen == []


def test_patch_lora_forwards_routes_through_the_helper():
    source = inspect.getsource(patch_lora_forwards)
    assert "_patch_lora_input_cast(source)" in source
    assert 'source.replace(replace, "")' not in source, (
        "the unconditional deletion of PEFT's cast is what #4127 was"
    )


def test_installed_peft_still_spells_the_cast_the_way_we_match():
    peft = pytest.importorskip("peft")
    from unsloth_zoo.peft_utils import get_lora_layer_modules

    layers = get_lora_layer_modules()
    if not layers:
        pytest.skip("no peft LoRA Linear layers discoverable in this environment")

    carriers = []
    for function, parent, child in layers:
        forward = getattr(function, "forward", None)
        if forward is None:
            continue
        try:
            source = inspect.getsource(forward)
        except (OSError, TypeError):
            continue
        if any(statement in source for statement in CAST_STATEMENTS):
            carriers.append(f"{parent}.{child}")
            rewritten = _patch_lora_input_cast(
                _dedent_like_compiler(source), force_float32=True
            )
            ast.parse(rewritten)
            assert _cast_assignments_in_adapter_loop(rewritten), (
                f"{parent}.{child}: rewritten forward has no in-scope cast"
            )

    assert carriers, (
        "DRIFT DETECTED: unsloth_zoo/compiler.py _LORA_INPUT_CASTS expects one of "
        f"{CAST_STATEMENTS} in a peft LoRA Linear forward, and peft "
        f"{getattr(peft, '__version__', 'unknown')} has none. The rewrite is now a "
        "no-op; re-derive the spelling from peft/tuners/lora/."
    )


def test_cast_spelling_table_is_well_formed():
    for statement, expression in _LORA_INPUT_CASTS:
        assert statement == f"x = {expression}"
        assert re.search(r"lora_A\.weight\.dtype", expression)
