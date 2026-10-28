# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""safe_serialization = None must mean the safetensors default, never a pickle: Unsloth's warning tells callers to pass
it to force safetensors, but forwarded verbatim None is falsy to peft and transformers and writes .bin
(unslothai/unsloth issue 1792); CPU-only."""

import ast
import inspect
from pathlib import Path

import pytest

from unsloth_zoo.saving_utils import normalize_safe_serialization
import unsloth_zoo.saving_utils as SU


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, True),
        (True, True),
        (False, False),
    ],
)
def test_none_is_the_safetensors_default(value, expected):
    """`None` is an override that asks for safetensors, not an absent argument."""
    assert normalize_safe_serialization(value) is expected


def test_false_still_means_a_pickle():
    """Only `None` is rewritten, so a caller who wants a pickle still gets one."""
    assert normalize_safe_serialization(False) is False


def test_it_is_exported():
    """Callers outside this module normalise with the same function, not a local copy."""
    assert "normalize_safe_serialization" in SU.__all__


def test_the_dequantizing_merge_normalises_before_it_writes():
    """Checked on source since the merge needs a real model: merge_and_dequantize_lora forwards the value to two
    save_pretrained calls, so normalise before either writer."""
    source = inspect.getsource(SU.merge_and_dequantize_lora)
    tree = ast.parse(source.lstrip())
    body = tree.body[0].body

    normalisation_index = None
    for index, node in enumerate(body):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "safe_serialization" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "normalize_safe_serialization"
        ):
            normalisation_index = index
            break
    assert normalisation_index is not None, (
        "merge_and_dequantize_lora must normalise safe_serialization"
    )

    forwarding_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword)
        and node.arg == "safe_serialization"
        and isinstance(node.value, ast.Name)
        and node.value.id == "safe_serialization"
    ]
    assert forwarding_lines, "expected safe_serialization to be forwarded to a writer"
    assert min(forwarding_lines) > body[normalisation_index].lineno


def test_no_save_entry_point_forwards_an_unnormalised_parameter():
    """Drift gate over the whole package: any function that takes safe_serialization and forwards that name must normalise it first, or None reaches a writer again."""
    package = Path(SU.__file__).resolve().parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = node.args
            names = [
                a.arg
                for a in list(arguments.posonlyargs) + list(arguments.args) + list(arguments.kwonlyargs)
            ]
            if "safe_serialization" not in names:
                continue
            forwards = any(
                isinstance(inner, ast.keyword)
                and inner.arg == "safe_serialization"
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "safe_serialization"
                for inner in ast.walk(node)
            )
            if not forwards:
                continue
            normalises = any(
                isinstance(inner, ast.Call)
                and getattr(inner.func, "id", None) == "normalize_safe_serialization"
                for inner in ast.walk(node)
            )
            if not normalises:
                offenders.append(f"{path.relative_to(package.parent)}::{node.name}")
    assert offenders == [], (
        "these take safe_serialization and forward it without normalising `None` first: "
        + ", ".join(offenders)
    )
