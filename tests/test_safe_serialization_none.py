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

"""`safe_serialization = None` means the safetensors default, never a pickle.

Unsloth's own warning, and the troubleshooting page it points at, tell a caller to pass
`safe_serialization = None` to FORCE safetensors. `None` is falsy to peft and to
transformers, so a save that forwards it verbatim selects the torch pickle writer and
produces `adapter_model.bin` or `pytorch_model.bin`: the documented remedy for `.bin`
files produces `.bin` files (unslothai/unsloth#1792).

Measured on unsloth main before the fix, on a two-step Llama-3.2-1B LoRA:
`model.save_pretrained(directory, safe_serialization = None)` wrote a 22.6 MB
`adapter_model.bin`, while the same call with the argument omitted wrote
`adapter_model.safetensors`.

CPU-only, no network, no model download.
"""

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
    """`merge_and_dequantize_lora` forwards the value into two `save_pretrained` calls.

    Asserted on the source rather than by running the merge: the function rewrites
    `PreTrainedModel.save_pretrained` from its own source text and needs a real model,
    which is neither CPU-cheap nor network-free, while the normalisation is a single
    statement whose position is the whole point. It must come before either writer.
    """
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
    """A drift gate over the package, not just over the one function fixed here.

    Any function that takes a `safe_serialization` parameter AND passes that same name on
    to something else has to normalise it first, or `None` reaches a writer again the
    next time one of these is added.
    """
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
