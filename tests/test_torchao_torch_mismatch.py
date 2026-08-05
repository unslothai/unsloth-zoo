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

"""A torchao built for a newer torch must give a readable error.

The ImportError lands in the `Unpack` guard in temporary_patches/utils.py
and used to fall through to a bare `raise Exception(e)`, naming neither
torchao nor torch. These cover what the new branch must catch, what it must
leave alone, and that it cannot itself raise.
"""

import ast
from pathlib import Path

import pytest

UTILS = (Path(__file__).resolve().parents[1] / "unsloth_zoo"
         / "temporary_patches" / "utils.py")


def _load_helpers():
    """Exec just the two helpers.

    Importing the module would run the very import guard under test and,
    on a healthy install, do nothing interesting -- and on a broken one,
    raise before a single assertion ran.
    """
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    wanted = {"_torchao_is_newer_than_torch", "_torchao_torch_mismatch_message"}
    ns: dict = {}
    for node in tree.body:
        keep = False
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            keep = True
        elif isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_TORCHAO_TORCH_SYMBOLS"
                for t in node.targets):
            keep = True
        if keep:
            exec(compile(ast.Module([node], []), "<utils>", "exec"), ns)
    missing = wanted - set(ns)
    assert not missing, f"helpers not found in utils.py: {missing}"
    return ns


HELPERS = _load_helpers()
looks_like = HELPERS["_torchao_is_newer_than_torch"]
message_of = HELPERS["_torchao_torch_mismatch_message"]

REAL = ("cannot import name 'ScalingType' from 'torch.nn.functional' "
        "(/usr/local/lib/python3.12/dist-packages/torch/nn/functional.py)")


# ---- what it must catch ---------------------------------------------------

def test_the_error_seen_in_the_wild():
    assert looks_like(REAL) is True


@pytest.mark.parametrize("sym", ["ScalingType", "ScalingGranularity",
                                 "Float8Tensor"])
def test_the_other_torchao_symbols(sym):
    assert looks_like(
        f"cannot import name '{sym}' from 'torch.nn.functional'") is True


# ---- what it must NOT catch ----------------------------------------------

def test_the_unpack_move_is_left_alone():
    """It sits directly beside the Unpack branch; swallowing that would
    hide a different problem with a different fix."""
    assert looks_like(
        "cannot import name 'Unpack' from 'transformers.processing_utils'"
    ) is False


def test_the_same_symbol_from_a_non_torch_module():
    assert looks_like("cannot import name 'ScalingType' from 'somelib.x'") is False


def test_a_non_import_error():
    assert looks_like("torchvision::nms does not exist") is False


def test_an_unrelated_missing_name_from_torch():
    assert looks_like(
        "cannot import name 'some_new_api' from 'torch.nn.functional'") is False


# ---- the guard must never be what raises ---------------------------------

@pytest.mark.parametrize("bad", [None, 123, object()])
def test_non_string_input_does_not_raise(bad):
    assert looks_like(bad) in (True, False)


def test_the_message_names_both_versions():
    m = message_of(REAL)
    assert "torchao" in m and "torch " in m
    assert REAL.split(" (")[0] in m, "the original error must survive"
    assert "torchao<0.18" in m, "the user needs something to run"


def test_the_message_survives_missing_metadata(monkeypatch):
    """Version lookup is best-effort; it must not turn a diagnostic into a
    second exception."""
    import importlib.metadata as md
    monkeypatch.setattr(md, "version",
                        lambda *_a, **_k: (_ for _ in ()).throw(Exception("no")))
    m = message_of(REAL)
    assert "unknown" in m


# ---- the call site --------------------------------------------------------

def test_the_branch_runs_before_the_generic_reraise():
    """Placed after `raise Exception(e)` it would never be reached."""
    src = UTILS.read_text(encoding="utf-8")
    guard = src.index("_torchao_is_newer_than_torch(e)")
    generic = src.index('elif "Unpack" not in e:')
    assert guard < generic, (
        "the torchao branch must precede the generic re-raise")


def test_the_branch_raises_runtimeerror_not_a_bare_exception():
    src = UTILS.read_text(encoding="utf-8")
    i = src.index("_torchao_is_newer_than_torch(e)")
    window = src[i:i + 200]
    assert "RuntimeError(_torchao_torch_mismatch_message(e))" in window


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
