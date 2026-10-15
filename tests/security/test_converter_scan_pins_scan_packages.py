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

"""Pin `unsloth_zoo/converter_scan.py` to `scripts/scan_packages.py`.

`scripts/` is excluded from the wheel by pyproject, so the GGUF export path
cannot import the canonical scanner at runtime and carries a vendored copy of
the patterns instead. Two copies of a ruleset drift, so this file pins them.

What these tests guarantee:

  * every pattern vendored into `converter_scan` is byte-identical to the
    same-named pattern in `scan_packages`, flags included, and
  * every `RE_*` in `scan_packages` is accounted for in `converter_scan`, either
    as vendored or as an explicit, reasoned omission, so adding a pattern to the
    canonical scanner fails here until someone decides about the converter.

What they do NOT guarantee: that the two produce the same findings. The vendored
scanner ports the CRITICAL and HIGH tiers of `check_py_file` only, and adds one
converter-specific rule for the argparse defaults `llama_cpp.py` eval()s. The
decision logic is therefore deliberately narrower than canonical and is covered
by `tests/test_llama_cpp_converter_scan.py`, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import scan_packages as canonical  # noqa: E402


def _load_converter_scan():
    """Load the vendored scanner by path.

    Importing `unsloth_zoo.converter_scan` would run the package's import-time
    device detection; the module itself needs nothing but the stdlib.
    """
    path = REPO_ROOT / "unsloth_zoo" / "converter_scan.py"
    spec = importlib.util.spec_from_file_location("converter_scan_under_test_pin", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vendored = _load_converter_scan()


def _canonical_pattern_names():
    return {name for name in dir(canonical) if name.startswith("RE_")}


def test_every_vendored_pattern_exists_in_canonical():
    missing = sorted(set(vendored.VENDORED_PATTERNS) - _canonical_pattern_names())
    assert not missing, f"vendored patterns absent from scripts/scan_packages.py: {missing}"


@pytest.mark.parametrize("name", sorted(vendored.VENDORED_PATTERNS))
def test_vendored_pattern_is_byte_identical(name):
    theirs = getattr(canonical, name)
    ours = vendored.VENDORED_PATTERNS[name]
    assert ours.pattern == theirs.pattern, (
        f"{name} drifted from scripts/scan_packages.py. Update the canonical "
        f"scanner first, then copy the pattern into unsloth_zoo/converter_scan.py."
    )
    assert ours.flags == theirs.flags, f"{name} flags drifted: {ours.flags} != {theirs.flags}"


def test_every_canonical_pattern_is_accounted_for():
    """A new RE_* in the canonical scanner must be vendored or refused by name."""
    accounted = set(vendored.VENDORED_PATTERNS) | set(vendored.PATTERNS_NOT_VENDORED)
    unaccounted = sorted(_canonical_pattern_names() - accounted)
    assert not unaccounted, (
        f"scripts/scan_packages.py grew patterns that unsloth_zoo/converter_scan.py "
        f"neither vendors nor lists in PATTERNS_NOT_VENDORED: {unaccounted}"
    )


def test_omissions_are_still_real_patterns():
    """Guards the other direction: a removed canonical pattern must not linger."""
    stale = sorted(set(vendored.PATTERNS_NOT_VENDORED) - _canonical_pattern_names())
    assert not stale, f"PATTERNS_NOT_VENDORED names patterns canonical no longer has: {stale}"


def test_omissions_carry_a_reason():
    for name, reason in vendored.PATTERNS_NOT_VENDORED.items():
        assert reason and reason.strip(), f"{name} is omitted with no reason given"


def test_no_pattern_is_both_vendored_and_omitted():
    overlap = sorted(set(vendored.VENDORED_PATTERNS) & set(vendored.PATTERNS_NOT_VENDORED))
    assert not overlap, overlap


def test_vendored_set_covers_what_check_py_file_consumes():
    """The vendored subset is defined as "what check_py_file reads". Pin that.

    Parses the canonical `check_py_file` body for the RE_* names it touches and
    requires the vendored registry to hold every one of them, so a new pattern
    wired into the .py checker cannot be quietly skipped on the export path.
    """
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(canonical.check_py_file))
    used = {
        node.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name) and node.id.startswith("RE_")
    }
    assert used, "failed to parse RE_* names out of check_py_file"
    missing = sorted(used - set(vendored.VENDORED_PATTERNS))
    assert not missing, (
        f"check_py_file consumes patterns the export-path scanner does not vendor: {missing}"
    )
