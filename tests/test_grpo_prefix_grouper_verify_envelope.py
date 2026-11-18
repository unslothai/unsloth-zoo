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

"""PrefixGrouper's grad self-verify trusts a shape inside the verified (T, longest segment) envelope.
Sampled completion lengths set a new maximum on most steps, so an exact-max envelope re-ran the
verify forward (packed + shared-prefix) on about half of all steps; it is stored rounded up to a
power of two instead, so a run re-verifies once per doubling."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text()


def _store_lines():
    tree = ast.parse(SOURCE)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "grpo_accumulated_loss")
    body = ast.get_source_segment(SOURCE, fn)
    m = re.search(r"\n(\s*)_pg_vT = (.+)\n\s*_pg_vS = (.+)\n", body)
    assert m, "grad self-verify envelope store not found"
    return m.group(2), m.group(3)


class _T:
    def __init__(self, shape1, maxpos):
        self.shape = (1, shape1)
        self._m = maxpos

    def max(self):
        return self._m


class _Layout:
    def __init__(self, T, maxpos):
        self.flat_ids = _T(T, None)
        self.position_ids = _T(None, maxpos)


def _envelope(T, maxpos):
    vT, vS = _store_lines()
    ns = {"_pg_layout": _Layout(T, maxpos), "int": int}
    return eval(vT, ns), eval(vS, ns)


def test_envelope_is_rounded_up_to_a_power_of_two():
    assert _envelope(1000, 700) == (1024, 1024)
    assert _envelope(1024, 1023) == (1024, 1024)
    assert _envelope(1025, 1024) == (2048, 2048)


def test_envelope_covers_the_next_longer_sampled_batch():
    vT, vS = _envelope(4700, 2300)
    assert 4790 <= vT and 2350 <= vS  # a slightly longer next batch stays trusted
