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

"""Fused lm_head + CE coverage on the installed transformers against tests/fused_ce_coverage.json.

A class that fused on the closest recorded transformers release must still fuse. Otherwise it trains
with the full logits tensor and nothing else fails. New classes that do not fuse are reported, not failed.
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fused_ce_coverage  # noqa: E402

pytest.importorskip("transformers")
from packaging.version import Version  # noqa: E402

UPDATE_HINT = (
    "If the change is intended (a model was removed upstream or now needs a new rewrite), "
    "record it with `python tests/fused_ce_coverage.py --update` on this transformers."
)


@pytest.fixture(scope = "module")
def census():
    # A fresh process: the import hook must see every modeling import, and the patches stay out of this session.
    env = dict(os.environ, PYTHONPATH = os.pathsep.join([os.path.dirname(HERE), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, os.path.join(HERE, "fused_ce_coverage.py")],
                          capture_output = True, text = True, timeout = 900, env = env)
    lines = proc.stdout.strip().splitlines()
    assert proc.returncode == 0 and lines, f"census failed:\n{proc.stderr[-4000:]}"
    return json.loads(lines[-1])


@pytest.fixture(scope = "module")
def reference(census):
    baseline = fused_ce_coverage.load_baseline()
    installed = Version(census["transformers"])
    older = [v for v in baseline if Version(v) <= installed]
    if not older:
        pytest.skip(f"transformers {installed} predates the recorded baseline")
    version = max(older, key = Version)
    return version, baseline[version]


def test_no_class_lost_fused_ce(census, reference):
    version, expected = reference
    present = set(census["candidates"])
    lost = sorted((expected["fused"] & present) - set(census["fused"]))
    assert not lost, (
        f"{len(lost)} class(es) fused on transformers {version} but not on {census['transformers']}: "
        f"{', '.join(lost)}. They now train with the full logits tensor. {UPDATE_HINT}"
    )


def test_report_coverage_changes(census, reference):
    version, expected = reference
    new_unfused = sorted(set(census["unfused_targets"]) - expected["unfused_targets"] - expected["fused"])
    newly_fused = sorted(set(census["fused"]) - expected["fused"])
    notes = []
    if new_unfused:
        notes.append(f"not fused (new since {version}): {', '.join(new_unfused)}")
    if newly_fused:
        notes.append(f"fused but not in the baseline for {version}: {', '.join(newly_fused)}")
    if notes:
        pytest.skip("; ".join(notes) + f". {UPDATE_HINT}")
