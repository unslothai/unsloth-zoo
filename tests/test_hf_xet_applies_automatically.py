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

"""The tuning is only worth anything if it reaches the process that does the downloading.

hf_xet reads its configuration once, natively, the first time the runtime is built. Setting the
variables after that is a no-op that leaves no trace, so "we set them" and "they took effect" are
different claims and only the second one matters. These tests make the second claim: import
unsloth_zoo the way a user does, in a clean environment, and check what the environment actually
holds afterwards and what a spawned worker inherits.

Importing unsloth_zoo is expensive, so the checks are grouped by the environment they need rather
than one subprocess per assertion.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ZOO_ROOT = str(Path(__file__).resolve().parents[1])

# Built, never inherited: this machine exports HF_XET_HIGH_PERFORMANCE=1, which is exactly the
# setting one of these tests is about, and inheriting it would make the test pass for free.
_BASE_ENV = {
    k: v for k, v in os.environ.items()
    if not k.startswith(("HF_XET_", "HF_HUB_", "UNSLOTH_"))
}
_BASE_ENV["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"


def _run(tmp_path: Path, body: str, extra_env: "dict[str, str] | None" = None) -> dict:
    """Run *body* as a FILE, not with -c: multiprocessing spawn re-imports __main__ in the child,
    which a -c script cannot provide, and the failure looks like an unrelated pickling error."""
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(body))
    env = dict(_BASE_ENV, PYTHONPATH = ZOO_ROOT + os.pathsep + _BASE_ENV.get("PYTHONPATH", ""))
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, str(script)], env = env, capture_output = True, text = True, timeout = 900,
    )
    assert proc.returncode == 0, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-3000:]}"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no result line in:\n{proc.stdout}\n{proc.stderr[-2000:]}")


@pytest.mark.timeout(1200)
def test_importing_unsloth_zoo_tunes_this_process_and_its_workers(tmp_path):
    """No flag, no call, no documentation to read: the download a user starts next is tuned, and so
    is the one a spawned worker starts, which is where downloads actually happen."""
    result = _run(tmp_path, """
        import json, multiprocessing, os, sys


        def child(queue):
            queue.put({k: v for k, v in os.environ.items() if k.startswith("HF_XET_")})


        if __name__ == "__main__":
            # If hf_xet were already built here the settings would arrive too late to be read.
            preloaded = "hf_xet" in sys.modules
            import unsloth_zoo  # noqa: F401
            from unsloth_zoo.hf_xet_tuning import xet_env_overrides

            ctx = multiprocessing.get_context("spawn")
            queue = ctx.Queue()
            proc = ctx.Process(target = child, args = (queue,))
            proc.start()
            inherited = queue.get(timeout = 600)
            proc.join(timeout = 600)

            print(json.dumps({
                "hf_xet_preloaded": preloaded,
                "seen": {k: v for k, v in os.environ.items() if k.startswith("HF_XET_")},
                "expected": {k: str(v) for k, v in xet_env_overrides(fail_fast = False).items()},
                "child": inherited,
            }))
    """)
    assert not result["hf_xet_preloaded"]
    seen, expected = result["seen"], result["expected"]
    assert expected, "the tuning resolved to nothing"
    for key, value in expected.items():
        assert seen.get(key) == value, f"{key}: import left {seen.get(key)!r}, wanted {value!r}"
        assert result["child"].get(key) == value, f"{key} did not reach the worker"


@pytest.mark.timeout(1200)
def test_a_user_who_asked_for_high_performance_still_has_it_after_import(tmp_path):
    """The end-to-end form of the setdefault rule. Overriding this cost a 192-core host 2.55x of
    its download throughput the moment it imported us."""
    result = _run(tmp_path, """
        import json, os
        import unsloth_zoo  # noqa: F401
        print(json.dumps({k: v for k, v in os.environ.items() if k.startswith("HF_XET_")}))
    """, extra_env = {"HF_XET_HIGH_PERFORMANCE": "1"})
    assert result["HF_XET_HIGH_PERFORMANCE"] == "1"
    # ...and the sizing that flag voids is stood down rather than left to fight it.
    from unsloth_zoo.hf_xet_tuning import _CAPS_VOIDED_BY_HIGH_PERFORMANCE

    for key in _CAPS_VOIDED_BY_HIGH_PERFORMANCE:
        assert key not in result, key
