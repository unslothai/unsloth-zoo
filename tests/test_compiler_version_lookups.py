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

"""Package versions for the generated-file header are read once per compile call.

Every generated file starts with the unsloth_zoo / unsloth / transformers / trl versions.
They used to be read from the installed metadata four times per generated file. One
`unsloth_compile_transformers` call now reads them once, and the next call reads them
again, so a package upgraded between two calls is still stamped correctly.
"""
import json
import os
import subprocess
import sys
import textwrap

_SCRIPT = textwrap.dedent(
    """
    import json, os, sys
    import unsloth_zoo.compiler as c

    real_version = c.importlib_version
    state = {"calls": 0, "trl": None}

    def fake_version(package):
        state["calls"] += 1
        if package == "trl" and state["trl"] is not None:
            return state["trl"]
        return real_version(package)

    c.importlib_version = fake_version
    cnf = c.create_new_function
    made = []

    def counting_cnf(*args, **kwargs):
        made.append(args[0])
        return cnf(*args, **kwargs)

    c.create_new_function = counting_cnf

    def header(name):
        path = os.path.join(os.environ["UNSLOTH_COMPILE_LOCATION"], name + ".py")
        with open(path, encoding = "utf-8") as f:
            return f.read().split("__UNSLOTH_VERSIONING__")[0].strip().strip('"').split()

    out = {}
    state["trl"] = "0.0.1"
    c.unsloth_compile_transformers("llama", disable = True)
    out["first"] = {"calls": state["calls"], "made": len(made), "header": header(made[-1])}
    state["calls"] = 0
    made.clear()
    state["trl"] = "0.0.2"
    # A second model type: a repeat of "llama" in one process returns before generating.
    c.unsloth_compile_transformers("qwen3", disable = True)
    out["second"] = {"calls": state["calls"], "made": len(made), "header": header(made[-1])}
    print("RESULT " + json.dumps(out))
    """
)


def _run(tmp_path):
    env = dict(os.environ)
    env["UNSLOTH_COMPILE_LOCATION"] = str(tmp_path / "cache")
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd = str(tmp_path), env = env, capture_output = True, text = True, timeout = 900,
    )
    lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT "):])


def test_versions_read_once_per_compile_call_and_again_on_the_next(tmp_path):
    out = _run(tmp_path)
    # Several generated files in one call, but only one read of each of the four packages.
    assert out["first"]["made"] > 1, out
    for key in ("first", "second"):
        assert out[key]["made"] >= 1, out
        assert out[key]["calls"] == 4, out
    # The second call sees the "upgraded" trl, so its header changes with it.
    assert out["first"]["header"][-1] == "0.0.1", out
    assert out["second"]["header"][-1] == "0.0.2", out
