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

"""The generated trainer must actually be installed, and failing is silent.

`_patch_trl_rl_trainers` and `patch_trl_rl_trainers` each swallow generation
errors, so a break raises nothing, prints nothing at default log level, and
every SFT run quietly falls back to plain `trl.SFTTrainer`.

That happened. `_maybe_compile` was added to `rl_replacements.py`, and since the
compiler copies function SOURCE verbatim into the generated module, decorator
line included, the generated module hit `NameError: name '_maybe_compile' is not
defined`. Three notebooks then failed with three unrelated-looking errors, all
downstream of trl's own SFTTrainer running instead of ours.

So this file asserts the end state -- the generated class is installed -- rather
than any particular reason it might not be.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _have_unsloth():
    """importlib rather than a sibling-checkout path, so this runs anywhere."""
    import importlib.util
    try:
        return importlib.util.find_spec("unsloth") is not None
    except Exception:
        return False


def _run(body: str, timeout: int = 900):
    """A subprocess: unsloth patches trl at import and the decision is then
    cached in sys.modules for the life of the process."""
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        {body}
    """)
    env = dict(os.environ)
    # NOT /tmp: writes are blocked in this sandbox, and a cache the compiler
    # cannot write to fails generation for an unrelated reason.
    cache = Path(os.environ.get("UNSLOTH_WORKSPACE", ROOT.parent)) / "temp" / "gen_test_cache"
    cache.mkdir(parents=True, exist_ok=True)
    env["UNSLOTH_COMPILE_LOCATION"] = str(cache)
    # conftest sets UNSLOTH_ALLOW_CPU=1 suite-wide, which makes PatchFastRL
    # early-return on purpose. Inherited here it would guarantee a plain
    # SFTTrainer and this file would "fail" on a healthy tree.
    env.pop("UNSLOTH_ALLOW_CPU", None)
    return subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=timeout, env=env)


@pytest.fixture(scope="module")
def installed():
    if not _have_unsloth():
        pytest.skip("unsloth is not installed")
    r = _run("""
        import unsloth, trl
        print("RESULT " + trl.SFTTrainer.__name__)
    """)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        pytest.skip(f"probe did not run: {r.stdout[-800:]} {r.stderr[-1500:]}")
    return line[0][len("RESULT "):].strip(), r


def test_the_generated_sft_trainer_is_what_trl_hands_out(installed):
    """This is the whole file. Everything else is detail."""
    installed, r = installed
    assert installed == "UnslothSFTTrainer", (
        f"\nSTDOUT:\n{r.stdout[-2500:]}\nSTDERR:\n{r.stderr[-2500:]}\n"
        f"trl.SFTTrainer is {installed!r}; generation failed and was swallowed. "
        f"Run unsloth.models.rl._patch_trl_rl_trainers_impl('sft_trainer') "
        f"directly to see the exception."
    )


def test_generation_raises_loudly_when_called_directly():
    """The swallow is deliberate (TRL 1.x renames classes), so the guard is that
    a direct call still surfaces the error -- the only way to debug this."""
    if not _have_unsloth():
        pytest.skip("unsloth is not installed")
    r = _run("""
        import unsloth
        from unsloth.models import rl
        print("HAS_IMPL " + str(hasattr(rl, "_patch_trl_rl_trainers_impl")))
    """)
    # _run drops UNSLOTH_ALLOW_CPU, so a CPU-only host aborts before printing.
    if "HAS_IMPL" not in r.stdout:
        pytest.skip(f"probe did not run: {r.stdout[-800:]} {r.stderr[-1500:]}")
    assert "HAS_IMPL True" in r.stdout, (r.stdout[-600:], r.stderr[-1200:])


# ---- the specific shape that broke, so it cannot come back ---------------

def test_every_name_in_a_copied_decorator_is_importable_by_the_generator():
    """Each decorator on a function whose source gets copied needs a matching
    import rule in compiler.py, or the generated module NameErrors at import."""
    import ast

    src = (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text(encoding="utf-8")
    compiler = (ROOT / "unsloth_zoo" / "compiler.py").read_text(encoding="utf-8")

    names = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            call = dec.func if isinstance(dec, ast.Call) else dec
            while isinstance(call, ast.Attribute):
                call = call.value
            if isinstance(call, ast.Name):
                names.add(call.id)

    # `torch` is in the generator's preamble; builtins need no import at all.
    names.discard("torch")
    names -= set(dir(__builtins__)) if isinstance(__builtins__, dict) is False \
        else set(__builtins__)
    import builtins
    names -= set(dir(builtins))
    missing = [n for n in sorted(names) if f'"{n}" in new_source' not in compiler]
    assert missing == [], (
        f"decorator name(s) {missing} are copied into generated modules but "
        f"compiler.py emits no import for them"
    )


def test_the_helper_lives_where_the_generated_module_can_reach_it():
    common = (ROOT / "unsloth_zoo" / "temporary_patches" / "common.py").read_text(encoding="utf-8")
    rl = (ROOT / "unsloth_zoo" / "rl_replacements.py").read_text(encoding="utf-8")
    assert "def _maybe_compile(" in common
    assert "def _maybe_compile(" not in rl, "defining it here is what broke generation"
    assert "_maybe_compile," in rl, "rl_replacements must import it"


def test_the_generator_emits_the_import():
    compiler = (ROOT / "unsloth_zoo" / "compiler.py").read_text(encoding="utf-8")
    assert '"_maybe_compile" in new_source' in compiler
    assert "import _maybe_compile" in compiler


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
