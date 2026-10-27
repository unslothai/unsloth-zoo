"""conftest's compile_enabled turns the kill switch off for one test and puts every frozen flag back.

Run in a fresh interpreter so the zoo modules are first imported inside the fixture, the order in
which a module could freeze its flags compile-on and keep them after teardown.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_CHILD = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, {tests!r})
    import pytest
    import conftest
    # conftest itself imports parts of unsloth_zoo; drop them so the fixture does the first import.
    for name in [m for m in sys.modules if m == "unsloth_zoo" or m.startswith("unsloth_zoo.")]:
        del sys.modules[name]
    mp = pytest.MonkeyPatch()
    conftest._enable_compile(mp)
    from unsloth_zoo.temporary_patches import common
    from unsloth_zoo.fused_losses import chunked_logprob
    during = (os.environ.get("UNSLOTH_COMPILE_DISABLE"), common.UNSLOTH_COMPILE_DISABLE, chunked_logprob._COMPILE)
    mp.undo()
    after = (os.environ.get("UNSLOTH_COMPILE_DISABLE"), common.UNSLOTH_COMPILE_DISABLE, chunked_logprob._COMPILE)
    print(repr((during, after)))
    """
)


def test_the_kill_switch_and_its_frozen_flags_come_back():
    env = dict(os.environ, UNSLOTH_COMPILE_DISABLE = "1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), env.get("PYTHONPATH")) if p)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD.format(tests = str(ROOT / "tests"))],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
        cwd = str(ROOT),
    )
    assert out.returncode == 0, out.stderr[-2000:]
    during, after = eval(out.stdout.strip().splitlines()[-1])
    assert during == (None, False, True)
    assert after == ("1", True, False)
