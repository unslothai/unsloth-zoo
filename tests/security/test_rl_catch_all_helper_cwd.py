# The catch-all strategy helper is a fresh `python -c` interpreter; `-c` puts the working directory first on sys.path,
# so a pickle.py in a shared cwd (e.g. /tmp) ran as the user even when the parent process itself never trusted cwd.
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ZOO_ROOT = Path(__file__).resolve().parents[2]


def test_helper_does_not_import_from_cwd(tmp_path):
    planted = tmp_path / "planted"
    planted.mkdir()
    marker = tmp_path / "marker"
    # pickle is imported by rl_environments and is not loaded at interpreter startup.
    (planted / "pickle.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\nfrom _pickle import *\n")
    driver = tmp_path / "driver"
    driver.mkdir()
    script = driver / "run.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ZOO_ROOT)!r})
        from unsloth_zoo import rl_environments as R
        print("helper", R._ensure_catch_all_helper())
        R._stop_catch_all_helper()
    """))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONSAFEPATH", "PYTHONPATH")}
    result = subprocess.run(
        [sys.executable, str(script)], cwd = planted, env = env, capture_output = True, text = True, timeout = 180,
    )
    assert "helper True" in result.stdout, result.stdout + result.stderr
    assert not marker.exists(), "catch-all helper imported a module from the working directory"
