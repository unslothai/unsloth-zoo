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

"""internal_scripts_dir_pin holds its lock for the whole conversion, so it has to be
reentrant.

A caller that pins around save_pretrained_gguf (Unsloth Studio's GGUF export does) has the
MLX save path in this package entering the same pin inside that call. With a plain
threading.Lock the second entry blocks on a lock its own thread is holding and the export
never finishes. The nested case is also the only way the "a pin already in the environment
is the user's" branch is reached on one thread, so it needs to run.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(module_name, relative_path):
    """Load a module by path, without importing the unsloth_zoo package."""
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_with_deadline(target, seconds = 20.0):
    """Run target in a thread so a deadlock is a failed assert, not a hung test run."""
    outcome = {}

    def _wrapped():
        try:
            target()
            outcome["done"] = True
        except BaseException as error:  # noqa: BLE001
            outcome["error"] = error

    thread = threading.Thread(target = _wrapped, daemon = True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "internal_scripts_dir_pin deadlocked when entered twice"
    if "error" in outcome:
        raise outcome["error"]
    assert outcome.get("done")


def test_nested_pin_completes(tmp_path, monkeypatch):
    llama_cpp = _load("llama_cpp_reentrant_probe", "unsloth_zoo/llama_cpp.py")

    installed = tmp_path / "llama.cpp"
    installed.mkdir()
    script = installed / "convert_hf_to_gguf.py"
    script.write_text("import gguf\n", encoding = "utf-8")
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    seen = {}

    def _pin_twice():
        with llama_cpp.internal_scripts_dir_pin(str(installed)):
            with llama_cpp.internal_scripts_dir_pin(str(installed)):
                seen["inner_env"] = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
                # Still Unsloth's routing, so still no strict-scan exemption.
                seen["inner_trusted"] = llama_cpp._converter_is_trusted_local(str(script))
            seen["outer_env"] = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
            seen["outer_trusted"] = llama_cpp._converter_is_trusted_local(str(script))

    _run_with_deadline(_pin_twice)

    assert seen["inner_env"] == str(installed)
    assert seen["outer_env"] == str(installed)
    assert seen["inner_trusted"] is False
    assert seen["outer_trusted"] is False
    # The inner pin must not take the outer one down with it when it exits.
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" not in os.environ


def test_nested_pin_leaves_a_user_pin_alone(tmp_path, monkeypatch):
    llama_cpp = _load("llama_cpp_reentrant_user_pin_probe", "unsloth_zoo/llama_cpp.py")

    chosen = tmp_path / "my-llama.cpp"
    chosen.mkdir()
    script = chosen / "convert_hf_to_gguf.py"
    script.write_text("import gguf\n", encoding = "utf-8")
    elsewhere = tmp_path / "auto-installed"
    elsewhere.mkdir()
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", str(chosen))
    seen = {}

    def _pin_twice():
        with llama_cpp.internal_scripts_dir_pin(str(elsewhere)):
            with llama_cpp.internal_scripts_dir_pin(str(elsewhere)):
                seen["env"] = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
                seen["trusted"] = llama_cpp._converter_is_trusted_local(str(script))

    _run_with_deadline(_pin_twice)

    assert seen["env"] == str(chosen)
    assert seen["trusted"] is True
    assert os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] == str(chosen)


def test_a_second_thread_still_waits_its_turn(tmp_path, monkeypatch):
    """Reentrant for the thread holding it, still exclusive for everyone else: two
    conversions must not interleave their edits to one process-wide variable."""
    llama_cpp = _load("llama_cpp_reentrant_exclusion_probe", "unsloth_zoo/llama_cpp.py")

    installed = tmp_path / "llama.cpp"
    installed.mkdir()
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", raising = False)
    inside = threading.Event()
    entered_second = threading.Event()

    def _second():
        with llama_cpp.internal_scripts_dir_pin(str(installed)):
            entered_second.set()

    with llama_cpp.internal_scripts_dir_pin(str(installed)):
        inside.set()
        other = threading.Thread(target = _second, daemon = True)
        other.start()
        assert not entered_second.wait(1.0), "another thread entered a held pin"
    other.join(20)
    assert entered_second.is_set()
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" not in os.environ
