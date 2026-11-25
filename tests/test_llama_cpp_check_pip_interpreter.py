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

"""check_pip() must install into the running interpreter: a uv venv has no pip, and the `pip` on PATH is then the system one."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import sysconfig

import pytest


llama_cpp = importlib.import_module("unsloth_zoo.llama_cpp")

_HERE = sysconfig.get_paths()["purelib"]
_FOREIGN = os.path.join(os.path.abspath(os.sep), "usr", "lib", "python3", "dist-packages")
_NOT_FOUND = "No module named pip"


def _version_line(site_packages):
    return f"pip 24.0 from {os.path.join(site_packages, 'pip')} (python 3.12)\n"


class _FakeShell:
    """Answers check_pip's probes; anything not listed is "command not found"."""

    def __init__(self, answers, ensurepip_ok = False):
        self.answers = dict(answers)
        self.ensurepip_ok = ensurepip_ok
        self.ran = []

    def __call__(self, command, *args, **kwargs):
        self.ran.append(command)
        if isinstance(command, list):
            if command[1:3] == ["-m", "ensurepip"]:
                if self.ensurepip_ok:
                    self.answers[f"{llama_cpp._RUNNING_PIP} --version"] = (0, _version_line(_HERE))
                return subprocess.CompletedProcess(command, 0 if self.ensurepip_ok else 1, stdout = "")
            # The pip probe after ensurepip: the same interpreter as _RUNNING_PIP.
            command = f"{llama_cpp._RUNNING_PIP} {' '.join(command[3:])}"
        code, out = self.answers.get(command, (127, f"sh: 1: {command.split()[0]}: not found\n"))
        return subprocess.CompletedProcess(command, code, stdout = out)


@pytest.fixture
def uv_venv_without_uv(monkeypatch):
    # The running interpreter has no pip and uv is not on PATH; the system pip is.
    shell = _FakeShell({
        f"{llama_cpp._RUNNING_PIP} --version": (1, f"{sys.executable}: {_NOT_FOUND}\n"),
        "pip --version": (0, _version_line(_FOREIGN)),
        "pip3 --version": (0, _version_line(_FOREIGN)),
        "python3 -m pip --version": (0, _version_line(_FOREIGN)),
    })
    monkeypatch.setattr(llama_cpp.subprocess, "run", shell)
    return shell


def test_pip_location_is_matched_against_this_interpreter():
    assert llama_cpp._pip_is_for_running_python(_version_line(_HERE))
    assert not llama_cpp._pip_is_for_running_python(_version_line(_FOREIGN))
    # Output check_pip cannot parse keeps the old behaviour.
    assert llama_cpp._pip_is_for_running_python("pip 24.0\n")


def test_system_pip_is_not_picked_for_a_venv_without_pip(uv_venv_without_uv):
    uv_venv_without_uv.ensurepip_ok = True
    assert llama_cpp.check_pip() == llama_cpp._RUNNING_PIP
    assert [sys.executable, "-m", "ensurepip", "--upgrade"] in uv_venv_without_uv.ran


def test_no_usable_pip_still_fails_loudly(uv_venv_without_uv):
    with pytest.raises(RuntimeError, match = "Tried all of"):
        llama_cpp.check_pip()


def test_running_interpreter_pip_is_kept(monkeypatch):
    shell = _FakeShell({f"{llama_cpp._RUNNING_PIP} --version": (0, _version_line(_HERE))})
    monkeypatch.setattr(llama_cpp.subprocess, "run", shell)
    assert llama_cpp.check_pip() == llama_cpp._RUNNING_PIP
    assert len(shell.ran) == 1


def test_pip_on_path_for_this_interpreter_is_kept(monkeypatch):
    shell = _FakeShell({
        f"{llama_cpp._RUNNING_PIP} --version": (1, f"{sys.executable}: {_NOT_FOUND}\n"),
        "pip --version": (0, _version_line(_HERE)),
    })
    monkeypatch.setattr(llama_cpp.subprocess, "run", shell)
    assert llama_cpp.check_pip() == "pip"


def test_uv_pip_installs_into_the_running_interpreter(monkeypatch):
    shell = _FakeShell({
        f"{llama_cpp._RUNNING_PIP} --version": (1, f"{sys.executable}: {_NOT_FOUND}\n"),
        "uv pip --help": (0, "Manage Python packages with a pip-compatible interface\n"),
    })
    monkeypatch.setattr(llama_cpp.subprocess, "run", shell)
    pip = llama_cpp.check_pip()
    assert pip == "uv pip"
    # Without --python, uv installs into $VIRTUAL_ENV, or refuses when none is activated.
    assert llama_cpp._pip_install_command(pip, "gguf") == f'uv pip install --python "{sys.executable}" gguf'
    assert llama_cpp._pip_install_command("pip", "gguf") == "pip install gguf"
