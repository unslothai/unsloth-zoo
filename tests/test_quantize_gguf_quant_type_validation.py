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

"""quant_type reaches a shell=True command line, so it must be a bare token.

`quantization_method` is a user facing argument all the way from
`save_pretrained_gguf(..., quantization_method=...)` down to
`quantize_gguf(quant_type=...)`, and the command is assembled as a string.
Every legitimate value (unsloth's ALLOWED_QUANTS / IMATRIX_QUANTS, and the MLX
quant_map) is `[a-z0-9_]+`, so anything carrying shell metacharacters is
rejected outright instead of being interpolated.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODULE = None


def _load_llama_cpp_module():
    """Loaded once per session: re-executing llama_cpp for every test churns
    global torch state and slows the suite down for no extra coverage."""
    global _MODULE
    if _MODULE is None:
        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
        spec = importlib.util.spec_from_file_location("llama_cpp_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE


def _install_fake_subprocess_run(monkeypatch, llama_cpp):
    captured: dict[str, object] = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="ok", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    return captured


def _stub_output_exists(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_size=4096))


MALICIOUS = [
    "q4_k_m; touch pwned; #",
    "q4_k_m && curl http://127.0.0.1/x",
    "q4_k_m`id`",
    "q4_k_m $(id)",
    "q4_k_m | sh",
    "",
    "q4 k m",
]


@pytest.mark.parametrize("quant_type", MALICIOUS)
def test_shell_metacharacters_are_rejected(monkeypatch, quant_type):
    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    with pytest.raises(ValueError):
        llama_cpp.quantize_gguf(
            input_gguf="/tmp/in.gguf",
            output_gguf="/tmp/out.gguf",
            quant_type=quant_type,
            quantizer_location="/usr/bin/llama-quantize",
            n_threads=4,
            print_output=False,
        )
    assert "cmd" not in captured, "subprocess must not run for a rejected quant_type"


def test_non_string_quant_type_is_rejected(monkeypatch):
    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    with pytest.raises(ValueError):
        llama_cpp.quantize_gguf(
            input_gguf="/tmp/in.gguf",
            output_gguf="/tmp/out.gguf",
            quant_type=None,
            quantizer_location="/usr/bin/llama-quantize",
            n_threads=4,
            print_output=False,
        )
    assert "cmd" not in captured


@pytest.mark.parametrize(
    "quant_type",
    ["q4_k_m", "q8_0", "bf16", "f16", "f32", "iq4_xs", "iq1_m", "q6_k", "Q4_K_M"],
)
def test_legitimate_quant_types_still_run(monkeypatch, quant_type):
    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    llama_cpp.quantize_gguf(
        input_gguf="/tmp/in.gguf",
        output_gguf="/tmp/out.gguf",
        quant_type=quant_type,
        quantizer_location="/usr/bin/llama-quantize",
        n_threads=4,
        print_output=False,
    )

    cmd = captured["cmd"]
    assert cmd == f"/usr/bin/llama-quantize /tmp/in.gguf /tmp/out.gguf {quant_type} 4"


def test_surrounding_whitespace_is_trimmed_not_rejected(monkeypatch):
    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    llama_cpp.quantize_gguf(
        input_gguf="/tmp/in.gguf",
        output_gguf="/tmp/out.gguf",
        quant_type="  q4_k_m  ",
        quantizer_location="/usr/bin/llama-quantize",
        n_threads=4,
        print_output=False,
    )
    assert captured["cmd"].endswith("q4_k_m 4")


@pytest.mark.parametrize("is_windows", [False, True])
def test_command_is_byte_identical_across_platforms(monkeypatch, is_windows):
    """The validated token is inserted bare, so cmd.exe and every POSIX shell
    see exactly the command previous releases produced."""
    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", is_windows)

    if is_windows:
        quantizer = "C:\\llama\\llama-quantize.exe"
        src, dst = "C:\\models\\m.BF16.gguf", "C:\\models\\m.Q4_K_M.gguf"
        expected = f'"{quantizer}" "{src}" "{dst}" q4_k_m 6'
    else:
        quantizer = "/usr/bin/llama-quantize"
        src, dst = "/models/m.BF16.gguf", "/models/m.Q4_K_M.gguf"
        expected = f"{quantizer} {src} {dst} q4_k_m 6"

    llama_cpp.quantize_gguf(input_gguf=src, output_gguf=dst, quant_type="q4_k_m",
                            quantizer_location=quantizer, n_threads=6, print_output=False)
    assert captured["cmd"] == expected


def test_n_threads_is_coerced_to_int(monkeypatch):
    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    llama_cpp.quantize_gguf(
        input_gguf="/tmp/in.gguf",
        output_gguf="/tmp/out.gguf",
        quant_type="q4_k_m",
        quantizer_location="/usr/bin/llama-quantize",
        n_threads="8",
        print_output=False,
    )
    assert captured["cmd"].endswith(" 8")
