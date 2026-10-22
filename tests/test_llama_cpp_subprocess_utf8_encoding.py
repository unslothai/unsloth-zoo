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

# Text-mode subprocess calls that omit an explicit codec decode with the locale code page
# (cp1252 on Windows) and crash on llama.cpp build / convert output (unslothai/unsloth#2660).

import ast
from pathlib import Path

LLAMA_CPP = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "llama_cpp.py"


def _kw(node, name):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true(value):
    return isinstance(value, ast.Constant) and value.value is True


def _text_mode_subprocess_calls():
    tree = ast.parse(LLAMA_CPP.read_text(encoding = "utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and (_is_true(_kw(node, "text")) or _is_true(_kw(node, "universal_newlines")))
    ]


def test_llama_cpp_text_mode_subprocess_calls_pin_utf8():
    calls = _text_mode_subprocess_calls()
    assert len(calls) >= 8, f"only {len(calls)} text-mode subprocess calls found"
    offenders = [
        node.lineno
        for node in calls
        if not (isinstance(_kw(node, "encoding"), ast.Constant) and _kw(node, "encoding").value == "utf-8")
    ]
    assert not offenders, f'text-mode subprocess calls missing encoding="utf-8" at lines {offenders}'


def test_try_execute_reads_non_locale_utf8_output():
    # U+201D encodes as E2 80 9D; 0x9D is undefined in cp1252, so a locale decode raises.
    import sys
    import unsloth_zoo.llama_cpp as llama_cpp

    child = "import sys; sys.stdout.buffer.write(('tensor ' + chr(0x201D) + ' ok' + chr(10)).encode('utf-8'))"
    lines = []
    llama_cpp.try_execute(f'"{sys.executable}" -c "{child}"', print_outputs = lines)
    assert "".join(lines) == "tensor ” ok\n"
