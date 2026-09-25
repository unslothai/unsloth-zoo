# Regression tests for the Windows UnicodeDecodeError class in the llama.cpp
# build/run path (unslothai/unsloth#2660).
#
# On Windows the default text encoding is the locale code page (e.g. cp1252),
# not UTF-8. ``subprocess.run`` / ``subprocess.Popen`` opened in text mode
# (``text=True`` / ``universal_newlines=True``) without an explicit
# ``encoding`` therefore decode child output with cp1252. The llama.cpp
# clone/cmake/quantize/convert pipeline emits non-ASCII content -- localized
# compiler/cmake messages, progress / box-drawing glyphs, and model/tensor
# names or file paths (e.g. ``C:\\Users\\Jose\\...``). When a byte undefined
# in cp1252 appears (e.g. ``0x9d``, which sits inside the UTF-8 encoding of
# common punctuation), the read raises ``UnicodeDecodeError`` and aborts the
# build/quantize/convert.
#
# ``test_llama_cpp_subprocess_text_calls_declare_utf8_encoding`` is a
# source-level drift detector: it parses ``unsloth_zoo/llama_cpp.py`` (no
# import, so it needs neither torch nor a built llama.cpp) and fails if any
# text-mode subprocess call is missing ``encoding="utf-8"``. Red before the
# fix, green after.

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

LLAMA_CPP = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "llama_cpp.py"


def _is_subprocess_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in {"Popen", "run"}
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )


def _kw(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true(value) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


def _is_text_mode(node: ast.Call) -> bool:
    return _is_true(_kw(node, "text")) or _is_true(_kw(node, "universal_newlines"))


def _text_mode_subprocess_calls() -> list[ast.Call]:
    tree = ast.parse(LLAMA_CPP.read_text(encoding="utf-8"), filename=str(LLAMA_CPP))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_subprocess_call(node) and _is_text_mode(node)
    ]


def test_text_mode_subprocess_calls_exist():
    # Guard the guard: if llama_cpp.py stops using text-mode subprocess calls
    # the drift test would pass vacuously.
    calls = _text_mode_subprocess_calls()
    assert len(calls) >= 8, (
        f"Expected many text-mode subprocess calls in {LLAMA_CPP.name}, "
        f"found {len(calls)} -- has the file been restructured?"
    )


def test_llama_cpp_subprocess_text_calls_declare_utf8_encoding():
    """Every text-mode subprocess call in llama_cpp.py must pin encoding='utf-8'.

    Without it, reading llama.cpp build/quantize/convert output crashes on
    Windows (cp1252). Fails before the #2660 fix, passes after.
    """
    offenders = []
    for node in _text_mode_subprocess_calls():
        enc = _kw(node, "encoding")
        if not (isinstance(enc, ast.Constant) and enc.value == "utf-8"):
            offenders.append(node.lineno)

    assert not offenders, (
        "Text-mode subprocess call(s) in unsloth_zoo/llama_cpp.py missing "
        'encoding="utf-8" (UnicodeDecodeError on Windows, #2660) at line(s): '
        + ", ".join(map(str, sorted(offenders)))
    )


def test_utf8_replace_decodes_non_cp1252_subprocess_output():
    """Document the failure and the fix with a real subprocess.

    The child emits U+201D (right double quote), whose UTF-8 encoding
    ``E2 80 9D`` contains byte 0x9D -- undefined in cp1252. Decoding the raw
    bytes as cp1252 raises (the bug); the kwargs the fix adds read it cleanly.
    """
    child = (
        "import sys; "
        "sys.stdout.buffer.write(('tensor ' + chr(0x201D) + ' x\\n').encode('utf-8'))"
    )

    raw = subprocess.run([sys.executable, "-c", child], capture_output=True).stdout
    assert b"\x9d" in raw  # precondition: output carries the cp1252-undefined byte

    # Failing behaviour before the fix: cp1252 (the Windows default) cannot
    # decode this output.
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp1252")

    # Correct behaviour after the fix: the kwargs llama_cpp.py now uses.
    result = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.stdout.startswith("tensor ")
    assert "”" in result.stdout
