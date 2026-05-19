"""Q2_K_L preset dispatch inside quantize_gguf.

Q2_K_L is an Unsloth-side preset, not a native llama.cpp ftype. It maps to
``llama-quantize --output-tensor-type q8_0 --token-embedding-type q8_0 IN OUT
q2_k NTHREADS``. Before this fix, the MLX/Studio export path forwarded the raw
``q2_k_l`` string to llama-quantize which then aborted with
``main: invalid ftype 'q2_k_l'``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_subprocess_run(monkeypatch, llama_cpp):
    """Replace subprocess.run with a capturing fake (no real binary invoked)."""

    captured: dict[str, object] = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="ok", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    return captured


def _stub_output_exists(monkeypatch):
    """Pretend the output file was produced (no real quantization happened)."""

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_size=4096))


def test_q2_k_l_expands_to_q2_k_with_output_and_embedding_q8_0(monkeypatch):
    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    llama_cpp.quantize_gguf(
        input_gguf="/tmp/in.gguf",
        output_gguf="/tmp/out.gguf",
        quant_type="q2_k_l",
        quantizer_location="/usr/bin/llama-quantize",
        n_threads=4,
        print_output=False,
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, str), f"command should be a shell string (existing convention); got {type(cmd)!r}"
    # The literal preset name must NOT reach llama-quantize.
    assert "q2_k_l" not in cmd, f"q2_k_l leaked into llama-quantize command: {cmd!r}"
    # The expanded ftype must appear, as a standalone token.
    assert " q2_k " in cmd, f"q2_k token missing: {cmd!r}"
    # Both preset flags must appear, in either order.
    assert "--output-tensor-type q8_0" in cmd, f"--output-tensor-type q8_0 missing: {cmd!r}"
    assert "--token-embedding-type q8_0" in cmd, f"--token-embedding-type q8_0 missing: {cmd!r}"
    # Sanity: input/output paths and thread count are still present.
    assert "/tmp/in.gguf" in cmd
    assert "/tmp/out.gguf" in cmd
    assert " 4" in cmd, f"n_threads missing: {cmd!r}"


def test_q2_k_l_is_case_insensitive(monkeypatch):
    """Studio frontend may send Q2_K_L / Q2_k_L / etc. Treat them identically."""

    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    for variant in ("Q2_K_L", "q2_K_L", "  q2_k_l  "):
        captured.clear()
        llama_cpp.quantize_gguf(
            input_gguf="/tmp/in.gguf",
            output_gguf="/tmp/out.gguf",
            quant_type=variant,
            quantizer_location="/usr/bin/llama-quantize",
            n_threads=4,
            print_output=False,
        )
        cmd = captured["cmd"]
        assert " q2_k " in cmd, f"variant {variant!r}: expansion missing: {cmd!r}"
        assert "--output-tensor-type q8_0" in cmd
        assert "--token-embedding-type q8_0" in cmd


def test_other_quant_types_are_untouched(monkeypatch):
    """Non-preset ftypes must traverse the original code path byte-for-byte.

    Linux + Windows non-regression: ensures the q2_k_l branch does not affect
    any other ftype. q3_k_l is a real llama.cpp ftype distinct from q2_k_l and
    must be passed through verbatim.
    """

    llama_cpp = _load_llama_cpp_module()
    captured = _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    for ftype in (
        "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",  # q3_k_l is a real ftype, NOT a preset
        "q4_0", "q4_1", "q4_k_s", "q4_k_m",
        "q5_0", "q5_1", "q5_k_s", "q5_k_m",
        "q6_k", "q8_0", "bf16", "f16", "f32",
    ):
        captured.clear()
        llama_cpp.quantize_gguf(
            input_gguf="/tmp/in.gguf",
            output_gguf="/tmp/out.gguf",
            quant_type=ftype,
            quantizer_location="/usr/bin/llama-quantize",
            n_threads=4,
            print_output=False,
        )
        cmd = captured["cmd"]
        assert f" {ftype} " in cmd, f"ftype {ftype!r} not preserved: {cmd!r}"
        assert "--output-tensor-type" not in cmd, (
            f"ftype {ftype!r} accidentally picked up preset flags: {cmd!r}"
        )
        assert "--token-embedding-type" not in cmd, (
            f"ftype {ftype!r} accidentally picked up preset flags: {cmd!r}"
        )


def test_q2_k_l_print_output_path_logs_preset_expansion(capsys, monkeypatch):
    """When print_output=True the user sees both the original request and the expansion."""

    llama_cpp = _load_llama_cpp_module()
    _install_fake_subprocess_run(monkeypatch, llama_cpp)
    _stub_output_exists(monkeypatch)

    llama_cpp.quantize_gguf(
        input_gguf="/tmp/in.gguf",
        output_gguf="/tmp/out.gguf",
        quant_type="q2_k_l",
        quantizer_location="/usr/bin/llama-quantize",
        n_threads=4,
        print_output=True,
    )

    out = capsys.readouterr().out
    assert "Quantizing to q2_k_l" in out, out
    assert "Expanding Q2_K_L preset" in out, out


def test_q2_k_l_error_message_keeps_original_preset_name(monkeypatch):
    """If llama-quantize fails, the RuntimeError should mention q2_k_l (what the
    user asked for) rather than q2_k (the rewritten internal ftype)."""

    llama_cpp = _load_llama_cpp_module()

    def failing_run(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="boom")

    monkeypatch.setattr(subprocess, "run", failing_run)
    monkeypatch.setattr(llama_cpp.subprocess, "run", failing_run)

    try:
        llama_cpp.quantize_gguf(
            input_gguf="/tmp/in.gguf",
            output_gguf="/tmp/out.gguf",
            quant_type="q2_k_l",
            quantizer_location="/usr/bin/llama-quantize",
            n_threads=4,
            print_output=False,
        )
    except RuntimeError as exc:
        assert "q2_k_l" in str(exc), f"error msg should keep preset name: {exc}"
    else:
        raise AssertionError("expected RuntimeError")
