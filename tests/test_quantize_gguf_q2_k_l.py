"""Q2_K_L preset dispatch inside quantize_gguf.

Q2_K_L is an Unsloth-side preset, not a native llama.cpp ftype. Recipe:

  llama-quantize \\
      --tensor-type ffn_down=Q3_K \\
      --output-tensor-type Q6_K \\
      --token-embedding-type Q4_K \\
      IN OUT q2_k NTHREADS

``--tensor-type ffn_down=Q3_K`` uses llama-quantize substring matching to
cover both ``ffn_down.weight`` (dense FFN) and ``ffn_down_exps.weight``
(MoE experts) across every block id without needing per-layer regex.
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


def test_q2_k_l_expands_to_q2_k_with_dynamic_recipe(monkeypatch):
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
    # All three preset flags must appear, in any order.
    assert "--output-tensor-type Q6_K" in cmd, f"--output-tensor-type Q6_K missing: {cmd!r}"
    assert "--token-embedding-type Q4_K" in cmd, f"--token-embedding-type Q4_K missing: {cmd!r}"
    assert "--tensor-type ffn_down=Q3_K" in cmd, f"--tensor-type ffn_down=Q3_K missing: {cmd!r}"
    # llama-quantize parses option flags before the positional model paths --
    # the new --tensor-type flag must land in the option region.
    assert cmd.index("--tensor-type ffn_down=Q3_K") < cmd.index("/tmp/in.gguf"), (
        f"--tensor-type must precede positional args: {cmd!r}"
    )
    # Sanity: input/output paths and thread count are still present.
    assert "/tmp/in.gguf" in cmd
    assert "/tmp/out.gguf" in cmd
    assert " 4" in cmd, f"n_threads missing: {cmd!r}"


def test_q2_k_l_ffn_down_pattern_covers_dense_and_moe(monkeypatch):
    """`--tensor-type ffn_down=Q3_K` uses llama-quantize substring matching, so a
    single pattern hits both `blk.N.ffn_down.weight` (dense) and
    `blk.N.ffn_down_exps.weight` (MoE experts) for every layer id without
    needing per-layer or per-architecture regex. Document that here so future
    refactors do not split the pattern back into per-layer overrides."""

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
    # One pattern, no per-layer enumeration. The bare substring "ffn_down"
    # matches both "ffn_down.weight" and "ffn_down_exps.weight" because
    # llama-quantize uses std::string::find on the pattern.
    assert cmd.count("--tensor-type ffn_down=Q3_K") == 1, (
        f"expected exactly one ffn_down override; got: {cmd!r}"
    )
    assert "blk." not in cmd, f"per-layer enumeration leaked into command: {cmd!r}"


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
        assert "--output-tensor-type Q6_K" in cmd
        assert "--token-embedding-type Q4_K" in cmd
        assert "--tensor-type ffn_down=Q3_K" in cmd


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
        assert "--tensor-type" not in cmd, (
            f"ftype {ftype!r} accidentally picked up tensor-type override: {cmd!r}"
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
    # Recipe details should appear in the log so users can audit what landed.
    assert "ffn_down" in out, out
    assert "Q6_K" in out, out
    assert "Q4_K" in out, out
    assert "Q3_K" in out, out


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
