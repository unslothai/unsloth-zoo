"""Qwen4Exp (Qwen3.8-Flash-Next) gated DeltaNet takes the vendored fla kernels even when its
modeling module was imported before Unsloth.

transformers >= 5.15 resolves fla once, when `@use_kernel_func_from_hub_with_fallback`
decorates `torch_chunk_gated_delta_rule`. Imported after unsloth_zoo the closure binds the
vendored kernel; imported before, it froze the pure-torch fallback (25.9 ms vs 3.7 ms fwd+bwd
at B=1 T=512 H=48 D=128, 65 ms vs 2.0 ms at B=2 T=2048 on a B200) and
_repair_kernel_hub_closures only re-resolved qwen3_5 / qwen3_5_moe / qwen3_next."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ZOO_ROOT = Path(__file__).resolve().parents[1]

_SUB = textwrap.dedent(
    """
    import sys
    import transformers.models.qwen4_exp.modeling_qwen4_exp as m  # BEFORE unsloth_zoo
    from transformers.integrations import hub_kernels
    if not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback"):
        print("SKIP_NO_KERNEL_HUB"); raise SystemExit(0)
    def resolved(wrapper):
        for cell in wrapper.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value):
                return value
    assert "unsloth_zoo" not in sys.modules
    before = resolved(m.torch_chunk_gated_delta_rule)
    assert getattr(before, "__module__", "").startswith("transformers."), before
    import unsloth_zoo  # noqa: F401  (runs patch_vendor_fla at import)
    from unsloth_zoo.temporary_patches.fla_vendor import _resolved_implementation, patch_vendor_fla
    patch_vendor_fla()
    fla = sys.modules.get("fla")
    if fla is None or not getattr(fla, "_UNSLOTH_VENDORED_FLA", False):
        print("SKIP_NOT_INJECTED"); raise SystemExit(0)
    live = sys.modules["fla.ops.gated_delta_rule"]
    assert _resolved_implementation(m.torch_chunk_gated_delta_rule) is live.chunk_gated_delta_rule
    assert _resolved_implementation(m.torch_recurrent_gated_delta_rule) is live.recurrent_gated_delta_rule
    print("QWEN4_EXP_REPAIRED_OK")
    """
)


def test_qwen4_exp_imported_first_gets_vendored_kernels():
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for k in ("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "UNSLOTH_DISABLE_VENDORED_FLA"):
        env.pop(k, None)
    proc = subprocess.run([sys.executable, "-c", _SUB], env=env, capture_output=True, text=True)
    if "SKIP_" in proc.stdout:
        pytest.skip(proc.stdout.strip())
    assert "QWEN4_EXP_REPAIRED_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-3000:]}"


def test_qwen4_exp_kill_switch_keeps_torch():
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["UNSLOTH_DISABLE_VENDORED_FLA"] = "1"
    env.pop("UNSLOTH_VENDORED_FLA_NO_AUTORUN", None)
    code = textwrap.dedent(
        """
        import importlib.util, sys
        if importlib.util.find_spec("fla") is not None and "unsloth_zoo" not in (importlib.util.find_spec("fla").origin or ""):
            print("SKIP_REAL_FLA_INSTALLED"); raise SystemExit(0)
        import unsloth_zoo  # noqa: F401
        import transformers.models.qwen4_exp.modeling_qwen4_exp as m
        from unsloth_zoo.temporary_patches.fla_vendor import _resolved_implementation
        impl = _resolved_implementation(m.torch_chunk_gated_delta_rule)
        assert getattr(impl, "__module__", "").startswith("transformers."), impl
        print("KILL_SWITCH_OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if "SKIP_" in proc.stdout:
        pytest.skip(proc.stdout.strip())
    assert "KILL_SWITCH_OK" in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-3000:]}"
