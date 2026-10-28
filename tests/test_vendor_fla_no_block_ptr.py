# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""Vendored fla must not use Triton block pointers, which triton main removed (triton-lang/triton#10833)."""

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
os.environ.setdefault("UNSLOTH_VENDORED_FLA_NO_AUTORUN", "1")

try:
    from unsloth_zoo.temporary_patches import fla_vendor
except ImportError as _e:
    pytest.skip(f"unsloth_zoo unavailable: {_e}", allow_module_level=True)

ZOO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = ZOO_ROOT / "unsloth_zoo" / "_vendored" / "fla"


def test_no_block_pointer_calls():
    hits = []
    for path in sorted(VENDORED.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in ("make_block_ptr", "advance"):
                hits.append(f"{path.relative_to(VENDORED)}:{node.lineno} {node.attr}")
    assert not hits, hits


def test_autotune_keys_are_kernel_arguments():
    # triton main rejects an autotune key that is not a kernel argument at import, which broke injection of the whole
    # vendored tree; upstream fla 1330 dropped the stale BT key.
    bad = []
    for path in sorted(VENDORED.rglob("*.py")):
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, ast.FunctionDef):
                continue
            args = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
            for deco in fn.decorator_list:
                if not isinstance(deco, ast.Call):
                    continue
                for kw in deco.keywords:
                    if kw.arg == "key" and isinstance(kw.value, ast.List):
                        bad += [
                            f"{path.relative_to(VENDORED)}:{deco.lineno} {fn.name} {e.value!r}"
                            for e in kw.value.elts
                            if isinstance(e, ast.Constant) and e.value not in args
                        ]
    assert not bad, bad


def _free_gib():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.mem_get_info()[0] / 2**30
    except Exception:
        return 0.0


# One sequence whose in-sequence offsets (t * H * K) pass 2**31; the full pass must match the same sequence run as two halves chained through the state.
_SUBPROCESS = textwrap.dedent(
    """
    import os
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    import torch
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla
    assert os.path.realpath(fla.__file__).startswith(os.path.realpath(_vendored_fla_dir()))
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_h import chunk_fwd_h
    from fla.ops.utils import chunk_local_cumsum

    torch.manual_seed(0)
    dev = "cuda"
    H, K, V = 32, 128, 16
    T = 2**19 + 2**16
    half = T // 2
    k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16).mul_(0.02)
    w = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16).mul_(0.02)
    u = torch.randn(1, T, H, V, device=dev, dtype=torch.bfloat16)
    g = chunk_local_cumsum(torch.full((1, T, H), -0.01, device=dev, dtype=torch.float32), chunk_size=64)

    def halves(*xs):
        return [x[:, :half].contiguous() for x in xs], [x[:, half:].contiguous() for x in xs]

    _, _, ht = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, output_final_state=True)
    (k1, w1, u1, g1), (k2, w2, u2, g2) = halves(k, w, u, g)
    _, _, h1 = chunk_gated_delta_rule_fwd_h(k=k1, w=w1, u=u1, g=g1, output_final_state=True)
    _, _, h2 = chunk_gated_delta_rule_fwd_h(k=k2, w=w2, u=u2, g=g2, initial_state=h1, output_final_state=True)
    gdn = (ht - h2).abs().max().item()
    del w, w1, w2, ht, h1, h2

    _, st = chunk_fwd_h(k=k, v=u, g=g, output_final_state=True)
    _, s1 = chunk_fwd_h(k=k1, v=u1, g=g1, output_final_state=True)
    _, s2 = chunk_fwd_h(k=k2, v=u2, g=g2, h0=s1, output_final_state=True)
    torch.cuda.synchronize()
    simple = (st - s2).abs().max().item()
    print(f"gdn {gdn} chunk_h {simple} peak_gib {torch.cuda.max_memory_allocated() / 2**30:.2f}")
    assert gdn == 0.0 and simple == 0.0, (gdn, simple)
    print("LONG_SEQUENCE_OK")
    """
)


# chunk_global_cumsum scans one sequence in a loop; zeros with ones on the last 4 tokens give exact small integers, so
# the tail past 2**31 elements must read back [1, 2, 3, 4] / [4, 3, 2, 1].
_CUMSUM_SUBPROCESS = textwrap.dedent(
    """
    import os
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["UNSLOTH_VENDORED_FLA_NO_AUTORUN"] = "1"
    import torch
    from unsloth_zoo.temporary_patches.fla_vendor import patch_vendor_fla, _vendored_fla_dir
    patch_vendor_fla()
    import fla
    assert os.path.realpath(fla.__file__).startswith(os.path.realpath(_vendored_fla_dir()))
    from fla.ops.utils import chunk_global_cumsum

    H, S = 32, 128
    T = 2**19 + 2**16
    s = torch.zeros(1, T, H, S, device="cuda", dtype=torch.bfloat16)
    s[:, -4:] = 1
    for reverse, tail, rest in ((False, [1, 2, 3, 4], 0), (True, [4, 3, 2, 1], 4)):
        o = chunk_global_cumsum(s, reverse=reverse, output_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        got = o[0, -4:, :, :].float().amax(dim=(1, 2)).tolist(), o[0, -4:, :, :].float().amin(dim=(1, 2)).tolist()
        assert got == (tail, tail), (reverse, got)
        head = o[0, :-4]
        assert head.amax().item() == rest and head.amin().item() == rest, reverse
        del head
        del o
    print(f"peak_gib {torch.cuda.max_memory_reserved() / 2**30:.2f}")
    print("GLOBAL_CUMSUM_OK")
    """
)


def _run(script, marker):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ZOO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert marker in proc.stdout, f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr[-4000:]}"


_needs_vendored_gpu = pytest.mark.skipif(
    not fla_vendor._vendored_injection_supported() or fla_vendor._gpu_lacks_dot_instructions(),
    reason="vendored fla kernels need CUDA + torch>=2.7 + triton>=3.3 (never on RDNA1)",
)


@_needs_vendored_gpu
def test_single_sequence_past_2_31_matches_split():
    if _free_gib() < 20:
        pytest.skip("needs about 20 GiB of free GPU memory")
    _run(_SUBPROCESS, "LONG_SEQUENCE_OK")


@_needs_vendored_gpu
def test_global_cumsum_single_sequence_past_2_31():
    if _free_gib() < 14:
        pytest.skip("needs about 14 GiB of free GPU memory")
    _run(_CUMSUM_SUBPROCESS, "GLOBAL_CUMSUM_OK")
