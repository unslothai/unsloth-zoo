# SPDX-License-Identifier: AGPL-3.0-only
"""The MLX simulation's platform spoof must not reach the build toolchain.

`tests/mlx_simulation` rebinds `platform.system` and `platform.machine` so
unsloth_zoo's `_IS_MLX` gate takes the Apple Silicon path. The rebind is
process-wide and permanent, so before it was scoped, every test collected after
an MLX module inherited it. torch's inductor reads `platform.machine()` to pick
a CPU vector ISA (`torch/_inductor/cpu_vec_isa.py`, behind `functools.cache`),
"arm64" on an x86_64 host yields an empty ISA list, and the first `torch.compile`
in the process then froze that. Inductor still emitted `at::vec::Vectorized`
without the vec headers, so g++ failed with `'at::vec' has not been declared`.

Downstream that surfaced nowhere near the cause: the GRPO packed-path verifier
caught the `CppCompileError`, set `_unsloth_seq_packing_grad_ok = False`, and
three `test_grpo_packed_verify_raw_logits` tests failed only when run after an
MLX module, passing in isolation.
"""

import platform

from tests.mlx_simulation import _spoof_apple_silicon_platform


def test_the_toolchain_still_sees_the_real_host():
    _spoof_apple_silicon_platform()

    # Read them the way torch does, from a module that is not on the allow-list.
    real_machine = platform._orig_machine_for_mlx_shim()
    real_system = platform._orig_system_for_mlx_shim()

    assert platform.machine() == real_machine
    assert platform.system() == real_system


def test_the_gate_still_sees_apple_silicon():
    _spoof_apple_silicon_platform()

    # Same call made from a module whose top-level package is on the allow-list.
    namespace = {"__name__": "unsloth_zoo.mlx.runtime", "platform": platform}
    exec("machine = platform.machine()\nsystem = platform.system()", namespace)

    assert namespace["machine"] == "arm64"
    assert namespace["system"] == "Darwin"


def test_inductor_can_still_choose_a_vector_isa():
    # The concrete consequence, asserted directly rather than via a compile.
    import torch

    _spoof_apple_silicon_platform()
    from torch._inductor import cpu_vec_isa

    cpu_vec_isa.valid_vec_isa_list.cache_clear()
    if platform._orig_machine_for_mlx_shim() in ("x86_64", "AMD64"):
        assert cpu_vec_isa.valid_vec_isa_list(), (
            "inductor found no vector ISA, so the spoof reached it and every "
            "torch.compile in this process would emit uncompilable at::vec code"
        )
