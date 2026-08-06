# SPDX-License-Identifier: AGPL-3.0-only
"""The MLX simulation's platform spoof must not reach the build toolchain.

`tests/mlx_simulation` rebinds `platform.system` / `platform.machine` so the
`_IS_MLX` gate takes the Apple Silicon path. The rebind is process-wide and
permanent, so before it was scoped every test collected after an MLX module
inherited it: inductor caches a CPU vector ISA from `platform.machine()`
(`torch/_inductor/cpu_vec_isa.py`), "arm64" on x86_64 gives an empty list, and
`torch.compile` then emitted `at::vec::Vectorized` without the vec headers so
g++ failed. Downstream, the GRPO packed-path verifier swallowed that
`CppCompileError` and three `test_grpo_packed_verify_raw_logits` tests failed
only when run after an MLX module.
"""

import platform

# `mlx_simulation`, not `tests.mlx_simulation`: conftest puts tests/ on sys.path
# for exactly this, the rest of the MLX suite imports it that way, and an
# installed package named `tests` would otherwise win over this directory.
from mlx_simulation import _spoof_apple_silicon_platform


def test_the_toolchain_still_sees_the_real_host():
    _spoof_apple_silicon_platform()

    # Read as torch does, from a module not on the allow-list.
    real_machine = platform._orig_machine_for_mlx_shim()
    real_system = platform._orig_system_for_mlx_shim()

    assert platform.machine() == real_machine
    assert platform.system() == real_system


def test_the_gate_still_sees_apple_silicon():
    _spoof_apple_silicon_platform()

    # Same call from a module whose top-level package is on the allow-list.
    namespace = {"__name__": "unsloth_zoo.mlx.runtime", "platform": platform}
    exec("machine = platform.machine()\nsystem = platform.system()", namespace)

    assert namespace["machine"] == "arm64"
    assert namespace["system"] == "Darwin"


def test_inductor_observes_the_real_machine(monkeypatch):
    # The concrete consequence, asserted without paying for a compile.
    import torch                                          # noqa: F401  (loads _inductor)
    from torch._inductor import cpu_vec_isa

    _spoof_apple_silicon_platform()

    # Read platform the way cpu_vec_isa does, from its own module namespace.
    namespace = {"__name__": cpu_vec_isa.__name__, "platform": platform}
    exec("machine = platform.machine()", namespace)
    assert namespace["machine"] == platform._orig_machine_for_mlx_shim()

    # And the ISA list it derives from that. Compared against the same call with
    # the real value forced, not against non-emptiness: a host that supports none
    # of this build's ISAs legitimately yields an empty list, and asserting
    # non-emptiness would fail there for a reason unrelated to the spoof.
    cpu_vec_isa.valid_vec_isa_list.cache_clear()
    observed = cpu_vec_isa.valid_vec_isa_list()
    monkeypatch.setattr(platform, "machine", platform._orig_machine_for_mlx_shim)
    cpu_vec_isa.valid_vec_isa_list.cache_clear()
    expected = cpu_vec_isa.valid_vec_isa_list()
    assert observed == expected, (
        "inductor derived a different vector ISA list than the real host gives, "
        "so the spoof reached it and every torch.compile in this process would "
        "emit uncompilable at::vec code"
    )
