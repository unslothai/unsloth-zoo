# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The MLX platform spoof must stay scoped to the two gate modules: a process-wide arm64 platform.machine() empties
inductor's vec ISA list, so later torch.compile builds fail."""

import platform

# Import as mlx_simulation, not tests.mlx_simulation: conftest puts tests/ on sys.path for this, and an installed package named tests would otherwise win.
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

    # Same call from one of the two gate modules.
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

    # Compare the ISA list to the real-machine result, not to non-emptiness: some hosts legitimately support none of the build's ISAs and yield an empty list.
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


def test_other_modules_in_the_same_package_still_see_the_real_host():
    """Allow-list exact module names, not packages: unsloth_zoo.llama_cpp picks its archive from the real host, and a
    package-wide spoof once selected the macOS build on Linux."""
    _spoof_apple_silicon_platform()

    for module in ("unsloth_zoo.llama_cpp", "unsloth_zoo.device_type",
                   "unsloth.models.loader", "unsloth_zoo"):
        namespace = {"__name__": module, "platform": platform}
        exec("machine = platform.machine()\nsystem = platform.system()", namespace)
        assert namespace["machine"] == platform._orig_machine_for_mlx_shim(), module
        assert namespace["system"] == platform._orig_system_for_mlx_shim(), module
