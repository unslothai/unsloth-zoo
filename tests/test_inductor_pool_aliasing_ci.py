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

# With CI set, torch >= 2.11 raises on a compiled function's first call under inference_mode
# when memory_planning pools buffers: inductor::_alloc_from_pool returns a view of its pool.
import os
import subprocess
import sys
import textwrap

import pytest
import torch

runtime_wrappers = pytest.importorskip("torch._functorch._aot_autograd.runtime_wrappers")
if not hasattr(runtime_wrappers, "_check_custom_op_aliasing"):
    pytest.skip("torch has no custom op aliasing check (< 2.11)", allow_module_level = True)

_RUN = textwrap.dedent("""
    import sys, warnings, torch
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config
    if sys.argv[1] == "fix":
        from unsloth_zoo.temporary_patches.common import allow_inductor_pool_aliasing
        allow_inductor_pool_aliasing()

    def f(x, y):
        a = x.sin() @ y
        b = a.cos() @ y
        return ((b + a).relu() @ y).sum(-1), b * 2

    x, y = torch.randn(64, 64), torch.randn(64, 64) / 8
    warnings.filterwarnings("error", message = ".*_alloc_from_pool")
    with functorch_config.patch(error_on_custom_op_aliasing = True), \\
         inductor_config.patch(memory_planning = True, memory_pool = "none"), torch.inference_mode():
        out = torch.compile(f, dynamic = True)(x, y)
    torch.testing.assert_close(out[0], f(x, y)[0])
    print("OK")
""")


def _run(mode, tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / mode)}
    return subprocess.run([sys.executable, "-c", _RUN, mode], capture_output = True, text = True, env = env, timeout = 600)


def test_memory_planning_inference_mode_with_ci(tmp_path):
    control = _run("stock", tmp_path)
    assert control.returncode != 0 and "inductor::_alloc_from_pool" in control.stderr, control.stdout + control.stderr
    fixed = _run("fix", tmp_path)
    assert fixed.returncode == 0 and "OK" in fixed.stdout, fixed.stdout + fixed.stderr


def test_other_custom_ops_still_checked():
    from unsloth_zoo.temporary_patches.common import allow_inductor_pool_aliasing
    allow_inductor_pool_aliasing()
    allow_inductor_pool_aliasing()  # idempotent: one wrapper
    check = runtime_wrappers._check_custom_op_aliasing
    assert check._unsloth_skips_alloc_from_pool
    x = torch.randn(4)
    check("inductor::_alloc_from_pool", (x, 0), {}, x[:2])
    import torch._functorch.config as functorch_config
    with functorch_config.patch(error_on_custom_op_aliasing = True):
        with pytest.raises(RuntimeError):
            check("mylib::aliasing_op", (x,), {}, x)
