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

"""torch 2.15 fixed two Inductor correctness bugs that 2.10-2.14 still ship: mix-order reductions writing wrong
norm-weight gradients (2.10-2.14) and rand4x giving correlated CUDA dropout masks, so a reentrant checkpoint's
recompute differentiates a different mask than its forward applied (2.13, 2.14). Unsloth turns both paths off on
the affected releases only, and leaves them alone when the user opts back in."""

import os
import subprocess
import sys
import textwrap
import types

import pytest
import torch


def _minor():
    major, minor = torch.__version__.split(".")[:2]
    return int(major), int(minor)


def _has_mix_order():
    import torch._inductor.config as config
    return hasattr(config.triton, "mix_order_reduction")


def _codegen_rand():
    """Generate Inductor's Triton `rand` for a 1D CUDA kernel (no GPU needed: only codegen strings)."""
    from torch._inductor.codegen.triton import TritonKernel, TritonOverrides
    from torch._inductor.virtualized import V

    kernel = object.__new__(TritonKernel)
    kernel.triton_tensor_ndim = lambda: 1
    kernel.dense_size_list = lambda: ["XBLOCK"]
    graph = types.SimpleNamespace(get_current_device_or_throw = lambda: torch.device("cuda"))
    with V.set_graph_handler(graph), V.set_kernel_handler(kernel):
        return TritonOverrides.rand("seed", "offset")


def test_mix_order_reduction_off_only_where_buggy():
    import torch._inductor.config as config
    from unsloth_zoo.temporary_patches.common import torch_compile_options
    if not _has_mix_order():
        pytest.skip("this torch has no mix-order reduction")
    if (2, 10) <= _minor() < (2, 15):
        assert config.triton.mix_order_reduction is False
        assert torch_compile_options.get("triton.mix_order_reduction") is False
    else:
        assert config.triton.mix_order_reduction is True
        assert "triton.mix_order_reduction" not in torch_compile_options


def test_compiled_cuda_dropout_never_uses_rand4x():
    import unsloth_zoo  # noqa: F401  (applies the workaround at import)
    try:
        code = _codegen_rand()
    except Exception as error:
        pytest.skip(f"Inductor codegen not reachable on torch {torch.__version__}: {error!r}")
    assert "rand4x" not in code, code


def test_rand4x_workaround_tags_compile_cache():
    from unsloth_zoo.temporary_patches.common import INDUCTOR_CORRECTNESS_CACHE_TAG
    from torch._inductor.codegen.triton import TritonOverrides
    import torch._inductor.config as config
    import torch.compiler.config as compiler_config
    buggy = hasattr(TritonOverrides, "_can_use_4x_random") and not hasattr(config, "use_rand4x")
    tags = compiler_config.cache_key_tag.split(",")
    assert (INDUCTOR_CORRECTNESS_CACHE_TAG in tags) == buggy


_CHILD = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import unsloth_zoo
    import torch._inductor.config as config
    from torch._inductor.codegen.triton import TritonOverrides
    from unsloth_zoo.temporary_patches.common import torch_compile_options
    print(json.dumps({
        "mix_order": getattr(config.triton, "mix_order_reduction", None),
        "option": torch_compile_options.get("triton.mix_order_reduction", "absent"),
        "rand4x_patched": getattr(getattr(TritonOverrides, "_can_use_4x_random", None), "_unsloth_rand4x_off", False),
    }))
""")


def _child(env_extra):
    import json
    env = dict(os.environ, UNSLOTH_ALLOW_CPU = "1", CUDA_VISIBLE_DEVICES = "", **env_extra)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", _CHILD, root], env = env, capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not ((2, 10) <= _minor() < (2, 15)), reason = "workarounds only apply on torch 2.10-2.14")
def test_opt_out_switches_keep_upstream_behaviour():
    default = _child({})
    assert default["mix_order"] is False and default["option"] is False
    kept = _child({"UNSLOTH_MIX_ORDER_REDUCTION": "1", "TORCHINDUCTOR_USE_RAND4X": "1"})
    assert kept["mix_order"] is True and kept["option"] == "absent"
    assert kept["rand4x_patched"] is False
    if _minor() >= (2, 13):
        assert default["rand4x_patched"] is True


_SKIP_CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import unsloth_zoo
    import torch._inductor.config as config
    from unsloth_zoo.patching_utils import patch_torch_compile
    cls = type(config)
    original = cls.__setattr__
    def refuse_dce(self, name, value):
        if self is config and name == "dce":
            raise AttributeError("torch._inductor.config.dce does not exist")
        return original(self, name, value)
    cls.__setattr__ = refuse_dce
    try:
        patch_torch_compile(debug = False, O3 = False, ignore_errors = True)
    finally:
        cls.__setattr__ = original
    print("DONE")
""")


@pytest.mark.parametrize("logging_on", [True, False])
def test_skipped_compiler_config_writes_are_logged(logging_on):
    """A config key this torch lacks used to vanish in a bare `except: pass`; with UNSLOTH_ENABLE_LOGGING it is named."""
    env = dict(os.environ, UNSLOTH_ALLOW_CPU = "1", CUDA_VISIBLE_DEVICES = "",
               UNSLOTH_ENABLE_LOGGING = "1" if logging_on else "0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", _SKIP_CHILD, root], env = env, capture_output = True, text = True, timeout = 600)
    assert out.returncode == 0 and "DONE" in out.stdout, out.stderr[-4000:]
    logged = "Skipped Inductor `config.dce = True`" in (out.stdout + out.stderr)
    assert logged == logging_on, (out.stdout + out.stderr)[-4000:]
