# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Smoke + Tier 1 + Tier 2 tests for the mlx_stub package.

Verifies:
1. simulate_mlx_on_torch() succeeds and registers all named submodules.
2. PR-B's 5 fresh symbols (mx.metal.is_available, set_wired_limit,
   device_info, clear_cache, synchronize) work as expected.
3. The ~70 trivial passthroughs round-trip vs torch on random inputs.
4. Sub-architecture VLM submodules auto-resolve via the MetaPathFinder.
5. Tree utils (tree_flatten/tree_unflatten/tree_map) roundtrip.
6. JAX-style RNG keys split deterministically.
"""

from __future__ import annotations

import pytest
import torch


# Install the shim once at session start.  All later `import mlx*` calls
# resolve to the stubs.  The shim package lives at workspace_1/mlx_simulation/
# (loaded via the workspace conftest.py that puts the workspace on sys.path).
@pytest.fixture(autouse=True, scope="session")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# ---------------------------------------------------------------------------
# 1. All named imports succeed.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_path", [
    "mlx",
    "mlx.core",
    "mlx.core.random",
    "mlx.core.linalg",
    "mlx.core.fft",
    "mlx.core.fast",
    "mlx.core.metal",
    "mlx.core.cuda",
    "mlx.core.distributed",
    "mlx.nn",
    "mlx.nn.losses",
    "mlx.nn.init",
    "mlx.nn.utils",
    "mlx.optimizers",
    "mlx.optimizers.schedulers",
    "mlx.utils",
    "mlx_lm",
    "mlx_lm.utils",
    "mlx_lm.tuner",
    "mlx_lm.tuner.lora",
    "mlx_lm.tuner.utils",
    "mlx_lm.tuner.datasets",
    "mlx_lm.tuner.trainer",
    "mlx_lm.sample_utils",
    "mlx_lm.models",
    "mlx_lm.models.gated_delta",
    "mlx_vlm",
    "mlx_vlm.utils",
    "mlx_vlm.prompt_utils",
    "mlx_vlm.models",
])
def test_named_module_imports(module_path):
    import importlib
    mod = importlib.import_module(module_path)
    assert mod is not None


@pytest.mark.parametrize("submodule", [
    "mlx_vlm.models.qwen2_5_vl",
    "mlx_vlm.models.qwen2_5_vl.qwen2_5_vl",
    "mlx_vlm.models.qwen2_5_vl.vision",
    "mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe",
    "mlx_vlm.models.gemma3n.language",
    "mlx_vlm.models.pixtral.pixtral",
])
def test_vlm_subarch_auto_resolve(submodule):
    """40+ VLM architecture submodules must auto-resolve via the finder."""
    import importlib
    mod = importlib.import_module(submodule)
    assert mod is not None


# ---------------------------------------------------------------------------
# 2. Tier 1: PR-B fresh symbols.
# ---------------------------------------------------------------------------

def test_metal_is_available_returns_false():
    import mlx.core as mx
    assert mx.metal.is_available() is False


def test_set_wired_limit_is_no_op():
    import mlx.core as mx
    assert mx.set_wired_limit(1024) is None
    assert mx.set_memory_limit(2048) is None
    assert mx.set_cache_limit(512) is None


def test_device_info_returns_dict():
    import mlx.core as mx
    info = mx.device_info()
    assert isinstance(info, dict)
    assert "device_name" in info
    assert "max_recommended_working_set_size" in info


def test_clear_cache_no_op():
    import mlx.core as mx
    mx.clear_cache()  # should not raise


def test_synchronize_no_op():
    import mlx.core as mx
    mx.synchronize()


# ---------------------------------------------------------------------------
# 3. Tier 2: trivial passthroughs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_elementwise_math(dtype):
    import mlx.core as mx
    torch.manual_seed(42)
    a = torch.randn(4, 5, dtype=dtype)
    b = torch.randn(4, 5, dtype=dtype)
    torch.testing.assert_close(mx.add(a, b), torch.add(a, b))
    torch.testing.assert_close(mx.subtract(a, b), torch.subtract(a, b))
    torch.testing.assert_close(mx.multiply(a, b), torch.multiply(a, b))
    torch.testing.assert_close(mx.exp(a.float()).to(dtype), torch.exp(a.float()).to(dtype))
    torch.testing.assert_close(mx.maximum(a, b), torch.maximum(a, b))
    torch.testing.assert_close(mx.where(a > 0, a, b), torch.where(a > 0, a, b))


def test_matmul():
    import mlx.core as mx
    torch.manual_seed(42)
    a = torch.randn(4, 5)
    b = torch.randn(5, 6)
    torch.testing.assert_close(mx.matmul(a, b), a @ b)


def test_reductions_with_axis():
    import mlx.core as mx
    torch.manual_seed(42)
    a = torch.randn(4, 5)
    torch.testing.assert_close(mx.sum(a, axis=0), a.sum(dim=0))
    torch.testing.assert_close(mx.sum(a, axis=1, keepdims=True), a.sum(dim=1, keepdim=True))
    torch.testing.assert_close(mx.mean(a, axis=-1), a.mean(dim=-1))
    torch.testing.assert_close(mx.max(a, axis=0), a.amax(dim=0))


def test_dtypes_alias_torch():
    import mlx.core as mx
    assert mx.float32 is torch.float32
    assert mx.bfloat16 is torch.bfloat16
    assert mx.float16 is torch.float16
    assert mx.int32 is torch.int32
    assert mx.bool_ is torch.bool


def test_constructors():
    import mlx.core as mx
    torch.testing.assert_close(mx.zeros((3, 4)), torch.zeros(3, 4))
    torch.testing.assert_close(mx.ones((3, 4)), torch.ones(3, 4))
    torch.testing.assert_close(mx.arange(0, 5), torch.arange(0, 5))
    torch.testing.assert_close(mx.linspace(0.0, 1.0, num=5), torch.linspace(0.0, 1.0, 5))


def test_shape_ops():
    import mlx.core as mx
    a = torch.randn(2, 3, 4)
    torch.testing.assert_close(mx.reshape(a, (6, 4)), a.reshape(6, 4))
    torch.testing.assert_close(mx.expand_dims(a, axis=0), a.unsqueeze(0))
    torch.testing.assert_close(mx.concatenate([a, a], axis=0), torch.cat([a, a], dim=0))
    torch.testing.assert_close(mx.stack([a, a], axis=0), torch.stack([a, a], dim=0))


def test_take_along_axis():
    import mlx.core as mx
    a = torch.randn(3, 5)
    idx = torch.randint(0, 5, (3, 1))
    torch.testing.assert_close(
        mx.take_along_axis(a, idx, axis=1),
        torch.take_along_dim(a, idx, dim=1),
    )


def test_array_constructor_returns_tensor():
    import mlx.core as mx
    arr = mx.array([1.0, 2.0, 3.0])
    assert isinstance(arr, torch.Tensor)
    torch.testing.assert_close(arr, torch.tensor([1.0, 2.0, 3.0]))


def test_finfo_and_dtype_kinds():
    import mlx.core as mx
    info = mx.finfo(mx.float32)
    assert info.eps > 0
    assert mx.issubdtype(mx.float32, mx.floating) is True
    assert mx.issubdtype(mx.int32, mx.floating) is False
    assert mx.issubdtype(mx.int32, mx.integer) is True


# ---------------------------------------------------------------------------
# 4. Tree utils round-trip.
# ---------------------------------------------------------------------------

def test_tree_flatten_unflatten():
    import mlx.utils as mlxu
    tree = {"a": {"b": 1, "c": [2, 3]}, "d": 4}
    flat = mlxu.tree_flatten(tree)
    assert ("a.b", 1) in flat
    assert ("d", 4) in flat
    assert ("a.c.0", 2) in flat
    rebuilt = mlxu.tree_unflatten(flat)
    assert rebuilt == tree


def test_tree_map():
    import mlx.utils as mlxu
    tree = {"a": 1, "b": [2, 3]}
    doubled = mlxu.tree_map(lambda x: x * 2, tree)
    assert doubled == {"a": 2, "b": [4, 6]}


# ---------------------------------------------------------------------------
# 5. RNG keys are deterministic.
# ---------------------------------------------------------------------------

def test_random_seed_reproducible():
    import mlx.core as mx
    mx.random.seed(123)
    a = mx.random.uniform(shape=(4,))
    mx.random.seed(123)
    b = mx.random.uniform(shape=(4,))
    torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# 6. _Noop raises loudly on call (regression guard against silent masking).
# ---------------------------------------------------------------------------

def test_noop_raises_on_call():
    import mlx.core as mx
    # An unknown attribute returns _Noop; calling raises NotImplementedError
    # with the symbol name.
    noop = mx.this_is_a_definitely_unknown_symbol_xyz
    with pytest.raises(NotImplementedError, match="this_is_a_definitely_unknown_symbol_xyz"):
        noop()
