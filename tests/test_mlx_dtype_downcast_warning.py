# Unsloth Zoo - Utilities for Unsloth
# Test the bf16->fp16 downcast warning in unsloth_zoo.mlx.loader._convert_mlx_dtype.
#
# Gemma3-270m's native bf16 storage carries activations whose magnitudes
# exceed fp16's finite range. Silent bf16->fp16 downcast in
# FastMLXModel.from_pretrained(dtype="float16") was observed to drop a
# single-row LoRA memorization fixture's greedy-decode pass rate from
# 47% (dtype=None) to 15% (dtype="float16") across 15 seeds. The cast
# remains supported (users on M1/M2 need it), but it now emits a
# warning so callers see the trade-off.

from __future__ import annotations

import pytest
import warnings


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


class _FakeArray:
    """Minimal stand-in for an mx.array carrying just a dtype field.

    _convert_mlx_dtype only reads .dtype and calls .astype on params, both
    of which the shim implements via the underlying torch tensor; using a
    real torch tensor here keeps the path fully exercised.
    """
    def __init__(self, dtype):
        import torch
        # Empty tensor is fine — _convert_mlx_dtype doesn't read data.
        self._t = torch.zeros((1,), dtype=dtype)

    @property
    def dtype(self):
        return self._t.dtype

    def astype(self, target_dtype):
        return _FakeArray(target_dtype)


class _FakeModel:
    """The smallest object _convert_mlx_dtype needs:
    a .parameters() returning a flat dict and .update() that accepts the
    tree_map output.
    """
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return self._params

    def update(self, new_params):
        self._params = new_params


def _make_model(dtype_map):
    """Build a fake model with named params at the given dtypes."""
    import torch
    return _FakeModel({name: _FakeArray(dt) for name, dt in dtype_map.items()})


def test_warns_on_bf16_to_fp16_downcast():
    """The exact production scenario: native bf16 weights, user asks for fp16."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.bfloat16, "w2": torch.bfloat16})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float16)

    msgs = [str(w.message) for w in caught]
    assert any("bfloat16" in m and "float16" in m for m in msgs), (
        f"Expected a bf16->fp16 downcast warning, got: {msgs}"
    )


def test_no_warning_on_bf16_to_fp32_upcast():
    """Upcasts are safe; no warning."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.bfloat16})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float32)

    bf_fp_msgs = [
        str(w.message) for w in caught
        if "bfloat16" in str(w.message) and "float16" in str(w.message)
    ]
    assert bf_fp_msgs == [], (
        f"Did not expect a bf16->fp16 warning for a bf16->fp32 upcast; got: {bf_fp_msgs}"
    )


def test_no_warning_on_fp32_to_fp16_downcast():
    """fp32->fp16 is lossy too but is a different (already-explicit) regime."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.float32})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float16)

    bf_fp_msgs = [
        str(w.message) for w in caught
        if "bfloat16" in str(w.message) and "float16" in str(w.message)
    ]
    assert bf_fp_msgs == [], (
        f"Did not expect a bf16->fp16 warning for a fp32->fp16 cast; got: {bf_fp_msgs}"
    )


def test_no_warning_when_no_cast_needed():
    """All params already at target_dtype -> early return, no warnings."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.float16, "w2": torch.float16})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _convert_mlx_dtype(model, torch.float16)

    bf_fp_msgs = [
        str(w.message) for w in caught
        if "bfloat16" in str(w.message) and "float16" in str(w.message)
    ]
    assert bf_fp_msgs == [], (
        f"Did not expect a warning when no cast is needed; got: {bf_fp_msgs}"
    )


def test_cast_still_happens_after_warning():
    """The warning is advisory only — the requested cast still occurs."""
    import torch
    from unsloth_zoo.mlx.loader import _convert_mlx_dtype

    model = _make_model({"w1": torch.bfloat16})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _convert_mlx_dtype(model, torch.float16)

    # _FakeArray.astype returns a fresh _FakeArray with the new dtype,
    # so the parameter tree now reflects the target dtype.
    assert model._params["w1"].dtype == torch.float16
