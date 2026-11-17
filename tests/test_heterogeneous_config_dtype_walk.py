# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A heterogeneous config must not kill the dtype walk in patch_model_and_tokenizer: transformers 5.15 raises
AmbiguousGlobalPerLayerAttributeError, which getattr's default does not catch (seen loading unsloth/gemma-4-E2B-it on
transformers 5.15.1); CPU-only synthetic test."""

import sys
import types

import pytest
import torch


class AmbiguousGlobalPerLayerAttributeError(Exception):
    """Stands in for the transformers exception; deliberately not an AttributeError, so the getattr default in the walk does not catch it."""


class _HeterogeneousConfig:
    """Config whose to_dict() lists head_dim (a transformers 5.15 MatFormer shape): a global value is serialised, but reading it raises because it varies per layer."""

    def __init__(self, **kw):
        self._data = dict(kw)
        self.per_layer_keys = {"head_dim"}

    def to_dict(self):
        return dict(self._data)

    def __getattr__(self, name):
        data = object.__getattribute__(self, "_data")
        if name in object.__getattribute__(self, "per_layer_keys"):
            raise AmbiguousGlobalPerLayerAttributeError(
                f"{name!r} is a per-layer attribute and may vary across layers."
            )
        if name in data:
            return data[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("_data", "per_layer_keys"):
            object.__setattr__(self, name, value)
        else:
            object.__getattribute__(self, "_data")[name] = value


def _fix_dtype_walk(config, correct_dtype):
    """Re-expresses the __fix_dtype closure from patch_model_and_tokenizer so it runs without a model; the fix is asserted against the real source separately."""
    if not hasattr(config, "to_dict"):
        return
    for key in config.to_dict():
        if key in ("torch_dtype", "dtype"):
            setattr(config, key, correct_dtype)
        else:
            try:
                child = getattr(config, key, None)
            except Exception:
                continue
            _fix_dtype_walk(child, correct_dtype)


def test_the_unfixed_walk_dies_on_a_heterogeneous_config():
    """Reproduces the bug first, so the fix is measured against a failure that genuinely happens rather than against nothing."""

    def unfixed(config, correct_dtype):
        if not hasattr(config, "to_dict"):
            return
        for key in config.to_dict():
            if key in ("torch_dtype", "dtype"):
                setattr(config, key, correct_dtype)
            else:
                unfixed(getattr(config, key, None), correct_dtype)

    config = _HeterogeneousConfig(head_dim = 256, dtype = torch.bfloat16)
    with pytest.raises(AmbiguousGlobalPerLayerAttributeError):
        unfixed(config, torch.float16)


def test_the_fixed_walk_survives_and_still_corrects_the_dtype():
    """The walk must both survive and still correct the dtype: skipping it silently would leave the model in bfloat16 on a card without bfloat16 support."""
    config = _HeterogeneousConfig(head_dim = 256, dtype = torch.bfloat16, num_layers = 4)
    _fix_dtype_walk(config, torch.float16)
    assert config.to_dict()["dtype"] is torch.float16


def test_a_nested_config_is_still_reached():
    """The walk must still descend into nested configs; catching around the recursion instead of the read would stop at the first awkward key."""
    inner = _HeterogeneousConfig(dtype = torch.bfloat16)
    outer = _HeterogeneousConfig(head_dim = 256, vision_config = inner, dtype = torch.bfloat16)
    _fix_dtype_walk(outer, torch.float16)
    assert inner.to_dict()["dtype"] is torch.float16, "the nested config was never reached"
    assert outer.to_dict()["dtype"] is torch.float16


def test_an_ordinary_config_is_unaffected():
    plain = _HeterogeneousConfig(dtype = torch.bfloat16, hidden_size = 8)
    plain.per_layer_keys = set()
    _fix_dtype_walk(plain, torch.float16)
    assert plain.to_dict()["dtype"] is torch.float16


def test_the_production_source_guards_the_read_and_not_the_recursion():
    """Checks the shipped source, not the re-expression: the try must wrap only the getattr, with the recursion outside it, so deeper errors are not swallowed."""
    import inspect

    from unsloth_zoo import patching_utils

    src = inspect.getsource(patching_utils.patch_model_and_tokenizer)
    assert "child = getattr(config, key, None)" in src
    assert "__fix_dtype(child)" in src
    lines = [l.strip() for l in src.splitlines()]
    try_at = lines.index("try:")
    read_at = lines.index("child = getattr(config, key, None)")
    except_at = lines.index("except Exception:")
    recurse_at = lines.index("__fix_dtype(child)")
    assert try_at < read_at < except_at < recurse_at, (
        "the recursion must be outside the try, or a failure deeper in the "
        "walk is swallowed as though the key were merely unreadable"
    )
