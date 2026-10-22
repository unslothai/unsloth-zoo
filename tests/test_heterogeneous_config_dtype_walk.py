# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A heterogeneous config must not kill the dtype walk in patch_model_and_tokenizer.

Observed on real hardware, not constructed here: `unsloth/gemma-4-E2B-it` on
transformers 5.15.1 with unsloth 2026.8.20 fails to load AT ALL, on a Tesla T4,
before a single weight is touched:

    File "unsloth_zoo/patching_utils.py", line 467, in __fix_dtype
        __fix_dtype(getattr(config, key, None))
    File "transformers/integrations/heterogeneity/configuration_utils.py", line 298
        raise AmbiguousGlobalPerLayerAttributeError(
    AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
    and may vary across layers.

`__fix_dtype` walks every key `config.to_dict()` reports, looking for nested
config objects that carry a dtype. From transformers 5.15, a MatFormer or other
heterogeneous config raises straight out of `PretrainedConfig.__getattribute__`
for any attribute that varies per layer -- so a key `to_dict()` itself just
listed cannot be read back.

The reason it is fatal rather than skipped is a detail worth stating, because
the code LOOKS defensive: `getattr(config, key, None)` has a default, and a
default only suppresses `AttributeError`. `AmbiguousGlobalPerLayerAttributeError`
is not one, so it propagates through the whole load.

Fully synthetic and CPU-only: the failure is a property of attribute access, so
reproducing it needs neither gemma-4 nor a GPU nor transformers 5.15.
"""

import sys
import types

import pytest
import torch


class AmbiguousGlobalPerLayerAttributeError(Exception):
    """Stands in for the transformers exception. Deliberately NOT an
    AttributeError, which is the entire reason the original getattr default
    does not catch it."""


class _HeterogeneousConfig:
    """A config whose to_dict() lists a key that cannot be read back.

    This is the shape transformers 5.15 gives a MatFormer: `head_dim` appears
    in the serialised dict because it has a global value, and reading it
    through the attribute path raises because it varies per layer.
    """

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
    """The production walk, extracted so it can be driven without a model.

    Imported from the module rather than copied: a copy would let the two drift
    and this test would then be guarding a fiction. `patch_model_and_tokenizer`
    defines `__fix_dtype` as a closure, so the walk is re-expressed here with
    the SAME structure and the fix is asserted against the real source below.
    """
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
    """The bug, reproduced. Shown first so the fix below is measured against a
    failure that genuinely happens rather than against nothing."""

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
    """Both halves matter. Surviving while silently skipping the dtype would be
    the same class of bug wearing a different hat: the load would succeed and
    the model would carry bfloat16 on a card that has no bfloat16."""
    config = _HeterogeneousConfig(head_dim = 256, dtype = torch.bfloat16, num_layers = 4)
    _fix_dtype_walk(config, torch.float16)
    assert config.to_dict()["dtype"] is torch.float16


def test_a_nested_config_is_still_reached():
    """The walk exists to descend. A fix that swallowed too much -- catching
    around the recursion rather than around the read -- would stop descending
    at the first awkward key and leave nested dtypes wrong."""
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
    """Asserted against the shipped file, because the walk above is a
    re-expression and a re-expression can be fixed while the product is not.

    Specifically: the try must wrap the `getattr`, and the recursion must sit
    OUTSIDE it. A try around both would swallow errors raised deeper in the
    walk, turning a real failure into a silent skip.
    """
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
