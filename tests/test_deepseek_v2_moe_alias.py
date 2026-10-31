"""Tests patch_deepseek_v2_moe_alias: aliases DeepseekV2MoE to DeepseekV2Moe only when just the new name exists
(transformers 5.x); no-op otherwise; uses a fake sys.modules module."""

from __future__ import annotations

import sys
import types

import pytest

# Side-effect import populates TEMPORARY_PATCHES and exposes the function.
from unsloth_zoo.temporary_patches.misc import patch_deepseek_v2_moe_alias

_PARENT = "transformers.models.deepseek_v2"
_LEAF = "transformers.models.deepseek_v2.modeling_deepseek_v2"


def _inject(monkeypatch, **attrs):
    """Inject a fake leaf module with a package parent, so the dotted import resolves purely from sys.modules, carrying the given attributes."""
    parent = types.ModuleType(_PARENT)
    parent.__path__ = []  # mark as package
    leaf = types.ModuleType(_LEAF)
    for k, v in attrs.items():
        setattr(leaf, k, v)
    parent.modeling_deepseek_v2 = leaf
    monkeypatch.setitem(sys.modules, _PARENT, parent)
    monkeypatch.setitem(sys.modules, _LEAF, leaf)
    return leaf


def _set_tf_version(monkeypatch, version):
    import transformers
    monkeypatch.setattr(transformers, "__version__", version, raising=False)


def test_aliases_new_name_to_old(monkeypatch):
    _set_tf_version(monkeypatch, "5.0.0")  # rename only exists on 5.x

    class DeepseekV2Moe:  # transformers 5 only exposes the new name
        pass

    leaf = _inject(monkeypatch, DeepseekV2Moe=DeepseekV2Moe)
    assert not hasattr(leaf, "DeepseekV2MoE")

    patch_deepseek_v2_moe_alias()

    assert leaf.DeepseekV2MoE is DeepseekV2Moe


def test_skips_eager_import_on_transformers_4(monkeypatch):
    # On 4.x the version gate returns before importing the module at all.
    _set_tf_version(monkeypatch, "4.57.6")
    monkeypatch.setitem(sys.modules, _LEAF, None)  # import would raise if reached

    patch_deepseek_v2_moe_alias()  # must not raise


def test_noop_when_old_name_already_present(monkeypatch):
    _set_tf_version(monkeypatch, "5.0.0")

    class DeepseekV2MoE:
        pass

    leaf = _inject(monkeypatch, DeepseekV2MoE=DeepseekV2MoE)

    patch_deepseek_v2_moe_alias()

    assert leaf.DeepseekV2MoE is DeepseekV2MoE
    assert not hasattr(leaf, "DeepseekV2Moe")


def test_noop_when_neither_name_present(monkeypatch):
    _set_tf_version(monkeypatch, "5.0.0")
    leaf = _inject(monkeypatch)  # empty module

    patch_deepseek_v2_moe_alias()  # must not raise

    assert not hasattr(leaf, "DeepseekV2MoE")
    assert not hasattr(leaf, "DeepseekV2Moe")


def test_noop_when_module_import_fails(monkeypatch):
    # None in sys.modules makes import_module raise ImportError, which the try/except must swallow.
    _set_tf_version(monkeypatch, "5.0.0")
    monkeypatch.setitem(sys.modules, _LEAF, None)

    patch_deepseek_v2_moe_alias()  # must not raise


def test_registered_in_temporary_patches():
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    assert patch_deepseek_v2_moe_alias in TEMPORARY_PATCHES
