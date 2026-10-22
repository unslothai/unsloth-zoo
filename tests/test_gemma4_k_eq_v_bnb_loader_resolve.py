# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""Gemma 4 k_eq_v patch must find the BnB loader in tree or in vllm-bnb-plugin (vLLM >= 0.28). vLLM is mocked."""
import os
import sys
import types
from contextlib import contextmanager

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

IN_TREE = "vllm.model_executor.model_loader.bitsandbytes_loader"
PLUGIN = "vllm_bnb_plugin.bitsandbytes_loader"
_PARENTS = {
    IN_TREE: ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader"),
    PLUGIN: ("vllm_bnb_plugin",),
}


def _make_loader_module(name):
    module = types.ModuleType(name)

    class BitsAndBytesModelLoader:
        def _stack_quantization_states(self, model, quant_state_dict):
            return dict(quant_state_dict)

    module.BitsAndBytesModelLoader = BitsAndBytesModelLoader
    return module


@contextmanager
def _loader_modules(in_tree = None, plugin = None):
    """Install `in_tree` / `plugin` as the loader modules; None means absent."""
    names = {IN_TREE, PLUGIN}
    for parents in _PARENTS.values():
        names.update(parents)
    saved = {name: sys.modules.get(name, KeyError) for name in names}
    try:
        for path, module in ((IN_TREE, in_tree), (PLUGIN, plugin)):
            if module is None:
                # A None entry makes importlib raise ModuleNotFoundError.
                sys.modules[path] = None
                continue
            for parent in _PARENTS[path]:
                if sys.modules.get(parent) is None:
                    pkg = types.ModuleType(parent)
                    pkg.__path__ = []
                    sys.modules[parent] = pkg
            sys.modules[path] = module
        yield
    finally:
        for name, prev in saved.items():
            if prev is KeyError:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _is_patched(module):
    fn = module.BitsAndBytesModelLoader._stack_quantization_states
    return getattr(fn, "_unsloth_gemma4_k_eq_v_patch", False)


def test_plugin_only_patches_plugin_loader():
    from unsloth_zoo import empty_model
    plugin = _make_loader_module(PLUGIN)
    with _loader_modules(in_tree = None, plugin = plugin):
        empty_model.patch_gemma4_vllm_k_eq_v_support()
        assert empty_model._import_vllm_bnb_loader_module() is plugin
    assert _is_patched(plugin)


def test_in_tree_only_patches_in_tree_loader():
    from unsloth_zoo import empty_model
    in_tree = _make_loader_module(IN_TREE)
    with _loader_modules(in_tree = in_tree, plugin = None):
        empty_model.patch_gemma4_vllm_k_eq_v_support()
        assert empty_model._import_vllm_bnb_loader_module() is in_tree
    assert _is_patched(in_tree)


def test_in_tree_preferred_when_both_present():
    from unsloth_zoo import empty_model
    in_tree = _make_loader_module(IN_TREE)
    plugin = _make_loader_module(PLUGIN)
    with _loader_modules(in_tree = in_tree, plugin = plugin):
        empty_model.patch_gemma4_vllm_k_eq_v_support()
    assert _is_patched(in_tree)
    assert not _is_patched(plugin)


def test_neither_raises_actionable_error():
    from unsloth_zoo import empty_model
    with _loader_modules(in_tree = None, plugin = None):
        with pytest.raises(RuntimeError, match = "pip install vllm-bnb-plugin"):
            empty_model.patch_gemma4_vllm_k_eq_v_support()
        assert empty_model._import_vllm_bnb_loader_module() is None


def test_repeated_patching_wraps_once():
    from unsloth_zoo import empty_model
    plugin = _make_loader_module(PLUGIN)
    with _loader_modules(in_tree = None, plugin = plugin):
        empty_model.patch_gemma4_vllm_k_eq_v_support()
        first = plugin.BitsAndBytesModelLoader._stack_quantization_states
        empty_model.patch_gemma4_vllm_k_eq_v_support()
        second = plugin.BitsAndBytesModelLoader._stack_quantization_states
    assert first is second


def test_patched_plugin_loader_duplicates_k_quant_state_to_v():
    import torch
    from unsloth_zoo import empty_model

    class _TextConfig:
        model_type = "gemma4_text"
        attention_k_eq_v = True
        layer_types = ["full_attention"]

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _TextConfig()
            self.k = torch.nn.Parameter(torch.zeros(1))

        def named_parameters(self, *args, **kwargs):
            yield "model.layers.0.self_attn.k_proj.weight", self.k
            yield "model.layers.0.self_attn.v_proj.weight", self.k

    plugin = _make_loader_module(PLUGIN)
    with _loader_modules(in_tree = None, plugin = plugin):
        empty_model.patch_gemma4_vllm_k_eq_v_support()
    k_name = "model.layers.0.self_attn.k_proj.weight"
    v_name = "model.layers.0.self_attn.v_proj.weight"
    loader = plugin.BitsAndBytesModelLoader()
    out = loader._stack_quantization_states(_Model(), {k_name: {"tag": "k"}})
    assert out[v_name] == {"tag": "k"}
    assert out[v_name] is not out[k_name]
