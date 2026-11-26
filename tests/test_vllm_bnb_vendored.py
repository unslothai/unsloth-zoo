# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""vLLM >= 0.28 has no built-in bitsandbytes; unsloth_zoo.vllm_bnb (a copy of vllm-bnb-plugin) fills in when the
plugin is not installed. Resolution order, the forced provider, and the copy's invariants. vLLM is not imported."""
import ast
import importlib.util
import os
import types

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
PKG = os.path.join(ROOT, "unsloth_zoo", "vllm_bnb")


def _load_vllm_bnb():
    # Load unsloth_zoo/vllm_bnb/__init__.py alone: importing unsloth_zoo would pull torch / vLLM.
    spec = importlib.util.spec_from_file_location("_fi4_vllm_bnb", os.path.join(PKG, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(vb, monkeypatch, *, modules, specs, plugin_raises = False):
    """modules: importable dotted names; specs: names find_spec sees. Records register calls."""
    calls = []

    def plugin_register():
        calls.append("plugin")
        if plugin_raises:
            raise RuntimeError("plugin built for another vLLM")

    def fake_import(name):
        if name == "vllm_bnb_plugin":
            return types.SimpleNamespace(register = plugin_register)
        if name not in modules:
            raise ModuleNotFoundError(name)
        return types.SimpleNamespace(__name__ = name)

    monkeypatch.setattr(vb, "_import", fake_import)
    monkeypatch.setattr(vb, "_find_spec", lambda name: object() if name in specs else None)
    monkeypatch.setattr(vb, "register", lambda: calls.append("unsloth"))
    monkeypatch.setattr(vb, "_RESOLVED", None)
    monkeypatch.delenv("UNSLOTH_VLLM_BNB", raising = False)
    return calls


ALL = set(_load_vllm_bnb()._IN_TREE + _load_vllm_bnb()._PLUGIN + _load_vllm_bnb()._VENDORED)
NO_IN_TREE = ALL - set(_load_vllm_bnb()._IN_TREE)


def test_in_tree_wins_and_registers_nothing(monkeypatch):
    vb = _load_vllm_bnb()
    calls = _env(vb, monkeypatch, modules = ALL, specs = {"vllm", "bitsandbytes", "vllm_bnb_plugin"})
    quant, loader, source = vb.resolve()
    assert source == "vllm" and quant.__name__ == vb._IN_TREE[0] and loader.__name__ == vb._IN_TREE[1]
    assert calls == []


def test_plugin_used_and_registered_when_installed(monkeypatch):
    vb = _load_vllm_bnb()
    calls = _env(vb, monkeypatch, modules = NO_IN_TREE, specs = {"vllm", "bitsandbytes", "vllm_bnb_plugin"})
    quant, loader, source = vb.resolve()
    assert source == "vllm-bnb-plugin" and quant.__name__ == vb._PLUGIN[0] and loader.__name__ == vb._PLUGIN[1]
    # Registered before load_vllm swaps the compute-dtype config into vLLM's registry.
    assert calls == ["plugin"]


def test_vendored_copy_when_vllm_has_no_bitsandbytes(monkeypatch):
    vb = _load_vllm_bnb()
    calls = _env(vb, monkeypatch, modules = NO_IN_TREE - set(vb._PLUGIN), specs = {"vllm", "bitsandbytes"})
    quant, loader, source = vb.resolve()
    assert source == "unsloth" and quant.__name__ == vb._VENDORED[0] and loader.__name__ == vb._VENDORED[1]
    assert calls == ["unsloth"]
    # Cached: a second call registers nothing more.
    assert vb.resolve()[2] == "unsloth" and calls == ["unsloth"]


def test_broken_plugin_falls_back_to_vendored_copy(monkeypatch, capsys):
    vb = _load_vllm_bnb()
    calls = _env(vb, monkeypatch, modules = NO_IN_TREE, specs = {"vllm", "bitsandbytes", "vllm_bnb_plugin"},
                 plugin_raises = True)
    assert vb.resolve()[2] == "unsloth"
    assert calls == ["plugin", "unsloth"]
    assert "using Unsloth's copy" in capsys.readouterr().out


def test_no_bitsandbytes_or_no_vllm_resolves_nothing(monkeypatch):
    vb = _load_vllm_bnb()
    calls = _env(vb, monkeypatch, modules = NO_IN_TREE, specs = {"vllm"})
    assert vb.resolve() == (None, None, None) and calls == []
    calls = _env(vb, monkeypatch, modules = ALL, specs = set())
    assert vb.resolve() == (None, None, None) and calls == []


def test_forced_unsloth_copy_overrides_in_tree(monkeypatch):
    vb = _load_vllm_bnb()
    calls = _env(vb, monkeypatch, modules = ALL, specs = {"vllm", "bitsandbytes"})
    monkeypatch.setenv("UNSLOTH_VLLM_BNB", "unsloth")
    assert vb.resolve()[2] == "unsloth" and calls == ["unsloth"]


def test_forced_provider_that_is_missing_raises(monkeypatch):
    vb = _load_vllm_bnb()
    _env(vb, monkeypatch, modules = NO_IN_TREE, specs = {"vllm", "bitsandbytes"})
    monkeypatch.setenv("UNSLOTH_VLLM_BNB", "vllm")
    with pytest.raises(ModuleNotFoundError):
        vb.resolve()


def _py_files():
    for d, _, files in os.walk(PKG):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(d, f)


def test_every_copied_file_keeps_its_apache_header_and_the_license_ships():
    for f in _py_files():
        head = open(f).read().splitlines()[:4]
        assert head[0] == "# SPDX-License-Identifier: Apache-2.0", f
        assert any("vllm-bnb-plugin" in line for line in head), f
    assert "Apache License" in open(os.path.join(PKG, "LICENSE")).read()
    assert "vllm_bnb/LICENSE" in open(os.path.join(ROOT, "pyproject.toml")).read()


def test_custom_op_name_cannot_collide_with_the_plugin():
    src = open(os.path.join(PKG, "quantization", "linear.py")).read()
    assert 'op_name="unsloth_apply_bnb_4bit"' in src and "torch.ops.vllm.unsloth_apply_bnb_4bit" in src
    assert 'op_name="apply_bnb_4bit"' not in src


def test_loader_never_imports_the_plugin_package():
    for f in _py_files():
        tree = ast.parse(open(f).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("vllm_bnb_plugin"), (f, node.module)
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("vllm_bnb_plugin") for a in node.names), f


def test_weights_mapper_falls_back_to_vllm_028_name():
    # vLLM 0.28.0 has WeightsMapper.get_unstacked_mapper only; >= 0.29 get_rename_mapper only.
    src = open(os.path.join(PKG, "bitsandbytes_loader.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_initialize_loader_state")
    seg = ast.get_source_segment(src, fn)
    assert "get_rename_mapper" in seg and "get_unstacked_mapper" in seg

    class Mapper028:
        def get_unstacked_mapper(self):
            return "m028"

    class Mapper029:
        def get_rename_mapper(self):
            return "m029"

    # Same expression as the loader.
    for mapper, want in ((Mapper028(), "m028"), (Mapper029(), "m029")):
        get = getattr(mapper, "get_rename_mapper", None) or getattr(mapper, "get_unstacked_mapper")
        assert get() == want
