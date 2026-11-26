# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.
"""unsloth_zoo.vllm_bnb: in-tree bitsandbytes on vLLM <= 0.27.1, else vllm-bnb-plugin (registered first, with the
vLLM 0.28.0 mapper alias), else nothing. vLLM is not imported."""
import importlib.util
import os
import types

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load():
    # Load the file alone: importing unsloth_zoo would pull torch / vLLM.
    spec = importlib.util.spec_from_file_location("_vllm_bnb_resolve", os.path.join(ROOT, "unsloth_zoo", "vllm_bnb.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Mapper028:
    def get_unstacked_mapper(self):
        return "unstacked"


class _Mapper029:
    def get_rename_mapper(self):
        return "rename"


def _env(vb, monkeypatch, *, modules, specs, mapper = _Mapper029):
    calls = []
    utils = types.SimpleNamespace(WeightsMapper = mapper)

    def fake_import(name):
        if name == "vllm_bnb_plugin":
            return types.SimpleNamespace(register = lambda: calls.append("register"))
        if name == "vllm.model_executor.models.utils":
            return utils
        if name not in modules:
            raise ModuleNotFoundError(name)
        return types.SimpleNamespace(__name__ = name)

    monkeypatch.setattr(vb, "_import", fake_import)
    monkeypatch.setattr(vb, "_find_spec", lambda name: object() if name in specs else None)
    monkeypatch.setattr(vb, "_RESOLVED", None)
    return calls


def test_in_tree_wins_and_the_plugin_is_left_alone(monkeypatch):
    vb = _load()
    calls = _env(vb, monkeypatch, modules = set(vb._IN_TREE) | set(vb._PLUGIN), specs = {"vllm", "vllm_bnb_plugin"})
    quant, loader, source = vb.resolve()
    assert (quant.__name__, loader.__name__, source) == (*vb._IN_TREE, "vllm")
    assert calls == []


def test_plugin_is_registered_before_use(monkeypatch):
    vb = _load()
    calls = _env(vb, monkeypatch, modules = set(vb._PLUGIN), specs = {"vllm", "vllm_bnb_plugin"})
    quant, loader, source = vb.resolve()
    assert (quant.__name__, loader.__name__, source) == (*vb._PLUGIN, "vllm-bnb-plugin")
    assert calls == ["register"]
    assert vb.resolve()[2] == "vllm-bnb-plugin" and calls == ["register"]  # cached


def test_no_plugin_or_no_vllm_resolves_nothing(monkeypatch):
    vb = _load()
    _env(vb, monkeypatch, modules = set(), specs = {"vllm"})
    assert vb.resolve() == (None, None, None)
    _env(vb, monkeypatch, modules = set(vb._IN_TREE), specs = set())
    assert vb.resolve() == (None, None, None)


def test_vllm_028_gets_the_mapper_name_plugin_003_calls(monkeypatch):
    vb = _load()

    class Mapper(_Mapper028):
        pass

    _env(vb, monkeypatch, modules = set(vb._PLUGIN), specs = {"vllm", "vllm_bnb_plugin"}, mapper = Mapper)
    vb.resolve()
    assert Mapper().get_rename_mapper() == "unstacked"


def test_newer_vllm_mapper_is_untouched(monkeypatch):
    vb = _load()

    class Mapper(_Mapper029):
        pass

    _env(vb, monkeypatch, modules = set(vb._PLUGIN), specs = {"vllm", "vllm_bnb_plugin"}, mapper = Mapper)
    vb.resolve()
    assert Mapper().get_rename_mapper() == "rename" and not hasattr(Mapper, "get_unstacked_mapper")


def test_missing_plugin_message_names_the_install_command():
    src = open(os.path.join(ROOT, "unsloth_zoo", "vllm_utils.py")).read()
    assert "pip install vllm-bnb-plugin" in src and "_vllm_bnb_error" in src
