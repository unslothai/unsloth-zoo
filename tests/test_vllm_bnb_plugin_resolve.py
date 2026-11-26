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


class _Calls(list):
    pass


class _Mapper028:
    def get_unstacked_mapper(self):
        return "unstacked"


class _Mapper029:
    def get_rename_mapper(self):
        return "rename"


def _env(vb, monkeypatch, *, modules, specs, mapper = _Mapper029):
    calls = _Calls()
    utils = types.SimpleNamespace(WeightsMapper = mapper)

    plugin = types.SimpleNamespace(register = lambda: calls.append("register"))

    def fake_import(name):
        if name == "vllm_bnb_plugin":
            return plugin
        if name == "vllm.model_executor.models.utils":
            return utils
        if name not in modules:
            raise ModuleNotFoundError(name)
        return types.SimpleNamespace(__name__ = name)

    monkeypatch.setattr(vb, "_import", fake_import)
    monkeypatch.setattr(vb, "_find_spec", lambda name: object() if name in specs else None)
    monkeypatch.setattr(vb, "_RESOLVED", None)
    calls.plugin = plugin
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


def test_builtin_bitsandbytes_takes_the_plugin_off_vllms_allow_list(monkeypatch):
    # vLLM <= 0.27.1 still loads an installed plugin's entry point; importing it double-registers the bnb op.
    vb = _load()
    eps = [types.SimpleNamespace(name = n, group = g) for n, g in (
        ("bitsandbytes", "vllm.general_plugins"), ("lora_filesystem_resolver", "vllm.general_plugins"),
        ("cuda_alt", "vllm.platform_plugins"), ("other", "console_scripts"))]
    dists = [types.SimpleNamespace(entry_points = eps[:2]), types.SimpleNamespace(entry_points = eps[2:])]
    monkeypatch.setattr(vb.importlib.metadata, "distributions", lambda: dists)
    monkeypatch.delenv("VLLM_PLUGINS", raising = False)
    calls = _env(vb, monkeypatch, modules = set(vb._IN_TREE) | set(vb._PLUGIN), specs = {"vllm", "vllm_bnb_plugin"})
    assert vb.resolve()[2] == "vllm"
    assert vb.os.environ["VLLM_PLUGINS"] == "cuda_alt,lora_filesystem_resolver"
    assert calls == []  # never imported, never registered


def test_a_user_allow_list_only_loses_the_plugin(monkeypatch):
    vb = _load()
    monkeypatch.setenv("VLLM_PLUGINS", "bitsandbytes,mine")
    _env(vb, monkeypatch, modules = set(vb._IN_TREE), specs = {"vllm", "vllm_bnb_plugin"})
    vb.resolve()
    assert vb.os.environ["VLLM_PLUGINS"] == "mine"


def test_no_plugin_leaves_vllm_plugins_alone(monkeypatch):
    vb = _load()
    monkeypatch.delenv("VLLM_PLUGINS", raising = False)
    _env(vb, monkeypatch, modules = set(vb._IN_TREE), specs = {"vllm"})
    vb.resolve()
    assert "VLLM_PLUGINS" not in vb.os.environ


def test_install_plugin_uses_no_deps_and_the_pinned_spec(monkeypatch):
    vb = _load()
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode = 0, stdout = "ok", stderr = "")

    monkeypatch.setattr(vb.subprocess, "run", fake_run)
    monkeypatch.setattr(vb, "_find_spec", lambda name: object())
    ok, _ = vb.install_plugin()
    assert ok and seen["cmd"][-2:] == ["--no-deps", "vllm-bnb-plugin==0.0.3"]
    assert seen["cmd"][:4] == [vb.sys.executable, "-m", "pip", "install"]


def test_install_plugin_reports_a_failed_install(monkeypatch):
    vb = _load()
    monkeypatch.setattr(vb.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(returncode = 1, stdout = "", stderr = "no network"))
    monkeypatch.setattr(vb, "_find_spec", lambda name: object() if name == "pip" else None)
    ok, text = vb.install_plugin()
    assert not ok and "no network" in text


def test_load_vllm_tries_the_first_use_install_before_raising():
    src = open(os.path.join(ROOT, "unsloth_zoo", "vllm_utils.py")).read()
    assert "and not _ensure_vllm_bnb():" in src and "UNSLOTH_VLLM_BNB_AUTO_INSTALL" in src
