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

"""Without transformers.models.deepseek_v41 the patches resolve the native module from sys.modules."""

import sys
import types

import pytest

pytest.importorskip("torch")
from unsloth_zoo.temporary_patches import deepseek_v41 as patches  # noqa: E402


@pytest.fixture
def no_port(monkeypatch):
    real = patches.importlib.util.find_spec
    monkeypatch.setattr(
        patches.importlib.util, "find_spec",
        lambda name, *a: None if name == "transformers.models.deepseek_v41" else real(name, *a),
    )


def test_resolves_the_native_module_once_imported(no_port, monkeypatch):
    for name in [n for n in sys.modules if n.endswith("deepseek_v41.modeling")]:
        monkeypatch.delitem(sys.modules, name)
    assert patches._modeling() is None
    native = types.ModuleType("unsloth.models.deepseek_v41.modeling")
    native.DeepseekV41DecoderLayer = type("DeepseekV41DecoderLayer", (), {})
    monkeypatch.setitem(sys.modules, "unsloth.models.deepseek_v41.modeling", native)
    assert patches._modeling() is native


def test_ignores_unrelated_modules_with_the_same_suffix(no_port, monkeypatch):
    for name in [n for n in sys.modules if n.endswith("deepseek_v41.modeling")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "other.deepseek_v41.modeling", types.ModuleType("other.deepseek_v41.modeling"))
    assert patches._modeling() is None


@pytest.mark.parametrize("registered", [False, True], ids = ["unsloth_path", "registered"])
def test_adapter_of_a_native_model_resolves_without_the_base_config(monkeypatch, registered):
    import peft
    import unsloth_zoo.compiler as compiler
    import unsloth_zoo.hf_utils as hf_utils

    location = "some.registered.deepseek_v41.modeling" if registered else "unsloth.models.deepseek_v41.modeling"
    monkeypatch.setattr(hf_utils, "NATIVE_MODEL_TYPES", {"deepseek_v41"})
    monkeypatch.setattr(compiler, "NATIVE_MODELING_MODULES", {"deepseek_v41": location} if registered else {})
    config = peft.LoraConfig(r = 4, target_modules = ["q_a_proj"])
    config.base_model_name_or_path = "/nonexistent/unsloth-dsv41-base"
    config.auto_mapping = {"base_model_class": "DeepseekV41ForCausalLM", "parent_library": location}
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # setenv, not delenv: monkeypatch restores UNSLOTH_MODEL_NAME, which the function writes.
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "")
    assert hf_utils.get_transformers_model_type(config) == ["deepseek_v41"]


def test_recompilation_restores_originals_stored_under_the_registered_location(monkeypatch):
    import unsloth_zoo.compiler as compiler

    module = types.ModuleType("unsloth.models.fake_native.modeling")
    original, compiled = object(), object()
    module.Block = compiled
    monkeypatch.setattr(compiler, "_MODELING_FILE_ORIGINALS", {"alias.fake_native.modeling": {(None, "Block"): original}})
    monkeypatch.setattr("unsloth_zoo.temporary_patches.TEMPORARY_PATCHES", [])
    compiler._unpatch_modeling_file(module, "alias.fake_native.modeling")
    assert module.Block is original


def _fake_native(monkeypatch, root_name = "fakenativepkg"):
    root = types.ModuleType(root_name)
    models = types.ModuleType(f"{root_name}.models")
    package = types.ModuleType(f"{root_name}.models.fake_dsv")
    modeling = types.ModuleType(f"{root_name}.models.fake_dsv.modeling")
    root.models, models.fake_dsv, package.modeling = models, package, modeling
    for module in (root, models, package, modeling):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return root, modeling


def test_resolver_falls_back_to_the_native_module_when_transformers_fails_to_import(monkeypatch):
    import importlib
    import unsloth_zoo.compiler as compiler

    root, modeling = _fake_native(monkeypatch)
    monkeypatch.setitem(compiler.NATIVE_MODELING_MODULES, "fake_dsv", modeling.__name__)
    real = importlib.import_module

    def broken(name, *args, **kwargs):
        if name.startswith("transformers.models.fake_dsv"):
            raise ImportError("cannot import name X from a half-installed transformers")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(compiler.importlib, "import_module", broken)
    assert compiler.resolve_modeling_module("fake_dsv") == (modeling.__name__, modeling)


def test_resolver_rebinds_a_reimported_native_root(monkeypatch):
    import unsloth_zoo.compiler as compiler

    root, modeling = _fake_native(monkeypatch)
    monkeypatch.setitem(compiler.NATIVE_MODELING_MODULES, "fake_dsv", modeling.__name__)
    monkeypatch.delitem(compiler.__dict__, "fakenativepkg", raising = False)
    assert compiler.resolve_modeling_module("fake_dsv")[1] is modeling
    assert compiler.__dict__["fakenativepkg"] is root
    root2, modeling2 = _fake_native(monkeypatch)
    assert compiler.resolve_modeling_module("fake_dsv")[1] is modeling2
    assert compiler.__dict__["fakenativepkg"] is root2
    monkeypatch.delitem(compiler.__dict__, "fakenativepkg", raising = False)
