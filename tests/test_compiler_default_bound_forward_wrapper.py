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

"""Forward wrappers that bind the original through a default argument (Jamba/Zamba slow-path patch) must compile; emitted verbatim, _mixer is undefined."""

import ast
import importlib
import inspect
import re
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

from unsloth_zoo import compiler
from unsloth_zoo.temporary_patches.utils import torch_compiler_disable_unless_decode

MISC = Path(__file__).resolve().parents[1] / "unsloth_zoo" / "temporary_patches" / "misc.py"


def _load_patch():
    source = MISC.read_text(encoding = "utf-8")
    namespace = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_force_slow_path_on_raising_mamba_mixers":
            # Compile with the real file name and line numbers so inspect.getsource reads the wrapper exactly as production does.
            code = compile(ast.Module(body = [node], type_ignores = []), str(MISC), "exec")
            exec(code, namespace)
    assert "_force_slow_path_on_raising_mamba_mixers" in namespace
    return namespace["_force_slow_path_on_raising_mamba_mixers"]


force_slow_path = _load_patch()

MODELING = {
    "jamba" : ("transformers.models.jamba.modeling_jamba", "JambaMambaMixer"),
    "zamba" : ("transformers.models.zamba.modeling_zamba", "ZambaMambaMixer"),
}
CONFIGS = {
    "transformers.models.jamba.configuration_jamba" : "JambaConfig",
    "transformers.models.zamba.configuration_zamba" : "ZambaConfig",
}

# Fake mixers: one with a plain forward (transformers 4.x), one behind a bare-closure decorator like @force_accelerate_hooks (transformers 5.x).
_FAKE_MODELING_SOURCE = """
    import torch
    from torch import nn


    def fake_accelerate_hooks(child_module_name):
        def decorator(forward_func):
            def wrapped(self, *args, **kwargs):
                self.hook_calls = getattr(self, "hook_calls", 0) + 1
                return forward_func(self, *args, **kwargs)
            return wrapped
        return decorator


    class {name}(nn.Module):
        def __init__(self, config = None, layer_idx = 0):
            super().__init__()
            self.use_fast_kernels = True
            self.in_proj = nn.Linear(4, 4, bias = False)

        {decorator}
        def forward(self, hidden_states, cache_params = None, attention_mask = None, **kwargs):
            if self.use_fast_kernels:
                raise ValueError("Fast Mamba kernels are not available")
            real_body_ran = self.in_proj(hidden_states)
            return real_body_ran * 3.0
"""


def _install_fakes(tmp_path, monkeypatch, decorated):
    for module_name in CONFIGS:
        config_module = types.ModuleType(module_name)
        config_module.__dict__[CONFIGS[module_name]] = type(CONFIGS[module_name], (), {"__init__": lambda self: None})
        monkeypatch.setitem(sys.modules, module_name, config_module)
    modules = {}
    for key, (module_name, class_name) in MODELING.items():
        file_name = f"fake_default_bound_{key}_{int(decorated)}"
        path = tmp_path / f"{file_name}.py"
        path.write_text(textwrap.dedent(_FAKE_MODELING_SOURCE).format(
            name = class_name,
            decorator = '@fake_accelerate_hooks("conv1d")' if decorated else "",
        ), encoding = "utf-8")
        spec = importlib.util.spec_from_file_location(file_name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, file_name, module)
        spec.loader.exec_module(module)
        # The patch finds the module by its transformers name, the compiler via eval.
        monkeypatch.setitem(sys.modules, module_name, module)
        monkeypatch.setattr(compiler, file_name, module, raising = False)
        modules[key] = (file_name, class_name, module)
    return modules


def _assert_generated_is_self_contained(generated, class_name):
    assert re.search(r"\b_mixer\b", generated) is None
    assert "_slow_only" not in generated
    assert "__orig" not in generated
    assert "real_body_ran = self.in_proj(hidden_states)" in generated
    assert f"return {class_name}_forward(" in generated
    compile(generated, f"<{class_name}-codegen>", "exec")


@pytest.mark.parametrize("decorated", [False, True], ids = ["plain_forward", "bare_closure_decorated_forward"])
@pytest.mark.parametrize("key", sorted(MODELING))
def test_default_bound_wrapper_emits_the_real_forward(tmp_path, monkeypatch, key, decorated):
    modules = _install_fakes(tmp_path, monkeypatch, decorated)
    force_slow_path()
    file_name, class_name, module = modules[key]
    cls = getattr(module, class_name)

    assert cls.forward.__name__ == "_slow_only"
    assert "_mixer.forward" in inspect.getsource(cls.forward)

    generated = compiler.create_standalone_class(class_name, file_name, dir(module), disable = True)
    _assert_generated_is_self_contained(generated, class_name)
    assert "@torch_compiler_disable_unless_decode" in generated

    namespace = {
        "torch": torch, "nn": torch.nn, "torch_compile_options": {},
        "fake_accelerate_hooks": module.fake_accelerate_hooks,
        "torch_compiler_disable_unless_decode": torch_compiler_disable_unless_decode,
    }
    exec(generated, namespace)
    generated_cls = namespace[class_name]
    # The patch re-runs on the compiled class; the guard must still wrap it.
    monkeypatch.setattr(module, class_name, generated_cls)
    force_slow_path()
    assert generated_cls.forward.__name__ == "_slow_only"

    generated_model = generated_cls()
    eager_model = cls()
    eager_model.load_state_dict(generated_model.state_dict())
    hidden_states = torch.randn(2, 3, 4)
    assert torch.equal(generated_model(hidden_states), eager_model(hidden_states))
    assert generated_model.use_fast_kernels is False
    if decorated:
        assert generated_model.hook_calls == 1


def test_unwrap_follows_a_single_function_default():
    class Owner:
        def forward(self, x):
            return x
    def wrapper(self, *a, __orig = Owner.forward, **kw):
        return __orig(self, *a, **kw)
    assert compiler._unwrap_undecorated_method(wrapper, Owner.__qualname__) is Owner.__dict__["forward"]


def test_unwrap_leaves_ambiguous_defaults_alone():
    class Owner:
        def forward(self, x):
            return x
    def other(self, x):
        return x
    def wrapper(self, x, first = Owner.forward, second = other):
        return first(self, x)
    assert compiler._unwrap_undecorated_method(wrapper, Owner.__qualname__) is wrapper


def test_unwrap_prefers_closure_over_defaults():
    """A closure cell already resolved the wrapper before defaults were read."""
    class Owner:
        def forward(self, x):
            return x
    def unrelated(self, x):
        return x
    def make(inner):
        def wrapped(self, x, helper = unrelated):
            return inner(self, x)
        return wrapped
    wrapper = make(Owner.forward)
    assert compiler._unwrap_undecorated_method(wrapper, Owner.__qualname__) is Owner.__dict__["forward"]


def test_self_contained_renamed_patch_is_kept(tmp_path, monkeypatch):
    """Negative case: a renamed, self-contained forward with no function default is kept as is."""
    modules = _install_fakes(tmp_path, monkeypatch, decorated = False)
    file_name, class_name, module = modules["jamba"]
    cls = getattr(module, class_name)
    def patched_mixer_forward(self, hidden_states, cache_params = None, attention_mask = None, **kwargs):
        return self.in_proj(hidden_states) * 5.0
    monkeypatch.setattr(cls, "forward", patched_mixer_forward)
    generated = compiler.create_standalone_class(class_name, file_name, dir(module), disable = True)
    assert "def patched_mixer_forward(" in generated
    assert "return patched_mixer_forward(" in generated
    compile(generated, "<renamed-patch-codegen>", "exec")


@pytest.mark.parametrize("key", sorted(MODELING))
def test_installed_transformers_mixer_generates(monkeypatch, key):
    module_name, class_name = MODELING[key]
    try:
        module = importlib.import_module(module_name)
        config_module = importlib.import_module(module_name.replace("modeling_", "configuration_"))
    except Exception as error:
        pytest.skip(f"{module_name} does not import here: {error}")
    import transformers
    monkeypatch.setattr(compiler, "transformers", transformers, raising = False)
    cls = getattr(module, class_name)
    config_cls = getattr(config_module, CONFIGS[config_module.__name__])
    monkeypatch.setattr(cls, "forward", cls.__dict__["forward"])
    monkeypatch.setattr(config_cls, "__init__", config_cls.__dict__["__init__"])
    if not cls.__dict__.get("_unsloth_slow_only", False):
        monkeypatch.setattr(cls, "_unsloth_slow_only", False, raising = False)
        force_slow_path()
    assert cls.forward.__name__ == "_slow_only"

    generated = compiler.create_standalone_class(class_name, module_name, dir(module), disable = True)
    assert re.search(r"\b_mixer\b", generated) is None
    assert "_slow_only" not in generated
    assert f"return {class_name}_forward(" in generated
    compile(generated, f"<{class_name}-codegen>", "exec")
