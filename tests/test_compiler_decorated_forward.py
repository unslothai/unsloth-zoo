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

"""`create_standalone_class` against a forward hidden behind a decorator that
does not use `functools.wraps`.

transformers decorates `Qwen3_5GatedDeltaNet.forward` with
`@force_accelerate_hooks("conv1d")`, and
`transformers/integrations/accelerate.py` returns a bare inner closure with no
`functools.wraps`. The class attribute is therefore
`force_accelerate_hooks.<locals>.decorator.<locals>.wrapped`, so
`inspect.getsource` returned the wrapper (defined in another file, closing over
free variables that do not exist at module scope) and `inspect.signature`
returned `(self, *args, **kwargs)`. The compiler emitted the wrapper body at
module scope and generated

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return wrapped(self, *args, **kwargs)

which is `NameError: name 'args' is not defined` at the first forward.

The fakes below reproduce that decorator shape without needing transformers or
a GPU. The negative cases pin that every other decorator shape keeps the
pre-existing behaviour byte for byte.
"""

import functools
import importlib.util
import inspect
import sys
import textwrap

import pytest
import torch

from unsloth_zoo import compiler


def _unwrap_undecorated_method(func, owner_qualname):
    """Indirection so this module still imports on a build without the fix.
    A missing helper then fails the tests that exercise it, instead of
    erroring collection for the whole file and hiding the codegen tests."""
    return compiler._unwrap_undecorated_method(func, owner_qualname)


# Mirrors transformers/integrations/accelerate.py force_accelerate_hooks: an
# inner closure over (child_module_name, forward_func), returned WITHOUT
# functools.wraps.
_FAKE_MODELING_SOURCE = """
    import torch
    from torch import nn


    def fake_accelerate_hooks(child_module_name):
        def decorator(forward_func):
            def wrapped(self, *args, **kwargs):
                self.hook_calls = getattr(self, "hook_calls", 0) + 1
                self.hooked_child = child_module_name
                return forward_func(self, *args, **kwargs)
            return wrapped
        return decorator


    class FakeGatedDeltaNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(4, 4, bias = False)

        @fake_accelerate_hooks("conv1d")
        def forward(self, hidden_states, cache_params = None, attention_mask = None, **kwargs):
            real_body_ran = self.in_proj(hidden_states)
            return real_body_ran * 3.0


    class FakePlainNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(4, 4, bias = False)

        def forward(self, hidden_states, cache_params = None, **kwargs):
            return self.in_proj(hidden_states) * 3.0
"""


def _load_fake_modeling_module(tmp_path, monkeypatch, name):
    """Write the fake modeling file to disk so inspect.getsource can read it,
    import it, and expose it to the `eval(f"{model_location}.{module}")` lookup
    inside create_standalone_class."""
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(_FAKE_MODELING_SOURCE), encoding = "utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(compiler, name, module, raising = False)
    return module


def _generate(module_name, class_name, module):
    return compiler.create_standalone_class(
        class_name, module_name, dir(module), disable = True,
    )


def test_bare_closure_decorator_really_hides_the_method(tmp_path, monkeypatch):
    """The premise: without unwrapping, the class attribute reports the
    wrapper's name, signature and source, not the method's."""
    module = _load_fake_modeling_module(tmp_path, monkeypatch, "fake_bare_closure_premise")
    attribute = module.FakeGatedDeltaNet.forward

    assert attribute.__name__ == "wrapped"
    assert not hasattr(attribute, "__wrapped__")
    assert str(inspect.signature(attribute)) == "(self, *args, **kwargs)"
    assert "real_body_ran" not in inspect.getsource(attribute)


def test_decorated_forward_emits_the_real_body_and_no_unbound_name(tmp_path, monkeypatch):
    module = _load_fake_modeling_module(tmp_path, monkeypatch, "fake_bare_closure_codegen")
    generated = _generate("fake_bare_closure_codegen", "FakeGatedDeltaNet", module)

    # The real body is emitted, and the decorator's wrapper is not.
    assert "real_body_ran = self.in_proj(hidden_states)" in generated
    assert "def wrapped(" not in generated
    assert "return wrapped(" not in generated

    # The forwarding call is built from the real signature, so `args` / `kwargs`
    # are never referenced without being bound.
    assert "return FakeGatedDeltaNet_forward(" in generated
    assert "hidden_states=hidden_states" in generated
    assert "cache_params=cache_params" in generated
    assert "attention_mask=attention_mask" in generated

    compile(generated, "<fake-bare-closure-codegen>", "exec")


def test_decorated_forward_keeps_the_disable_classification(tmp_path, monkeypatch):
    """The wrapper's name is `wrapped`, not `forward`, so the renamed-forward
    heuristic used to classify the method as an Unsloth temporary patch and set
    `disable = None`, silently dropping the caller's compile decision."""
    module = _load_fake_modeling_module(tmp_path, monkeypatch, "fake_bare_closure_disable")

    disabled = _generate("fake_bare_closure_disable", "FakeGatedDeltaNet", module)
    assert "@torch.compiler.disable(recursive = False)" in disabled

    compiled = compiler.create_standalone_class(
        "FakeGatedDeltaNet", "fake_bare_closure_disable", dir(module), disable = False,
    )
    # A compile decorator, whichever spelling: fullgraph regions are emitted
    # as `torch_compile_with_fallback` so cache exhaustion cannot hard-fail.
    assert "@torch_compile_with_fallback(" in compiled or "@torch.compile(" in compiled


def test_decorated_forward_runs_and_fires_the_decorator_exactly_once(tmp_path, monkeypatch):
    """`inspect.getsource` on the real method returns the class-body text
    INCLUDING the decorator line, so the decorator travels onto the standalone
    function and is dropped from the emitted class method. It must therefore
    still fire exactly once, and the numerics must match eager."""
    module = _load_fake_modeling_module(tmp_path, monkeypatch, "fake_bare_closure_runtime")
    generated = _generate("fake_bare_closure_runtime", "FakeGatedDeltaNet", module)

    assert '@fake_accelerate_hooks("conv1d")' in generated
    assert generated.count('@fake_accelerate_hooks("conv1d")') == 1

    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "torch_compile_options": {},
        "fake_accelerate_hooks": module.fake_accelerate_hooks,
    }
    exec(generated, namespace)

    generated_model = namespace["FakeGatedDeltaNet"]()
    eager_model = module.FakeGatedDeltaNet()
    eager_model.load_state_dict(generated_model.state_dict())

    hidden_states = torch.randn(2, 3, 4)
    generated_out = generated_model(hidden_states)

    assert generated_model.hook_calls == 1
    assert generated_model.hooked_child == "conv1d"
    assert torch.equal(generated_out, eager_model(hidden_states))


def test_undecorated_forward_is_untouched(tmp_path, monkeypatch):
    """Negative case: a plain forward must generate exactly what it always did."""
    module = _load_fake_modeling_module(tmp_path, monkeypatch, "fake_bare_closure_plain")
    generated = _generate("fake_bare_closure_plain", "FakePlainNet", module)

    assert "def FakePlainNet_forward(self, hidden_states, cache_params = None, **kwargs):" in generated
    assert (
        "return FakePlainNet_forward(self, hidden_states=hidden_states, "
        "cache_params=cache_params, **kwargs)"
    ) in generated
    assert "@fake_accelerate_hooks" not in generated
    compile(generated, "<fake-bare-closure-plain>", "exec")


# ---------------------------------------------------------------------------
# _unwrap_undecorated_method must be inert for every other shape.
# ---------------------------------------------------------------------------

def test_unwrap_returns_plain_method_identically():
    class Plain:
        def forward(self, x):
            return x

    assert _unwrap_undecorated_method(Plain.forward, Plain.__qualname__) is Plain.forward


def test_unwrap_leaves_functools_wraps_decorators_alone():
    """functools.wraps copies __qualname__ (and sets __wrapped__, which
    inspect.getsource / inspect.signature already follow), so the wrapper is
    returned unchanged and behaviour is identical to before the fix."""
    def polite(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        return wrapper

    class Wrapped:
        @polite
        def forward(self, x):
            return x

    attribute = Wrapped.forward
    assert _unwrap_undecorated_method(attribute, Wrapped.__qualname__) is attribute


def test_unwrap_leaves_inherited_forward_alone():
    class Base:
        def forward(self, x):
            return x

    class Child(Base):
        pass

    assert _unwrap_undecorated_method(Child.forward, Child.__qualname__) is Base.forward


def test_unwrap_leaves_ambiguous_closures_alone():
    """Two functions in the closure is ambiguous, so nothing is unwrapped."""
    def ambiguous(func):
        def helper(x):
            return x
        def wrapper(self, *args, **kwargs):
            return func(self, helper(0), *args, **kwargs)
        return wrapper

    class Ambiguous:
        @ambiguous
        def forward(self, unused, x):
            return x

    attribute = Ambiguous.forward
    assert _unwrap_undecorated_method(attribute, Ambiguous.__qualname__) is attribute


def test_unwrap_refuses_a_function_from_another_class():
    """The recovered function must belong to the class being compiled."""
    class Other:
        def forward(self, x):
            return x

    def steal(_func):
        target = Other.forward
        def wrapper(self, *args, **kwargs):
            return target(self, *args, **kwargs)
        return wrapper

    class Thief:
        @steal
        def forward(self, x):
            return x

    attribute = Thief.forward
    assert _unwrap_undecorated_method(attribute, Thief.__qualname__) is attribute


def test_unwrap_survives_non_function_attributes():
    class Weird:
        forward = staticmethod(print)

    assert _unwrap_undecorated_method(Weird.forward, Weird.__qualname__) is Weird.forward


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_unwrap_walks_stacked_bare_decorators(depth):
    """Stacked bare closures still resolve to the real method."""
    def bare(func):
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        return wrapper

    class Stacked:
        def forward(self, x):
            return x

    real = Stacked.forward
    attribute = real
    for _ in range(depth):
        attribute = bare(attribute)

    assert _unwrap_undecorated_method(attribute, Stacked.__qualname__) is real
