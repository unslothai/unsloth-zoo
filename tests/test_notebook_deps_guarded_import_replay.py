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

"""Installing the backend is not enough: the guarded import has to be replayed.

A module imported while it was missing never binds the name under its
`if is_<backend>_available():` guard, so once `requires_backends` starts succeeding the
body dies on a bare `NameError` instead of the ImportError the install replaced.
"""

from __future__ import annotations

import ast
import importlib
import sys
import types

import pytest


notebook_deps = importlib.import_module("unsloth_zoo.temporary_patches.notebook_deps")

BACKEND = "json"
GUARD = "is_json_available"


@pytest.fixture
def iu():
    module = types.SimpleNamespace(available = False)
    module.is_json_available = lambda: module.available
    return module


def _load(tmp_path, monkeypatch, name, source, guard_result):
    path = tmp_path / (name.replace(".", "_") + ".py")
    path.write_text(source, encoding = "utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    namespace = vars(module)
    namespace[GUARD] = lambda: guard_result
    exec(compile(source, str(path), "exec"), namespace)
    monkeypatch.setitem(sys.modules, name, module)
    return module


PLAIN = f"""
if {GUARD}():
    import {BACKEND}


def use():
    return {BACKEND}.dumps({{}})
"""

FROM_IMPORT = f"""
if {GUARD}():
    from {BACKEND}.decoder import JSONDecoder
"""

NEGATED = f"""
if not {GUARD}():
    import {BACKEND} as should_not_be_bound
"""


def test_a_module_that_skipped_the_guard_gets_the_import_replayed(tmp_path, monkeypatch, iu):
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.modeling_fake", PLAIN, False
    )
    assert not hasattr(module, BACKEND), "precondition: the guard was skipped"

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert hasattr(module, BACKEND), (
        "the name the module guards is still unbound, so the caller that "
        "requires_backends now waves through raises NameError instead"
    )
    assert module.use() == "{}", "the function that uses it has to actually work"


def test_the_from_import_form_is_replayed_too(tmp_path, monkeypatch, iu):
    # configuration_timm_wrapper guards a from-import; the package name is not enough.
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.configuration_fake", FROM_IMPORT, False
    )
    assert not hasattr(module, "JSONDecoder")

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert hasattr(module, "JSONDecoder")


def test_a_negated_guard_is_not_executed(tmp_path, monkeypatch, iu):
    # The fallback branch: running it once available installs a stub over the real thing.
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.negated_fake", NEGATED, True
    )
    assert not hasattr(module, "should_not_be_bound")

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert not hasattr(module, "should_not_be_bound")


def test_a_module_that_already_has_the_name_is_left_alone(tmp_path, monkeypatch, iu):
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.ok_fake", PLAIN, True
    )
    before = getattr(module, BACKEND)
    assert before is not None

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert getattr(module, BACKEND) is before


def test_a_non_transformers_module_is_never_touched(tmp_path, monkeypatch, iu):
    module = _load(tmp_path, monkeypatch, "some_other_package.thing", PLAIN, False)

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert not hasattr(module, BACKEND), (
        "only transformers' own guarded imports are in scope; rewriting "
        "arbitrary third-party module namespaces is not"
    )


def test_nothing_runs_while_the_backend_is_still_unavailable(tmp_path, monkeypatch, iu):
    # The install can fail; replaying then binds a name whose import still fails.
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.still_missing", PLAIN, False
    )

    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert not hasattr(module, BACKEND)


BROKEN = f"""
if {GUARD}():
    from {BACKEND} import this_name_does_not_exist
"""

# The try/except shape BINDS the name to None, so a hasattr-based missing test never rebinds.

FAKE_BACKEND = "unsloth_replay_probe_pkg"

CONSUMER = f"""
try:
    import {FAKE_BACKEND} as alias
except ImportError:
    alias = None


def use():
    return alias.VALUE
"""

UNRELATED_CONSUMER = """
try:
    import unsloth_replay_probe_absent as other
except ImportError:
    other = None
"""

LOGIC_CONSUMER = f"""
SIDE_EFFECTS = []
try:
    import {FAKE_BACKEND} as alias
    SIDE_EFFECTS.append(1)
except ImportError:
    alias = None
"""


@pytest.fixture
def fake_backend(tmp_path, monkeypatch):
    """Absent at import, present later: the directory joins sys.path only afterwards."""
    site = tmp_path / "site"
    site.mkdir()
    (site / f"{FAKE_BACKEND}.py").write_text("VALUE = 42\n", encoding = "utf-8")
    sys.modules.pop(FAKE_BACKEND, None)

    def install():
        # syspath_prepend, not sys.path.insert: the bare form is not undone at teardown.
        monkeypatch.syspath_prepend(str(site))

    yield install
    # Leaving it behind lets the next test resolve it from a tmp_path that is gone.
    sys.modules.pop(FAKE_BACKEND, None)


@pytest.fixture
def backend_iu():
    module = types.SimpleNamespace(available = False)
    setattr(
        module,
        f"is_{FAKE_BACKEND}_available",
        lambda: module.available,
    )
    return module


def _load_consumer(tmp_path, monkeypatch, name, source):
    path = tmp_path / (name.replace(".", "_") + ".py")
    path.write_text(source, encoding = "utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), vars(module))
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_a_try_except_import_that_bound_none_is_replayed(
    tmp_path, monkeypatch, fake_backend, backend_iu
):
    module = _load_consumer(
        tmp_path, monkeypatch, "transformers.tokenization_fake", CONSUMER
    )
    assert hasattr(module, "alias"), "precondition: the handler BOUND the name"
    assert module.alias is None, "precondition: it bound it to None"

    fake_backend()
    backend_iu.available = True
    assert notebook_deps._replay_skipped_guarded_imports(backend_iu, FAKE_BACKEND) is True

    assert module.alias is not None, (
        "the name is still None, so the caller that requires_backends now waves "
        "through raises AttributeError on NoneType instead of using the package"
    )
    assert module.use() == 42


def test_a_try_except_for_a_different_package_is_left_alone(
    tmp_path, monkeypatch, fake_backend, backend_iu
):
    # Only statements naming the backend just installed are replayed.
    module = _load_consumer(
        tmp_path, monkeypatch, "transformers.unrelated_fake", UNRELATED_CONSUMER
    )
    assert module.other is None

    fake_backend()
    backend_iu.available = True
    assert notebook_deps._replay_skipped_guarded_imports(backend_iu, FAKE_BACKEND) is True
    assert module.other is None


def test_a_try_block_with_real_logic_is_not_re_run(
    tmp_path, monkeypatch, fake_backend, backend_iu
):
    # A try whose body is not purely imports is logic, and re-running repeats it.
    module = _load_consumer(tmp_path, monkeypatch, "transformers.logic_fake", LOGIC_CONSUMER)
    assert module.alias is None and module.SIDE_EFFECTS == []

    fake_backend()
    backend_iu.available = True
    notebook_deps._replay_skipped_guarded_imports(backend_iu, FAKE_BACKEND)

    assert module.alias is None
    assert module.SIDE_EFFECTS == []


def test_a_replay_that_fails_is_reported_not_swallowed(tmp_path, monkeypatch, iu):
    # A failed replay leaves the consumer unbound: the NameError this prevents.
    _load(
        tmp_path, monkeypatch, "transformers.models.fake.modeling_broken", BROKEN, False
    )

    iu.available = True
    assert notebook_deps._replay_skipped_guarded_imports(iu, BACKEND) is False


def test_a_missing_name_is_reported_as_a_missing_name(tmp_path, monkeypatch, iu):
    # The useful error names the attribute, not a module path nobody wrote.
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.modeling_msg", BROKEN, False
    )
    statement = ast.parse(BROKEN).body[0].body[0]

    with pytest.raises(ImportError) as exc:
        notebook_deps._perform_import(statement, module)
    assert "cannot import name 'this_name_does_not_exist'" in str(exc.value)
    assert "No module named" not in str(exc.value)


def test_the_installer_path_replays_before_letting_the_retry_through(
    tmp_path, monkeypatch, iu
):
    """The wrapper must replay, not just refresh the availability flag: a consumer whose
    guarded import was skipped is bound by the time the retried call returns."""
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.modeling_ordered", PLAIN, False
    )
    seen = {}

    import_utils = types.ModuleType("transformers.utils.import_utils")
    import_utils.is_json_available = lambda: import_utils.available
    import_utils.available = False
    import_utils.BACKENDS_MAPPING = {BACKEND: (lambda: import_utils.available, "needs {0}")}

    def _original(obj, backends):
        if not import_utils.available:
            raise ImportError("requires the json library")
        seen["bound_at_return"] = hasattr(module, BACKEND)

    import_utils.requires_backends = _original
    utils = types.ModuleType("transformers.utils")
    utils.import_utils = import_utils
    utils.requires_backends = _original
    root = types.ModuleType("transformers")
    root.utils = utils
    for name, mod in (
        ("transformers", root),
        ("transformers.utils", utils),
        ("transformers.utils.import_utils", import_utils),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")
    for off in ("UNSLOTH_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(off, raising = False)
    monkeypatch.setitem(notebook_deps._ALLOW_LIST, BACKEND, None)

    def _install(pkg):
        import_utils.available = True     # what a successful install looks like
        return True

    monkeypatch.setattr(notebook_deps, "_try_install_and_import", _install)

    notebook_deps.patch_requires_backends_autoinstall()
    import_utils.requires_backends(object(), [BACKEND])

    assert seen.get("bound_at_return") is True, (
        "the retry was allowed to succeed while the consumer module still had "
        "no binding for the backend it guards"
    )


# A guard body does not only import: transformers/audio_utils.py assigns TORCHCODEC_VERSION
# under its guard, and load_audio reads it as soon as the probe says yes. Replaying the
# imports alone turns the ImportError into a NameError.

ASSIGNS_STATE = f"""
RELOADS.append(1)
if {GUARD}():
    import {BACKEND}
    VERSION_TAG = {BACKEND}.dumps({{"v": 1}})


def use():
    return VERSION_TAG
"""

IMPORTS_ONLY = f"""
RELOADS.append(1)
if {GUARD}():
    import {BACKEND}
"""


def _load_reloadable(tmp_path, monkeypatch, iu, leaf, source):
    """A module importlib.reload can re-run: a real file plus a parent package in sys.modules
    whose __path__ contains it, which is what reload looks the spec up through."""
    package = "transformers.models.unsloth_reload_probe"
    directory = tmp_path / "pkg"
    directory.mkdir(exist_ok = True)
    parent = types.ModuleType(package)
    parent.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package, parent)

    path = directory / (leaf + ".py")
    path.write_text(source, encoding = "utf-8")
    name = f"{package}.{leaf}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.RELOADS = []
    setattr(module, GUARD, lambda: iu.available)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_a_guarded_assignment_is_bound_too(tmp_path, monkeypatch, iu):
    module = _load_reloadable(tmp_path, monkeypatch, iu, "assigns_state", ASSIGNS_STATE)
    assert not hasattr(module, "VERSION_TAG"), "precondition: the guard was skipped"

    iu.available = True
    assert notebook_deps._replay_skipped_guarded_imports(iu, BACKEND) is True

    assert hasattr(module, "VERSION_TAG"), (
        "the guard body's assignment is still unbound, so the caller the install was "
        "supposed to rescue raises NameError instead of working"
    )
    assert module.use() == '{"v": 1}'


def test_a_function_that_was_already_imported_elsewhere_sees_the_new_state(
    tmp_path, monkeypatch, iu
):
    """reload re-runs in the SAME __dict__, so `from x import use` elsewhere keeps working."""
    module = _load_reloadable(tmp_path, monkeypatch, iu, "stale_ref", ASSIGNS_STATE)
    stale = module.use

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert stale is not module.use, "reload rebinds the attribute"
    assert stale() == '{"v": 1}', "but the old function object shares the refreshed globals"


def test_a_module_with_only_guarded_imports_is_not_re_run(tmp_path, monkeypatch, iu):
    """Blast radius: re-running is the fallback for state an import replay cannot bind."""
    module = _load_reloadable(tmp_path, monkeypatch, iu, "imports_only", IMPORTS_ONLY)
    assert module.RELOADS == [1]

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert hasattr(module, BACKEND)
    assert module.RELOADS == [1], "the module was re-run when a plain import would do"


def test_the_re_run_really_is_what_binds_it(tmp_path, monkeypatch, iu):
    """Non-vacuity for the test above: this module IS re-run, and once only."""
    module = _load_reloadable(tmp_path, monkeypatch, iu, "counted", ASSIGNS_STATE)
    assert module.RELOADS == [1]

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert module.RELOADS == [1, 1]


def test_a_failing_re_run_is_reported_rather_than_waved_through(tmp_path, monkeypatch, iu):
    module = _load_reloadable(tmp_path, monkeypatch, iu, "boom", ASSIGNS_STATE)

    def _explode(mod):
        raise RuntimeError("re-import failed")

    monkeypatch.setattr(notebook_deps.importlib, "reload", _explode)
    iu.available = True

    assert notebook_deps._replay_skipped_guarded_imports(iu, BACKEND) is False, (
        "the wrapper re-raises the original ImportError on False; silently returning "
        "True here hands the caller a NameError instead"
    )


def test_the_transformers_package_itself_is_never_re_run(monkeypatch, iu):
    """The lazy-module entry point. Re-running it is far more than binding one name."""
    root = types.ModuleType("transformers")
    root.__file__ = "/nonexistent/transformers/__init__.py"
    tree = ast.parse(f"if {GUARD}():\n    STATE = 1\n")

    calls = []
    monkeypatch.setattr(notebook_deps.importlib, "reload", lambda m: calls.append(m))

    assert notebook_deps._rerun_for_guarded_state(root, tree, GUARD, BACKEND) is True
    assert calls == []
