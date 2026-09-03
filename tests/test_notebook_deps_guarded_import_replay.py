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

A transformers modeling file guards its dependency at module scope::

    from ...utils import is_timm_available, requires_backends
    if is_timm_available():
        import timm

A module imported while timm was missing therefore never binds the name, and
nothing revisits that later. Once the auto-installer makes `requires_backends`
succeed, the constructor runs on into `timm.create_model` and dies with a bare
`NameError` -- strictly worse than the ImportError the install replaced.

Reproduced against the real transformers timm_wrapper before this was fixed
(`REPRO NameError: name 'timm' is not defined`). The tests here use synthetic
modules on disk with the same shape so they need neither transformers nor a
missing package, and the "backend" is `json`, which is always importable, so
nothing can reach pip or the network.
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
    """Stand-in for transformers.utils.import_utils, backend unavailable."""
    module = types.SimpleNamespace(available = False)
    module.is_json_available = lambda: module.available
    return module


def _load(tmp_path, monkeypatch, name, source, guard_result):
    """Import `source` as `name` the way Python did when the guard was False."""
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
    # configuration_timm_wrapper guards `from timm.data import ImageNetInfo,
    # infer_imagenet_subset`, so binding the package name alone would not do.
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.configuration_fake", FROM_IMPORT, False
    )
    assert not hasattr(module, "JSONDecoder")

    iu.available = True
    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert hasattr(module, "JSONDecoder")


def test_a_negated_guard_is_not_executed(tmp_path, monkeypatch, iu):
    # `if not is_x_available():` is the fallback branch. Running it once the
    # package IS available would install a stub over the real thing.
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
    # The install can fail. Replaying then would bind a name whose import is
    # still going to fail, or mask the failure entirely.
    module = _load(
        tmp_path, monkeypatch, "transformers.models.fake.still_missing", PLAIN, False
    )

    notebook_deps._replay_skipped_guarded_imports(iu, BACKEND)

    assert not hasattr(module, BACKEND)


BROKEN = f"""
if {GUARD}():
    from {BACKEND} import this_name_does_not_exist
"""

# --- try/except ImportError, the shape that BINDS the name to None ----------
# transformers 5.5's tokenization_utils_sentencepiece, verbatim:
#     try:
#         import sentencepiece as spm
#     except ImportError:
#         spm = None
# and TokenizerBase.__init__ calls requires_backends("sentencepiece") before
# reaching spm.SentencePieceProcessor. hasattr reports the name as present, so
# an "is it missing" test based on hasattr never rebinds it.

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
    """A package that is genuinely absent at import time and present later.

    No simulation: the consumer below really does take its except branch,
    because the directory holding the package only joins sys.path afterwards,
    which is what pip installing it amounts to.
    """
    site = tmp_path / "site"
    site.mkdir()
    (site / f"{FAKE_BACKEND}.py").write_text("VALUE = 42\n", encoding = "utf-8")
    sys.modules.pop(FAKE_BACKEND, None)

    def install():
        # syspath_prepend only, never a bare sys.path.insert: the bare form is
        # not undone at teardown, so a later test's consumer would import the
        # package successfully at load time and never take its except branch,
        # which silently turns this whole file green for the wrong reason.
        monkeypatch.syspath_prepend(str(site))

    yield install
    # The module object caches the path it came from, so leaving it behind lets
    # the next test resolve it out of sys.modules from a tmp_path that is gone.
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
    # Only statements naming the backend just installed are replayed; another
    # optional import in the same module must not be re-attempted.
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
    # A try whose body is not purely imports is program logic, not an import
    # guard, and re-running it would repeat whatever else is in there.
    module = _load_consumer(tmp_path, monkeypatch, "transformers.logic_fake", LOGIC_CONSUMER)
    assert module.alias is None and module.SIDE_EFFECTS == []

    fake_backend()
    backend_iu.available = True
    notebook_deps._replay_skipped_guarded_imports(backend_iu, FAKE_BACKEND)

    assert module.alias is None
    assert module.SIDE_EFFECTS == []


def test_a_replay_that_fails_is_reported_not_swallowed(tmp_path, monkeypatch, iu):
    # A statement that cannot be replayed leaves the consumer unbound, so
    # swallowing it hands the caller the NameError this function exists to
    # prevent. It has to be visible and it has to be reported as a failure.
    _load(
        tmp_path, monkeypatch, "transformers.models.fake.modeling_broken", BROKEN, False
    )

    iu.available = True
    assert notebook_deps._replay_skipped_guarded_imports(iu, BACKEND) is False


def test_a_missing_name_is_reported_as_a_missing_name(tmp_path, monkeypatch, iu):
    # `from a import b` falls back to importing b as a submodule. When that is
    # not it either, the useful error names the attribute, not a module path
    # nobody wrote: "cannot import name 'ImageNetInfo' from 'timm.data'" rather
    # than "No module named 'timm.data.ImageNetInfo'".
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
    """The wrapper must replay, not just refresh the availability flag.

    Pins the ordering end to end: a consumer whose guarded import was skipped
    has the name bound by the time the retried `requires_backends` returns.
    """
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
