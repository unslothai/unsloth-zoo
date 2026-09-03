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
