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

"""Installing a backend does not un-dummy an export transformers already froze.

`transformers/__init__.py` picks the real module or a generated `utils/dummy_*_objects.py`
stub at import time, so a session imported without sentencepiece has `convert_slow_tokenizer`
bound to a body that is only `requires_backends(...)`. Once the auto-installer makes that
call succeed the stub runs to completion and returns None, so the wrapper has to refuse.
"""

from __future__ import annotations

import importlib

import pytest


notebook_deps = importlib.import_module("unsloth_zoo.temporary_patches.notebook_deps")

DUMMY_MODULE = "transformers.utils.dummy_sentencepiece_and_tokenizers_objects"


@pytest.fixture
def patched(monkeypatch):
    """Wrapper over a `requires_backends` that fails once, then succeeds, as a real install does."""
    iu = importlib.import_module("transformers.utils.import_utils")
    calls = {"n": 0}

    def original(obj, backends):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ImportError("requires the sentencepiece library")
        return None

    monkeypatch.setattr(iu, "requires_backends", original, raising = False)
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "1")
    for offline in ("UNSLOTH_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(offline, raising = False)
    monkeypatch.setattr(notebook_deps, "_try_install_and_import", lambda pkg: True)
    monkeypatch.setattr(
        notebook_deps, "_replay_skipped_guarded_imports", lambda iu, b: True
    )

    notebook_deps.patch_requires_backends_autoinstall()
    return iu.requires_backends


def _dummy_function():
    """Shaped like the real stub: calls requires_backends, then falls off the end."""


_dummy_function.__module__ = DUMMY_MODULE


class _DummyTokenizer:
    pass


_DummyTokenizer.__module__ = DUMMY_MODULE


class _RealTokenizer:
    pass


_RealTokenizer.__module__ = "transformers.models.t5.tokenization_t5"


def test_a_frozen_dummy_export_is_not_waved_through(patched):
    with pytest.raises(ImportError) as caught:
        patched(_dummy_function, ["sentencepiece"])

    message = str(caught.value)
    assert "restart" in message.lower(), message
    assert "sentencepiece" in message, "the original ImportError text is kept"


def test_a_dummy_instance_is_refused_too(patched):
    """`class AlbertTokenizer(metaclass=DummyObject)` passes `self`, not the class."""
    with pytest.raises(ImportError):
        patched(_DummyTokenizer(), ["sentencepiece"])


def test_a_real_object_still_succeeds_after_the_install(patched):
    """Non-vacuity: the whole point of the patch must keep working."""
    assert patched(_RealTokenizer, ["sentencepiece"]) is None


def test_transformers_really_freezes_this_export_onto_a_dummy():
    """Premise pin: if upstream stops generating dummy objects the guard above is dead code."""
    import inspect

    import transformers

    source = inspect.getsource(transformers)
    assert "dummy_sentencepiece_and_tokenizers_objects" in source, (
        "transformers no longer binds convert_slow_tokenizer to a dummy stub"
    )

    dummies = importlib.import_module(DUMMY_MODULE)
    body = inspect.getsource(dummies.convert_slow_tokenizer)
    assert "requires_backends" in body and "return" not in body, body
    assert notebook_deps._is_dummy_export(dummies.convert_slow_tokenizer)
