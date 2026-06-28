# Unsloth Zoo - Utilities for Unsloth
# Test that the MLX loader only coerces list-valued extra_special_tokens on
# transformers versions that would otherwise crash with
# "'list' object has no attribute 'keys'", and preserves the list on versions
# that support it (so v5 exports keep their special tokens).

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


@pytest.fixture
def base_init():
    # Save/restore PreTrainedTokenizerBase.__init__ around each test.
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    saved = PreTrainedTokenizerBase.__init__
    yield PreTrainedTokenizerBase
    PreTrainedTokenizerBase.__init__ = saved


def test_list_preserved_when_init_supports_lists(base_init):
    # v5-like baseline accepts a list -> the wrapper must NOT coerce it.
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens
    seen = {}
    base_init.__init__ = lambda self, **kw: seen.update(kw)
    _coerce_list_extra_special_tokens()
    base_init.__init__(object(), extra_special_tokens=["<a>", "<b>"])
    assert seen["extra_special_tokens"] == ["<a>", "<b>"]


def test_list_coerced_only_when_init_crashes(base_init):
    # old-transformers-like baseline raises the .keys() error on a list.
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens
    seen = {}

    def crashing_init(self, **kw):
        if isinstance(kw.get("extra_special_tokens"), list):
            raise AttributeError("'list' object has no attribute 'keys'")
        seen.update(kw)

    base_init.__init__ = crashing_init
    _coerce_list_extra_special_tokens()
    base_init.__init__(object(), extra_special_tokens=["<a>", "<b>"])
    assert seen["extra_special_tokens"] == {}  # coerced after the crash

    seen.clear()
    base_init.__init__(object(), extra_special_tokens={"image_token": "<img>"})
    assert seen["extra_special_tokens"] == {"image_token": "<img>"}  # dict untouched


def test_unrelated_attributeerror_not_swallowed(base_init):
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens

    def boom_init(self, **kw):
        raise AttributeError("unrelated boom")

    base_init.__init__ = boom_init
    _coerce_list_extra_special_tokens()
    with pytest.raises(AttributeError, match="unrelated"):
        base_init.__init__(object(), extra_special_tokens=["<a>"])


def test_idempotent_and_guard_shared(base_init):
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens
    base_init.__init__ = lambda self, **kw: None
    _coerce_list_extra_special_tokens()
    wrapped = base_init.__init__
    assert getattr(wrapped, "_unsloth_extra_special_tokens_patched", False) is True
    _coerce_list_extra_special_tokens()  # second call must not re-wrap
    assert base_init.__init__ is wrapped
