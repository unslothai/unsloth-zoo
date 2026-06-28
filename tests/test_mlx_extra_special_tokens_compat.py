# Unsloth Zoo - Utilities for Unsloth
# Test that the MLX loader coerces list-valued extra_special_tokens so tokenizer
# loading does not crash with "'list' object has no attribute 'keys'".

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def test_coerces_list_extra_special_tokens():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens

    saved = PreTrainedTokenizerBase.__init__
    seen = {}

    def spy(self, **kwargs):
        seen.clear()
        seen.update(kwargs)

    PreTrainedTokenizerBase.__init__ = spy  # unpatched baseline
    try:
        _coerce_list_extra_special_tokens()
        patched = PreTrainedTokenizerBase.__init__
        assert patched is not spy  # it wrapped the baseline

        # list -> {}
        patched(object(), extra_special_tokens=["<a>", "<b>"])
        assert seen["extra_special_tokens"] == {}

        # valid dict left untouched
        patched(object(), extra_special_tokens={"image_token": "<img>"})
        assert seen["extra_special_tokens"] == {"image_token": "<img>"}

        # idempotent: a second call does not re-wrap
        _coerce_list_extra_special_tokens()
        assert PreTrainedTokenizerBase.__init__ is patched
    finally:
        PreTrainedTokenizerBase.__init__ = saved


def test_guard_is_shared_with_temporary_patch():
    # The MLX coercion and patch_tokenizer_extra_special_tokens use the same
    # guard attribute, so applying one stops the other from double-wrapping.
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from unsloth_zoo.mlx.loader import _coerce_list_extra_special_tokens

    saved = PreTrainedTokenizerBase.__init__
    try:
        PreTrainedTokenizerBase.__init__ = lambda self, **kw: None
        _coerce_list_extra_special_tokens()
        wrapped = PreTrainedTokenizerBase.__init__
        assert getattr(wrapped, "_unsloth_extra_special_tokens_patched", False) is True
    finally:
        PreTrainedTokenizerBase.__init__ = saved
