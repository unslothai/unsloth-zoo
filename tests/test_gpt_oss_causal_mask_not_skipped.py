# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""gpt-oss inference must never be handed a None causal mask.

`patch_GptOssModel` wraps `create_causal_mask` so that training with flex
attention skips dense mask creation -- flex builds its own BlockMask and a dense
one costs O(seq_len^2). The skip used to trigger on `input_embeds.requires_grad`
alone, which is NOT a training signal: unsloth calls `enable_input_require_grads()`
on every LoRA-capable load, so embeddings require grad during inference too.

Measured consequence on a B200 before the fix, `unsloth/gpt-oss-20b`, 163 tokens
of fixed prose: eager attention received `attention_mask=None`, i.e. no causal
masking at all, and perplexity went from 4.507 to 642.7 (bnb-4bit: 4.936 to
73.3). Output stayed fluent, so only a perplexity measurement exposed it. The
patch is skipped entirely under UNSLOTH_COMPILE_DISABLE, which is why the
uncompiled runs were correct and the compiled ones were not.

CPU-only: the wrapper is pure dispatch logic over a fabricated config.
"""
import os
import sys
import types

import pytest
import torch


@pytest.fixture
def wrapper(monkeypatch):
    """Install patch_GptOssModel's mask wrapper and hand back the live binding."""
    pytest.importorskip("transformers.masking_utils")
    import transformers.masking_utils as masking_utils

    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-20b")
    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)

    from unsloth_zoo.temporary_patches import gpt_oss

    if gpt_oss.UNSLOTH_COMPILE_DISABLE:
        pytest.skip("patch_GptOssModel is inert under UNSLOTH_COMPILE_DISABLE")

    calls = []

    def factory(*args, **kwargs):
        cfg = kwargs.get("config", None)
        if cfg is None:
            cfg = next((a for a in args if hasattr(a, "_attn_implementation")), None)
        # Record the VALUE at call time: the wrapper restores it afterwards, so
        # holding the object would only ever show the restored setting.
        calls.append({
            "kwargs": kwargs,
            "attn_implementation": getattr(cfg, "_attn_implementation", None),
        })
        return "DENSE_MASK"

    # Re-patching is gated on this sentinel, so clear it and restore every binding
    # the patch touches; conftest fails any test that leaks module state.
    for name in ("__patched_causal_mask__", "_old_create_causal_mask",
                 "_old_create_sliding_window_causal_mask"):
        if hasattr(masking_utils, name):
            monkeypatch.delattr(masking_utils, name, raising = False)
    monkeypatch.setattr(masking_utils, "create_causal_mask", factory, raising = False)
    monkeypatch.setattr(masking_utils, "create_sliding_window_causal_mask", factory, raising = False)
    monkeypatch.setattr(masking_utils, "create_masks_for_generate", factory, raising = False)
    import transformers.generation.utils as generation_utils
    monkeypatch.setattr(generation_utils, "create_masks_for_generate", factory, raising = False)

    gpt_oss.patch_GptOssModel()
    live = masking_utils.create_causal_mask
    if live is factory:
        pytest.skip("patch_GptOssModel declined to install (model gate or import)")
    return types.SimpleNamespace(fn = live, calls = calls)


def _config(attn_implementation):
    cfg = types.SimpleNamespace()
    cfg._attn_implementation = attn_implementation
    return cfg


def _embeds(requires_grad):
    x = torch.zeros(1, 4, 8, dtype = torch.float32)
    x.requires_grad_(requires_grad)
    return x


def test_eager_inference_gets_a_real_mask(wrapper):
    """The regression: requires_grad is set by unsloth, not by training."""
    with torch.enable_grad():
        got = wrapper.fn(
            config = _config("eager"),
            input_embeds = _embeds(True),
            attention_mask = None,
        )
    assert got == "DENSE_MASK"
    assert wrapper.calls, "the factory must actually be called"


def test_eager_inference_without_grad_gets_a_real_mask(wrapper):
    with torch.no_grad():
        got = wrapper.fn(
            config = _config("eager"),
            input_embeds = _embeds(False),
            attention_mask = None,
        )
    assert got == "DENSE_MASK"


def test_flex_training_still_skips_the_dense_mask(wrapper):
    """Preserved: flex builds its own BlockMask, and a dense mask at long
    context is the O(seq_len^2) allocation this skip exists to avoid."""
    with torch.enable_grad():
        got = wrapper.fn(
            config = _config("flex_attention"),
            input_embeds = _embeds(True),
            attention_mask = "RAW_2D_MASK",
        )
    assert got == "RAW_2D_MASK"
    assert not wrapper.calls


def test_flex_inference_under_no_grad_builds_the_mask(wrapper):
    """generate() runs under no_grad, where flex-with-KV-cache is not used."""
    with torch.no_grad():
        got = wrapper.fn(
            config = _config("flex_attention"),
            input_embeds = _embeds(True),
            attention_mask = None,
        )
    assert got == "DENSE_MASK"


def test_flex_config_is_swapped_to_eager_while_building(wrapper):
    """A BlockMask cannot be consumed by the eager forward, so the factory must
    be called with _attn_implementation temporarily set to eager, and restored."""
    cfg = _config("flex_attention")
    with torch.no_grad():
        wrapper.fn(config = cfg, input_embeds = _embeds(False), attention_mask = None)
    assert wrapper.calls[-1]["attn_implementation"] == "eager", "swapped for the call"
    assert cfg._attn_implementation == "flex_attention", "and restored afterwards"


def test_no_config_still_builds_a_mask(wrapper):
    """Unknown shape of call: prefer a correct mask over a skipped one."""
    with torch.enable_grad():
        got = wrapper.fn(input_embeds = _embeds(True), attention_mask = None)
    assert got == "DENSE_MASK"


def test_positional_embeds_are_recognised(wrapper):
    """4.x passes some of these positionally; the skip must not misfire."""
    with torch.enable_grad():
        got = wrapper.fn(_config("eager"), _embeds(True))
    assert got == "DENSE_MASK"
