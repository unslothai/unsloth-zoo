# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The patched mask builders take the call shape of every transformers since 4.57.

transformers called the embeddings argument `input_embeds` up to 5.0 and
`inputs_embeds` after, and dropped `cache_position` from the builders after
5.0. Remote modeling code is written against one shape: Nemotron-H calls
`create_causal_mask(input_embeds = ..., cache_position = ...)`, which on
transformers 5.17 raised "unexpected keyword argument 'input_embeds'" and,
once that was renamed, "unexpected keyword argument 'cache_position'" (the
Step-3.7 failure) on the first training step. The patched builder now passes
whichever embeddings name this transformers accepts, drops `cache_position`
when it has no parameter for it, and the requires_grad short cut reads both
spellings.

Each test states which arm it measures; the failing arm is a tree whose
wrapper forwards the caller's spelling unchanged.
"""
import inspect

import pytest
import torch
import transformers.masking_utils as masking_utils

from unsloth_zoo.temporary_patches.misc import patch_transformers_masks


def _embeds_name():
    params = inspect.signature(masking_utils._unsloth_original_create_causal_mask).parameters
    return "inputs_embeds" if "inputs_embeds" in params else "input_embeds"


def _other(name):
    return "input_embeds" if name == "inputs_embeds" else "inputs_embeds"


@pytest.fixture(scope = "module")
def patched():
    patch_transformers_masks()
    assert hasattr(masking_utils, "_unsloth_original_create_causal_mask")
    return masking_utils


def _call(builder, name, **extra):
    from transformers import AutoConfig
    config = AutoConfig.for_model("llama", num_hidden_layers = 1, hidden_size = 8, num_attention_heads = 2)
    config._attn_implementation = "sdpa"
    embeds = torch.zeros(1, 4, 8)
    kwargs = {
        name: embeds,
        "attention_mask": torch.ones(1, 4, dtype = torch.long),
        "cache_position": torch.arange(4),
        "past_key_values": None,
        "position_ids": torch.arange(4)[None],
    }
    kwargs.update(extra)
    return builder(config = config, **kwargs)


def test_the_original_refuses_the_other_spelling(patched):
    """The precondition, on this transformers version."""
    name = _embeds_name()
    with pytest.raises(TypeError, match = "unexpected keyword argument"):
        _call(patched._unsloth_original_create_causal_mask, _other(name))


def test_the_original_refuses_cache_position_when_it_dropped_it(patched):
    """The precondition of the Step-3.7 failure, on a transformers past 5.0."""
    params = inspect.signature(patched._unsloth_original_create_causal_mask).parameters
    if "cache_position" in params:
        pytest.skip("this transformers still takes cache_position")
    with pytest.raises(TypeError, match = "cache_position"):
        _call(patched._unsloth_original_create_causal_mask, _embeds_name())


def test_patched_builder_takes_the_older_call_shape(patched):
    """Remote code's exact call: the older spelling plus cache_position."""
    name = _embeds_name()
    out = _call(patched.create_causal_mask, _other(name))
    assert out is None or isinstance(out, torch.Tensor)


def test_patched_builder_accepts_both_spellings(patched):
    """The arm that fails on a tree whose wrapper forwards the caller's spelling."""
    name = _embeds_name()
    native = _call(patched.create_causal_mask, name)
    remote = _call(patched.create_causal_mask, _other(name))
    if native is None:
        assert remote is None
    else:
        assert torch.equal(native, remote)


def test_sliding_window_builder_accepts_both_spellings(patched):
    name = _embeds_name()
    from transformers import AutoConfig
    config = AutoConfig.for_model("llama", num_hidden_layers = 1, hidden_size = 8, num_attention_heads = 2)
    config._attn_implementation = "sdpa"
    config.sliding_window = 2
    kwargs = dict(
        attention_mask = torch.ones(1, 4, dtype = torch.long),
        cache_position = torch.arange(4),
        past_key_values = None,
        position_ids = torch.arange(4)[None],
    )
    native = patched.create_sliding_window_causal_mask(config = config, **{name: torch.zeros(1, 4, 8)}, **kwargs)
    remote = patched.create_sliding_window_causal_mask(config = config, **{_other(name): torch.zeros(1, 4, 8)}, **kwargs)
    if native is None:
        assert remote is None
    else:
        assert torch.equal(native, remote)


def test_requires_grad_short_cut_reads_both_spellings(patched):
    """A 4-D mask handed in with grad-tracking embeddings comes straight back."""
    name = _embeds_name()
    mask = torch.zeros(1, 1, 4, 4)
    embeds = torch.zeros(1, 4, 8, requires_grad = True)
    for spelling in (name, _other(name)):
        out = _call(patched.create_causal_mask, spelling, attention_mask = mask)
        # the value under the other spelling is renamed first, so the short cut sees it
        assert out is mask or torch.equal(out, mask)
        out = patched.create_causal_mask(
            config = None, **{spelling: embeds}, attention_mask = mask,
            cache_position = None, past_key_values = None,
        )
        assert out is mask
