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

"""Patched mask builders accept both `input_embeds` / `inputs_embeds` and a stale `cache_position`."""
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
    name = _embeds_name()
    try:
        _call(patched._unsloth_original_create_causal_mask, _other(name))
    except TypeError as error:
        assert "unexpected keyword argument" in str(error)
    else:
        # 5.4 to 5.5 still map `input_embeds` through a deprecate_kwarg alias of their own.
        pytest.skip("this transformers still accepts the other spelling itself")


def test_the_original_refuses_cache_position_when_it_dropped_it(patched):
    params = inspect.signature(patched._unsloth_original_create_causal_mask).parameters
    if "cache_position" in params:
        pytest.skip("this transformers still takes cache_position")
    with pytest.raises(TypeError, match = "cache_position"):
        _call(patched._unsloth_original_create_causal_mask, _embeds_name())


def test_patched_builder_takes_the_older_call_shape(patched):
    name = _embeds_name()
    out = _call(patched.create_causal_mask, _other(name))
    assert out is None or isinstance(out, torch.Tensor)


def test_patched_builder_accepts_both_spellings(patched):
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
    name = _embeds_name()
    mask = torch.zeros(1, 1, 4, 4)
    embeds = torch.zeros(1, 4, 8, requires_grad = True)
    for spelling in (name, _other(name)):
        out = _call(patched.create_causal_mask, spelling, attention_mask = mask)
        assert out is mask or torch.equal(out, mask)
        out = patched.create_causal_mask(
            config = None, **{spelling: embeds}, attention_mask = mask,
            cache_position = None, past_key_values = None,
        )
        assert out is mask


def test_aliases_are_installed_with_compilation_disabled():
    # Fresh interpreter: the flag is read at import.
    import subprocess, sys, os
    code = (
        "import os; os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'\n"
        "import torch\n"
        "import unsloth_zoo\n"
        "from unsloth_zoo.temporary_patches.misc import patch_transformers_masks\n"
        "import transformers.masking_utils as mu\n"
        "patch_transformers_masks()\n"
        "assert hasattr(mu, '_unsloth_original_create_causal_mask')\n"
        "import inspect\n"
        "p = inspect.signature(mu._unsloth_original_create_causal_mask).parameters\n"
        "name = 'inputs_embeds' if 'inputs_embeds' in p else 'input_embeds'\n"
        "other = 'input_embeds' if name == 'inputs_embeds' else 'inputs_embeds'\n"
        "from transformers import AutoConfig\n"
        "config = AutoConfig.for_model('llama', num_hidden_layers = 1, hidden_size = 8, num_attention_heads = 2)\n"
        "config._attn_implementation = 'sdpa'\n"
        "kw = {other: torch.zeros(1, 4, 8), 'attention_mask': torch.ones(1, 4, dtype = torch.long),"
        " 'cache_position': torch.arange(4), 'past_key_values': None, 'position_ids': torch.arange(4)[None]}\n"
        "mu.create_causal_mask(config = config, **kw)\n"
        "print('ALIAS_OK')\n"
    )
    env = dict(os.environ, UNSLOTH_COMPILE_DISABLE = "1")
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, env = env, timeout = 600)
    assert "ALIAS_OK" in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]


def test_generate_masks_keep_their_per_layer_mapping(patched):
    from transformers import AutoConfig

    original = patched._unsloth_original_create_masks_for_generate
    params = inspect.signature(original).parameters
    embeds = "inputs_embeds" if "inputs_embeds" in params else "input_embeds"
    config = AutoConfig.for_model(
        "gemma3_text", num_hidden_layers = 2, hidden_size = 8, num_attention_heads = 2,
        num_key_value_heads = 1, head_dim = 4, sliding_window = 2,
    )
    config._attn_implementation = "sdpa"
    if not getattr(config, "layer_types", None):
        pytest.skip("this transformers has no per-layer attention types")
    kwargs = {
        "config": config,
        embeds: torch.zeros(1, 4, 8, requires_grad = True),
        "attention_mask": torch.zeros(1, 1, 4, 4),
        "past_key_values": None,
        "position_ids": torch.arange(4).unsqueeze(0),
    }
    if "cache_position" in params:
        kwargs["cache_position"] = torch.arange(4)
    expected = original(**kwargs)
    got = patched.create_masks_for_generate(**kwargs)
    assert type(got) is type(expected)
    if isinstance(expected, dict):
        assert set(got) == set(expected)
