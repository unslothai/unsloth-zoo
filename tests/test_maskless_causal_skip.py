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

"""create_causal_mask returns None (SDPA `is_causal=True`) only for a plain causal call; every other case keeps the mask."""
import inspect

import pytest
import torch
import transformers.masking_utils as masking_utils

from unsloth_zoo.temporary_patches import misc
from unsloth_zoo.temporary_patches.misc import patch_transformers_masks


@pytest.fixture(scope = "module")
def patched():
    patch_transformers_masks()
    return masking_utils


@pytest.fixture(autouse = True)
def _skip_enabled(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SKIP_CAUSAL_MASK", raising = False)


def _signature(patched):
    return inspect.signature(patched._unsloth_original_create_causal_mask)


def _config(attn = "sdpa", **overrides):
    from transformers import LlamaConfig
    config = LlamaConfig(
        hidden_size = 16, intermediate_size = 32, num_attention_heads = 2,
        num_key_value_heads = 2, num_hidden_layers = 2, vocab_size = 64, **overrides,
    )
    config._attn_implementation = attn
    return config


def _kwargs(patched, config = None, batch = 2, length = 6, attention_mask = "ones", **extra):
    params = _signature(patched).parameters
    embeds_name = "inputs_embeds" if "inputs_embeds" in params else "input_embeds"
    if isinstance(attention_mask, str) and attention_mask == "ones":
        attention_mask = torch.ones(batch, length, dtype = torch.long)
    kwargs = {
        "config": _config() if config is None else config,
        embeds_name: torch.zeros(batch, length, 16),
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": torch.arange(length)[None].expand(batch, -1),
    }
    if "cache_position" in params:
        kwargs["cache_position"] = torch.arange(length)
    kwargs.update(extra)
    return kwargs


def _decide(patched, **kwargs):
    return misc._maskless_causal_arguments(_signature(patched), (), kwargs)


def _reference(patched, kwargs):
    """What the unpatched compiled path builds: the original with the skip forced off."""
    import os
    previous = os.environ.get("UNSLOTH_SKIP_CAUSAL_MASK")
    os.environ["UNSLOTH_SKIP_CAUSAL_MASK"] = "0"
    try:
        return patched.create_causal_mask(**kwargs)
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_SKIP_CAUSAL_MASK", None)
        else:
            os.environ["UNSLOTH_SKIP_CAUSAL_MASK"] = previous


def _causal(length):
    return torch.ones(length, length, dtype = torch.bool).tril()




def test_plain_unpadded_sdpa_call_returns_none_and_is_counted(patched):
    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    assert patched.create_causal_mask(**_kwargs(patched)) is None
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before + 1


def test_no_mask_and_no_position_ids_also_skips(patched):
    kwargs = _kwargs(patched, attention_mask = None, position_ids = None)
    assert _decide(patched, **kwargs) is not None
    assert patched.create_causal_mask(**kwargs) is None


def test_the_mask_it_replaces_is_exactly_causal(patched):
    """The skipped call would otherwise have built the plain causal mask."""
    kwargs = _kwargs(patched, batch = 2, length = 6)
    mask = _reference(patched, kwargs)
    assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
    assert torch.equal(mask, _causal(6).expand(2, 1, 6, 6))


def test_sdpa_turns_none_into_the_same_causal_attention(patched):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 6, 8) for _ in range(3))
    module = torch.nn.Module()
    module.is_causal = True
    masked, _ = sdpa_attention_forward(module, q, k, v, _causal(6).expand(2, 1, 6, 6))
    maskless, _ = sdpa_attention_forward(module, q, k, v, None)
    torch.testing.assert_close(maskless, masked, rtol = 1e-5, atol = 1e-6)




@pytest.mark.parametrize("attn", ["eager", "flex_attention", "flash_attention_2", "flash_attention_3", None])
def test_only_sdpa_consumes_none_as_causal(patched, attn):
    # Eager applies no mask at all to None, so returning None there trains on future tokens.
    kwargs = _kwargs(patched, config = _config(attn))
    assert _decide(patched, **kwargs) is None


def test_eager_attention_still_gets_a_causal_mask(patched):
    kwargs = _kwargs(patched, config = _config("eager"))
    mask = patched.create_causal_mask(**kwargs)
    assert isinstance(mask, torch.Tensor)
    blocked = mask[0, 0] < -1 if mask.is_floating_point() else ~mask[0, 0]
    assert torch.equal(blocked, ~_causal(6))


def test_non_causal_config_keeps_its_mask(patched):
    config = _config()
    config.is_causal = False
    assert _decide(patched, **_kwargs(patched, config = config)) is None




def test_any_cache_keeps_the_mask(patched):
    from transformers import DynamicCache

    # `is_causal` is top-left aligned: wrong as soon as keys outnumber queries.
    try:
        cache = DynamicCache(config = _config())
    except TypeError:
        cache = DynamicCache()
    assert _decide(patched, **_kwargs(patched, past_key_values = cache)) is None


def test_single_query_keeps_the_mask(patched):
    assert _decide(patched, **_kwargs(patched, length = 1)) is None


def test_mask_longer_than_the_queries_keeps_the_mask(patched):
    kwargs = _kwargs(patched, length = 6, attention_mask = torch.ones(2, 9, dtype = torch.long))
    assert _decide(patched, **kwargs) is None




@pytest.mark.parametrize("side", ["right", "left"])
def test_padding_keeps_the_mask(patched, side):
    attention_mask = torch.ones(2, 6, dtype = torch.long)
    if side == "right":
        attention_mask[1, -2:] = 0
    else:
        attention_mask[1, :2] = 0
    kwargs = _kwargs(patched, attention_mask = attention_mask)
    assert _decide(patched, **kwargs) is None
    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    mask = patched.create_causal_mask(**kwargs)
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before
    assert isinstance(mask, torch.Tensor)
    assert torch.equal(mask, _reference(patched, kwargs))


def test_float_mask_keeps_the_mask(patched):
    kwargs = _kwargs(patched, attention_mask = torch.ones(2, 6))
    assert _decide(patched, **kwargs) is None


def _packed_position_ids(batch = 2):
    return torch.tensor([0, 1, 2, 0, 1, 2])[None].expand(batch, -1)


@pytest.mark.parametrize("with_mask", [False, True])
def test_packed_position_ids_keep_the_mask(patched, with_mask):
    # Padding-free packing: reset position_ids. Unsloth drops the attention mask here.
    kwargs = _kwargs(
        patched, attention_mask = "ones" if with_mask else None, position_ids = _packed_position_ids(),
    )
    assert _decide(patched, **kwargs) is None


def test_packed_batch_without_mask_never_attends_across_documents(patched):
    kwargs = _kwargs(patched, attention_mask = None, position_ids = _packed_position_ids())
    mask = patched.create_causal_mask(**kwargs)
    assert isinstance(mask, torch.Tensor)
    # 4.57.6 cannot tell packed from unpacked and masks causally; newer builds block the documents.
    assert not bool(mask[0, 0, 3, :3].any()) or torch.equal(mask, _reference(patched, kwargs))


def test_multi_stream_position_ids_keep_the_mask(patched):
    position_ids = torch.arange(6)[None, None].expand(3, 2, -1)
    assert _decide(patched, **_kwargs(patched, position_ids = position_ids)) is None




def test_a_prepared_4d_mask_is_returned_untouched(patched):
    mask = _causal(6).expand(2, 1, 6, 6).clone()
    kwargs = _kwargs(patched, attention_mask = mask)
    assert _decide(patched, **kwargs) is None
    assert patched.create_causal_mask(**kwargs) is mask


def test_a_block_mask_keeps_the_mask(patched):
    flex = pytest.importorskip("torch.nn.attention.flex_attention")
    block_mask = flex.create_block_mask(
        lambda b, h, q, kv: q >= kv, B = None, H = None, Q_LEN = 128, KV_LEN = 128, device = "cpu",
    )
    assert _decide(patched, **_kwargs(patched, length = 128, attention_mask = block_mask)) is None


@pytest.mark.parametrize("name", ["or_mask_function", "and_mask_function", "block_sequence_ids", "encoder_hidden_states"])
def test_overlays_keep_the_mask(patched, name):
    params = _signature(patched).parameters
    if name not in params and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        pytest.skip(f"this transformers has no {name}")
    value = (lambda b, h, q, kv: q < 3) if name.endswith("_function") else torch.zeros(2, 6, dtype = torch.long)
    kwargs = _kwargs(patched, **{name: value})
    assert _decide(patched, **kwargs) is None


def test_bidirectional_image_tokens_survive(patched):
    if "or_mask_function" not in _signature(patched).parameters:
        pytest.skip("this transformers has no or_mask_function")
    kwargs = _kwargs(patched, or_mask_function = lambda b, h, q, kv: (q < 3) & (kv < 3))
    mask = patched.create_causal_mask(**kwargs)
    assert isinstance(mask, torch.Tensor)
    assert bool(mask[0, 0, 0, 2])  # token 0 sees token 2: bidirectional block kept


def test_allow_is_causal_skip_false_keeps_the_mask(patched):
    if "allow_is_causal_skip" not in _signature(patched).parameters:
        pytest.skip("this transformers has no allow_is_causal_skip")
    assert _decide(patched, **_kwargs(patched, allow_is_causal_skip = False)) is None




def test_sliding_window_builder_is_untouched(patched):
    params = _signature(patched).parameters
    config = _config(sliding_window = 2)
    kwargs = _kwargs(patched, config = config)
    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    mask = patched.create_sliding_window_causal_mask(**kwargs)
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before
    assert isinstance(mask, torch.Tensor)
    assert not bool(mask[0, 0, 5, 0])


def test_tracing_keeps_the_compiled_path(patched, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert _decide(patched, **_kwargs(patched)) is None


def test_kill_switch(patched, monkeypatch):
    monkeypatch.setenv("UNSLOTH_SKIP_CAUSAL_MASK", "0")
    kwargs = _kwargs(patched)
    assert _decide(patched, **kwargs) is None
    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    assert isinstance(patched.create_causal_mask(**kwargs), torch.Tensor)
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before


def test_the_registered_sdpa_mask_interface_still_decides(patched, monkeypatch):
    sentinel = _causal(6).expand(2, 1, 6, 6).clone()
    mapping = patched.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    monkeypatch.setitem(mapping, "sdpa", lambda *args, **kwargs: sentinel)
    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    assert patched.create_causal_mask(**_kwargs(patched)) is sentinel
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before


def test_positional_call_binds(patched):
    kwargs = _kwargs(patched)
    args = []
    for name, parameter in _signature(patched).parameters.items():
        if name not in kwargs or parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            break
        args.append(kwargs.pop(name))
    assert len(args) >= 3
    assert misc._maskless_causal_arguments(_signature(patched), tuple(args), kwargs) is not None




def _tiny_model(attn):
    from transformers import LlamaForCausalLM
    import transformers.models.llama.modeling_llama as modeling_llama

    torch.manual_seed(0)
    model = LlamaForCausalLM(_config(attn)).eval()
    return model, modeling_llama


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_tiny_model_matches_the_masked_path_and_stays_causal(patched, monkeypatch, attn):
    model, modeling_llama = _tiny_model(attn)
    monkeypatch.setattr(modeling_llama, "create_causal_mask", patched.create_causal_mask, raising = False)
    ids = torch.randint(0, 64, (2, 12), generator = torch.Generator().manual_seed(1))
    attention_mask = torch.ones_like(ids)

    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    with torch.no_grad():
        head = model(input_ids = ids, attention_mask = attention_mask, use_cache = False).logits
    skipped = misc.CAUSAL_MASK_SKIP_STATS["skipped"] - before
    assert skipped == (1 if attn == "sdpa" else 0)

    monkeypatch.setenv("UNSLOTH_SKIP_CAUSAL_MASK", "0")
    with torch.no_grad():
        base = model(input_ids = ids, attention_mask = attention_mask, use_cache = False).logits
    monkeypatch.delenv("UNSLOTH_SKIP_CAUSAL_MASK")
    torch.testing.assert_close(head, base, rtol = 1e-5, atol = 1e-5)

    future = ids.clone()
    future[:, -1] = (future[:, -1] + 1) % 64
    with torch.no_grad():
        moved = model(input_ids = future, attention_mask = attention_mask, use_cache = False).logits
    torch.testing.assert_close(moved[:, :-1], head[:, :-1], rtol = 1e-5, atol = 1e-5)


def test_tiny_model_generate_is_unchanged(patched, monkeypatch):
    model, modeling_llama = _tiny_model("sdpa")
    monkeypatch.setattr(modeling_llama, "create_causal_mask", patched.create_causal_mask, raising = False)
    ids = torch.randint(0, 64, (2, 5), generator = torch.Generator().manual_seed(2))
    attention_mask = torch.ones_like(ids)
    attention_mask[1, 0] = 0  # left padded row
    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    head = model.generate(input_ids = ids, attention_mask = attention_mask, max_new_tokens = 6, do_sample = False)
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before  # a cache is always present
    monkeypatch.setenv("UNSLOTH_SKIP_CAUSAL_MASK", "0")
    base = model.generate(input_ids = ids, attention_mask = attention_mask, max_new_tokens = 6, do_sample = False)
    assert torch.equal(head, base)


def _register_flex(monkeypatch, reroutes):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    def flex_attention_forward(*args, **kwargs):
        raise AssertionError("not called")
    if reroutes:
        flex_attention_forward._unsloth_maskless_causal_sdpa = True
    mapping = getattr(ALL_ATTENTION_FUNCTIONS, "_global_mapping", None)
    target = mapping if mapping is not None else ALL_ATTENTION_FUNCTIONS
    monkeypatch.setitem(target, "flex_attention", flex_attention_forward)


def test_flex_drops_the_mask_only_when_unsloth_reroutes_it_to_sdpa(patched, monkeypatch):
    # Stock flex reads a None mask as full bidirectional attention.
    kwargs = _kwargs(patched, config = _config("flex_attention"))
    _register_flex(monkeypatch, reroutes = False)
    assert _decide(patched, **kwargs) is None
    assert patched.create_causal_mask(**kwargs) is not None

    _register_flex(monkeypatch, reroutes = True)
    before = misc.CAUSAL_MASK_SKIP_STATS["skipped"]
    assert patched.create_causal_mask(**kwargs) is None
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before + 1
    padded = _kwargs(patched, config = _config("flex_attention"),
                     attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]))
    assert patched.create_causal_mask(**padded) is not None


def test_a_model_local_flex_override_keeps_the_mask(patched, monkeypatch):
    # A modeling module's own AttentionInterface beats the global registry at dispatch.
    import sys, types
    from transformers.modeling_utils import AttentionInterface
    kwargs = _kwargs(patched, config = _config("flex_attention"))
    _register_flex(monkeypatch, reroutes = True)
    assert patched.create_causal_mask(**kwargs) is None

    interface = AttentionInterface()
    interface._local_mapping = {"flex_attention": lambda *args, **kw: None}
    module = types.ModuleType("transformers.models.fake_local_flex.modeling_fake_local_flex")
    module.ALL_ATTENTION_FUNCTIONS = interface
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert patched.create_causal_mask(**kwargs) is not None
