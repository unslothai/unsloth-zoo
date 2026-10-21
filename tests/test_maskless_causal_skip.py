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
    monkeypatch.delenv("UNSLOTH_SKIP_RIGHT_PADDING_MASK", raising = False)


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


def _compile_disabled():
    """patch_transformers_masks wraps the stock builders uncompiled under UNSLOTH_COMPILE_DISABLE=1 or partial."""
    from unsloth_zoo.temporary_patches import common
    return bool(common.UNSLOTH_COMPILE_DISABLE)


def _reference(patched, kwargs):
    """What the unpatched compiled path builds: the original with the skip forced off.

    Compiled, transformers sees is_compiling() and always builds the mask. Under
    UNSLOTH_COMPILE_DISABLE=1 the wrapper is the stock function, whose own SDPA
    shortcut returns None for an unpadded causal batch, so the reference reports
    compiling the same way to get the mask the compiled path would have built.
    """
    import os
    from unittest import mock
    previous = os.environ.get("UNSLOTH_SKIP_CAUSAL_MASK")
    os.environ["UNSLOTH_SKIP_CAUSAL_MASK"] = "0"
    try:
        if _compile_disabled():
            with mock.patch.object(torch.compiler, "is_compiling", lambda: True):
                return patched.create_causal_mask(**kwargs)
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
def test_only_sdpa_consumes_none_as_causal(patched, attn, monkeypatch):
    # Eager applies no mask at all to None, so returning None there trains on future tokens.
    if attn == "flex_attention":
        _register_flex(monkeypatch, reroutes = False)  # stock flex; an imported unsloth reroutes
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
    mask = patched.create_causal_mask(**kwargs)
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before
    if _compile_disabled():
        # Switched off, the call is exactly the stock one, which may itself skip the mask.
        stock = patched._unsloth_original_create_causal_mask(**kwargs)
        assert (mask is None) == (stock is None)
        assert stock is None or torch.equal(mask, stock)
    else:
        assert isinstance(mask, torch.Tensor)


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
        flex_attention_forward._unsloth_maskless_causal_sdpa_accepts = lambda config: True
    mapping = getattr(ALL_ATTENTION_FUNCTIONS, "_global_mapping", None)
    target = mapping if mapping is not None else ALL_ATTENTION_FUNCTIONS
    monkeypatch.setitem(target, "flex_attention", flex_attention_forward)
    # Dispatch reads the local mapping first; an imported unsloth registers its flex wrapper there.
    local = getattr(ALL_ATTENTION_FUNCTIONS, "_local_mapping", None)
    if isinstance(local, dict) and "flex_attention" in local:
        monkeypatch.setitem(local, "flex_attention", flex_attention_forward)


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


def test_flex_keeps_the_mask_when_the_wrapper_vetoes_the_config(patched, monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    kwargs = _kwargs(patched, config = _config("flex_attention"))
    _register_flex(monkeypatch, reroutes = True)
    function = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    monkeypatch.setattr(function, "_unsloth_maskless_causal_sdpa_accepts", lambda config: False)
    assert patched.create_causal_mask(**kwargs) is not None
    monkeypatch.delattr(function, "_unsloth_maskless_causal_sdpa_accepts")
    assert patched.create_causal_mask(**kwargs) is not None


def test_a_model_local_flex_override_keeps_the_mask(patched, monkeypatch):
    # A modeling module's own AttentionInterface beats the global registry at dispatch.
    import sys, types
    from transformers.modeling_utils import AttentionInterface
    kwargs = _kwargs(patched, config = _config("flex_attention"))
    _register_flex(monkeypatch, reroutes = True)
    assert patched.create_causal_mask(**kwargs) is None

    # transformers.models.*, and trust_remote_code's transformers_modules.* under any name.
    for name, attribute in (
        ("transformers.models.fake_local_flex.modeling_fake_local_flex", "ALL_ATTENTION_FUNCTIONS"),
        ("transformers_modules.someone.fake_remote.modeling_fake_remote", "MY_ATTENTION"),
    ):
        interface = AttentionInterface()
        interface._local_mapping = {"flex_attention": lambda *args, **kw: None}
        module = types.ModuleType(name)
        setattr(module, attribute, interface)
        with monkeypatch.context() as scoped:
            scoped.setitem(sys.modules, name, module)
            assert patched.create_causal_mask(**kwargs) is not None, name
        assert patched.create_causal_mask(**kwargs) is None


def test_a_local_flex_registered_after_the_scan_keeps_the_mask(patched, monkeypatch):
    # The scan is cached; registering a local flex on an already-loaded interface must invalidate it.
    import sys, types
    from transformers.modeling_utils import AttentionInterface
    kwargs = _kwargs(patched, config = _config("flex_attention"))
    _register_flex(monkeypatch, reroutes = True)
    interface = AttentionInterface()
    module = types.ModuleType("transformers_modules.someone.late_flex.modeling_late_flex")
    module.ATTENTION = interface
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert patched.create_causal_mask(**kwargs) is None  # scanned and cached: nothing local yet
    interface["flex_attention"] = lambda *args, **kw: None
    assert patched.create_causal_mask(**kwargs) is not None
    del interface["flex_attention"]
    assert patched.create_causal_mask(**kwargs) is None


def _right_padded(batch = 3, length = 8):
    attention_mask = torch.ones(batch, length, dtype = torch.long)
    attention_mask[1, -3:] = 0
    attention_mask[2, -5:] = 0
    return attention_mask


def test_right_padding_skips_the_mask_when_opted_in(patched, monkeypatch):
    monkeypatch.setattr(misc, "_flash_sdpa_takes_causal", lambda config, embeds: True)
    monkeypatch.setenv("UNSLOTH_SKIP_RIGHT_PADDING_MASK", "1")
    kwargs = _kwargs(patched, batch = 3, length = 8, attention_mask = _right_padded())
    before = dict(misc.CAUSAL_MASK_SKIP_STATS)
    assert patched.create_causal_mask(**kwargs) is None
    assert misc.CAUSAL_MASK_SKIP_STATS["skipped"] == before["skipped"] + 1
    assert misc.CAUSAL_MASK_SKIP_STATS["right_padded"] == before["right_padded"] + 1
    # Every real query row of the mask it replaces is exactly causal.
    reference = _reference(patched, kwargs)
    real = _right_padded().bool()
    for row in range(3):
        n = int(real[row].sum())
        assert torch.equal(reference[row, 0, :n, :n], _causal(n))
        assert not reference[row, 0, :n, n:].any()


def test_right_padding_skip_is_on_by_default(patched, monkeypatch):
    monkeypatch.setattr(misc, "_flash_sdpa_takes_causal", lambda config, embeds: True)
    kwargs = _kwargs(patched, batch = 3, length = 8, attention_mask = _right_padded())
    before = misc.CAUSAL_MASK_SKIP_STATS["right_padded"]
    assert patched.create_causal_mask(**kwargs) is None
    assert misc.CAUSAL_MASK_SKIP_STATS["right_padded"] == before + 1


@pytest.mark.parametrize("env", [("UNSLOTH_SKIP_RIGHT_PADDING_MASK", "0"), ("UNSLOTH_SKIP_CAUSAL_MASK", "0")])
def test_right_padding_kill_switch_keeps_the_mask(patched, monkeypatch, env):
    monkeypatch.setattr(misc, "_flash_sdpa_takes_causal", lambda config, embeds: True)
    monkeypatch.setenv(*env)
    kwargs = _kwargs(patched, batch = 3, length = 8, attention_mask = _right_padded())
    before = misc.CAUSAL_MASK_SKIP_STATS["right_padded"]
    assert isinstance(patched.create_causal_mask(**kwargs), torch.Tensor)
    assert misc.CAUSAL_MASK_SKIP_STATS["right_padded"] == before


@pytest.mark.parametrize("attention_mask", [
    torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]]),
    torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1]]),
    torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]]),
], ids = ["left", "hole", "empty_row"])
def test_other_padding_keeps_the_mask_when_opted_in(patched, monkeypatch, attention_mask):
    monkeypatch.setattr(misc, "_flash_sdpa_takes_causal", lambda config, embeds: True)
    monkeypatch.setenv("UNSLOTH_SKIP_RIGHT_PADDING_MASK", "1")
    kwargs = _kwargs(patched, batch = 2, length = 4, attention_mask = attention_mask)
    assert _decide(patched, **kwargs) is None
    assert isinstance(patched.create_causal_mask(**kwargs), torch.Tensor)


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_right_padded_training_matches_the_masked_path(patched, monkeypatch, attn):
    from transformers import LlamaForCausalLM
    import transformers.models.llama.modeling_llama as modeling_llama
    monkeypatch.setattr(modeling_llama, "create_causal_mask", patched.create_causal_mask, raising = False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(_config(attn)).train()
    ids = torch.randint(0, 64, (3, 8), generator = torch.Generator().manual_seed(1))
    attention_mask = _right_padded()
    labels = ids.masked_fill(attention_mask == 0, -100)

    def run():
        model.zero_grad(set_to_none = True)
        hidden = model.model(input_ids = ids, attention_mask = attention_mask, use_cache = False).last_hidden_state
        logits = model.lm_head(hidden)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return loss.detach(), hidden.detach(), grads

    base_loss, base_logits, base_grads = run()
    monkeypatch.setenv("UNSLOTH_SKIP_RIGHT_PADDING_MASK", "1")
    monkeypatch.setattr(misc, "_flash_sdpa_takes_causal", lambda config, embeds: True)
    before = misc.CAUSAL_MASK_SKIP_STATS["right_padded"]
    head_loss, head_logits, head_grads = run()
    assert misc.CAUSAL_MASK_SKIP_STATS["right_padded"] - before == (1 if attn == "sdpa" else 0)

    torch.testing.assert_close(head_loss, base_loss, rtol = 1e-5, atol = 1e-6)
    real = attention_mask.bool()
    torch.testing.assert_close(head_logits[real], base_logits[real], rtol = 1e-5, atol = 1e-5)
    assert head_grads.keys() == base_grads.keys()
    for name in base_grads:
        torch.testing.assert_close(head_grads[name], base_grads[name], rtol = 1e-4, atol = 1e-6)

    # The pad tail cannot reach a real token.
    changed = ids.clone()
    changed[~real] = (changed[~real] + 7) % 64
    with torch.no_grad():
        moved = model.model(input_ids = changed, attention_mask = attention_mask, use_cache = False).last_hidden_state
    torch.testing.assert_close(moved[real], head_logits[real], rtol = 1e-5, atol = 1e-5)


def test_right_padding_keeps_the_mask_without_flash_sdpa(patched, monkeypatch):
    # fp32 / CPU / pre-sm80 SDPA would run enable_gqa on the math kernel: keep the mask there.
    monkeypatch.setenv("UNSLOTH_SKIP_RIGHT_PADDING_MASK", "1")
    kwargs = _kwargs(patched, batch = 3, length = 8, attention_mask = _right_padded())
    assert _decide(patched, **kwargs) is None
    assert isinstance(patched.create_causal_mask(**kwargs), torch.Tensor)
    config = _config()
    assert not misc._flash_sdpa_takes_causal(config, torch.zeros(1, 2, 16, dtype = torch.bfloat16))
    assert not misc._flash_sdpa_takes_causal(config, torch.zeros(1, 2, 16))


def test_flash_sdpa_check_reads_heterogeneous_head_dims(monkeypatch):
    # transformers >= 5.17 raises on config.head_dim when it varies per layer (Gemma4Unified).
    try:
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
    except Exception:
        pytest.skip("this transformers has no heterogeneous Gemma4UnifiedTextConfig")
    import types
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (9, 0))
    monkeypatch.setattr(misc, "_flash_sdpa_accepts", lambda *args: True)
    embeds = types.SimpleNamespace(dtype = torch.bfloat16, is_cuda = True, device = "cuda:0")
    wide = Gemma4UnifiedTextConfig(num_hidden_layers = 4)
    assert wide.is_heterogeneous
    assert misc._flash_sdpa_takes_causal(wide, embeds) is False  # 512-dim full-attention layers
    narrow = Gemma4UnifiedTextConfig(num_hidden_layers = 4, global_head_dim = 128)
    assert misc._flash_sdpa_takes_causal(narrow, embeds) is True
    assert misc._flash_sdpa_takes_causal(_config(), embeds) is True


def _flash_rule(capability, torch_2_6):
    # can_use_flash_attention for dense square causal enable_gqa inputs (torch 2.6 / 2.13 sdp_utils.cpp).
    def accepts(device, dtype, num_heads, num_kv_heads, head_dim, dropout, requires_grad):
        if torch_2_6:
            if not ((8, 0) <= capability <= (9, 0)): return False
            narrow = (8, 6) <= capability <= (8, 9)
        else:
            if not ((8, 0) <= capability <= (12, 1)): return False
            narrow = (8, 6) <= capability <= (8, 9) or (12, 0) <= capability <= (12, 1)
        if requires_grad and narrow and (192 < head_dim <= 224 or (head_dim > 224 and dropout > 0)):
            return False
        return True
    return accepts


@pytest.mark.parametrize("torch_2_6", [True, False], ids = ["torch2.6", "torch2.13"])
@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize("head_dim", [128, 200, 256])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("grad", [True, False])
def test_flash_sdpa_check_defers_to_the_dispatcher(monkeypatch, torch_2_6, capability, head_dim, dropout, grad):
    # Gemma-2B GQA training (head_dim 256, 8:1 heads, dropout 0.1) on sm86 / sm89 must keep the mask:
    # without it flash refuses and SDPA falls to the quadratic math kernel.
    import types
    from transformers import GemmaConfig
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: capability)
    seen = []
    rule = _flash_rule(capability, torch_2_6)
    def fake_accepts(*args):
        seen.append(args)
        return rule(*args)
    monkeypatch.setattr(misc, "_flash_sdpa_accepts", fake_accepts)
    config = GemmaConfig(hidden_size = 2048, num_attention_heads = 8, num_key_value_heads = 1,
                         head_dim = head_dim, attention_dropout = dropout)
    config._attn_implementation = "sdpa"
    embeds = types.SimpleNamespace(dtype = torch.bfloat16, is_cuda = True, device = "cuda:0")
    with torch.set_grad_enabled(grad):
        got = misc._flash_sdpa_takes_causal(config, embeds)
    assert seen == [("cuda:0", torch.bfloat16, 8, 1, head_dim, dropout, grad)]
    assert got is rule("cuda:0", torch.bfloat16, 8, 1, head_dim, dropout, grad)


def test_flash_sdpa_check_keeps_the_mask_when_the_probe_fails(monkeypatch):
    import types
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (9, 0))
    monkeypatch.setattr(misc, "_FLASH_SDPA_ACCEPTS_CACHE", {})
    monkeypatch.setattr(misc, "_EFFICIENT_SDPA_ACCEPTS_CACHE", {}, raising = False)
    def broken(*args, **kwargs):
        raise RuntimeError("no SDPAParams")
    monkeypatch.setattr(torch.backends.cuda, "SDPAParams", broken)
    embeds = types.SimpleNamespace(dtype = torch.bfloat16, is_cuda = True, device = "cuda:0")
    assert misc._flash_sdpa_takes_causal(_config(), embeds) is False


@pytest.mark.parametrize("kv_heads", [8, 1], ids = ["mha", "gqa"])
@pytest.mark.parametrize("efficient", [True, False])
@pytest.mark.parametrize("efficient_enabled", [True, False])
def test_mha_rejected_by_flash_runs_maskless_on_the_efficient_kernel(monkeypatch, kv_heads, efficient, efficient_enabled):
    # sm86 / sm89 / sm120 flash refuses some training shapes. Equal head counts need no enable_gqa,
    # so mem-efficient SDPA still runs the maskless causal call; GQA would fall to the math kernel.
    import types
    from transformers import LlamaConfig
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: efficient_enabled)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (8, 9))
    monkeypatch.setattr(misc, "_flash_sdpa_accepts", lambda *args: False)
    asked = []
    def efficient_accepts(*args):
        asked.append(args)
        return efficient
    monkeypatch.setattr(misc, "_efficient_sdpa_accepts_mha", efficient_accepts, raising = False)
    config = LlamaConfig(hidden_size = 2048, num_attention_heads = 8, num_key_value_heads = kv_heads, head_dim = 256)
    config._attn_implementation = "sdpa"
    embeds = types.SimpleNamespace(dtype = torch.bfloat16, is_cuda = True, device = "cuda:0")
    got = misc._flash_sdpa_takes_causal(config, embeds)
    assert got is (kv_heads == 8 and efficient and efficient_enabled)
    if kv_heads == 1 or not efficient_enabled:
        assert asked == []


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "needs an NVIDIA GPU")
def test_efficient_mha_probe_matches_a_real_maskless_call(monkeypatch):
    monkeypatch.setattr(misc, "_EFFICIENT_SDPA_ACCEPTS_CACHE", {}, raising = False)
    device = torch.device("cuda", torch.cuda.current_device())
    assert misc._efficient_sdpa_accepts_mha(device, torch.bfloat16, 8, 128, 0.0, True) is True
    q = torch.randn(1, 8, 64, 128, device = device, dtype = torch.bfloat16, requires_grad = True)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal = True)
        out.float().sum().backward()
    assert torch.isfinite(q.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason = "needs an NVIDIA GPU")
def test_flash_sdpa_check_matches_the_real_dispatcher():
    from transformers import GemmaConfig
    device = torch.device("cuda", torch.cuda.current_device())
    embeds = torch.zeros(1, 2, 8, device = device, dtype = torch.bfloat16)
    for head_dim, dropout in ((128, 0.0), (256, 0.1), (200, 0.0)):
        config = GemmaConfig(hidden_size = 2048, num_attention_heads = 8, num_key_value_heads = 1,
                             head_dim = head_dim, attention_dropout = dropout)
        config._attn_implementation = "sdpa"
        query = torch.randn(1, 8, 64, head_dim, device = device, dtype = torch.bfloat16, requires_grad = True)
        key = torch.randn(1, 1, 64, head_dim, device = device, dtype = torch.bfloat16, requires_grad = True)
        params = torch.backends.cuda.SDPAParams(query, key, key, None, dropout, True, True)
        expected = torch.backends.cuda.flash_sdp_enabled() and torch.cuda.get_device_capability(device) >= (8, 0) \
            and torch.backends.cuda.can_use_flash_attention(params, False)
        assert misc._flash_sdpa_takes_causal(config, embeds) is bool(expected)


class _FakeProbe:
    """Stands in for torch's CUDA probe on a CPU host: records the current device per flash check."""
    def __init__(self, monkeypatch, gqa_argument = True, accepts_on = ("cuda:1",)):
        import contextlib
        self.current = "cuda:0"
        self.checked_on = []
        self.param_counts = []
        real_empty = torch.empty
        monkeypatch.setattr(misc, "_FLASH_SDPA_ACCEPTS_CACHE", {})
        monkeypatch.setattr(torch, "empty", lambda shape, device = None, **kwargs: real_empty(shape, **kwargs))
        @contextlib.contextmanager
        def device(target):
            previous, self.current = self.current, str(target)
            try:
                yield
            finally:
                self.current = previous
        monkeypatch.setattr(torch.cuda, "device", device)
        def params(*args):
            if len(args) == 7 and not gqa_argument:
                raise TypeError("SDPAParams takes 6 arguments")
            self.param_counts.append(len(args))
            return args
        monkeypatch.setattr(torch.backends.cuda, "SDPAParams", params)
        def can_use(params, debug = False):
            self.checked_on.append(self.current)
            return self.current in accepts_on
        monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", can_use)


def test_flash_probe_runs_on_the_embeds_device(monkeypatch):
    # torch reads the current device's arch: with device_map across mixed GPUs a cuda:1 probe must not ask cuda:0.
    probe = _FakeProbe(monkeypatch, accepts_on = ("cuda:1",))
    assert misc._flash_sdpa_accepts("cuda:1", torch.bfloat16, 8, 1, 256, 0.1, True) is True
    assert misc._flash_sdpa_accepts("cuda:0", torch.bfloat16, 8, 1, 256, 0.1, True) is False
    assert probe.checked_on == ["cuda:1", "cuda:0"]
    assert probe.current == "cuda:0"


def test_flash_probe_without_enable_gqa_argument(monkeypatch):
    # torch 2.4 SDPAParams has no enable_gqa: MHA is still probed, GQA keeps the mask.
    probe = _FakeProbe(monkeypatch, gqa_argument = False, accepts_on = ("cuda:0",))
    assert misc._flash_sdpa_accepts("cuda:0", torch.bfloat16, 8, 8, 128, 0.0, True) is True
    assert misc._flash_sdpa_accepts("cuda:0", torch.bfloat16, 8, 1, 128, 0.0, True) is False
    assert probe.param_counts == [6]


class _SplitModel(torch.nn.Module):
    """Stands in for a decoder whose forward builds the mask; `devices` fakes where its layers live."""

    def __init__(self, devices = None, hf_device_map = None):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(2))
        self._devices = devices
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map

    def parameters(self, recurse = True):
        if self._devices is None:
            yield from super().parameters(recurse)
            return
        for device in self._devices:
            yield torch.empty(0, device = device)

    def forward(self, create_causal_mask, kwargs):
        return create_causal_mask(**kwargs)


@pytest.mark.parametrize("model", [
    lambda: _SplitModel(devices = ["cpu", "meta"]),
    lambda: _SplitModel(hf_device_map = {"model.embed_tokens": 0, "model.layers.1": 1}),
], ids = ["layer_devices", "hf_device_map"])
def test_right_padding_keeps_the_mask_on_a_split_model_by_default(patched, monkeypatch, model):
    monkeypatch.setattr(misc, "_flash_sdpa_takes_causal", lambda config, embeds: True)
    kwargs = _kwargs(patched, batch = 3, length = 8, attention_mask = _right_padded())
    model = model()
    before = misc.CAUSAL_MASK_SKIP_STATS["right_padded"]
    assert isinstance(model(patched.create_causal_mask, kwargs), torch.Tensor)
    assert misc.CAUSAL_MASK_SKIP_STATS["right_padded"] == before
    # The explicit opt-in keeps skipping on split models.
    monkeypatch.setenv("UNSLOTH_SKIP_RIGHT_PADDING_MASK", "1")
    assert model(patched.create_causal_mask, kwargs) is None
    assert misc.CAUSAL_MASK_SKIP_STATS["right_padded"] == before + 1


@pytest.mark.parametrize("hf_device_map", [None, {"": 0}, {"model.embed_tokens": 0, "lm_head": "cuda:0"}])
def test_right_padding_skips_on_a_single_device_model(patched, monkeypatch, hf_device_map):
    monkeypatch.setattr(misc, "_flash_sdpa_takes_causal", lambda config, embeds: True)
    kwargs = _kwargs(patched, batch = 3, length = 8, attention_mask = _right_padded())
    model = _SplitModel(hf_device_map = hf_device_map)
    before = misc.CAUSAL_MASK_SKIP_STATS["right_padded"]
    assert model(patched.create_causal_mask, kwargs) is None
    assert model(patched.create_causal_mask, kwargs) is None
    assert misc.CAUSAL_MASK_SKIP_STATS["right_padded"] == before + 2
    assert misc._MODEL_SPANS_DEVICES[model] is False
