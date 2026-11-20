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

"""DeepSeek-V4.1 training patches (temporary_patches/deepseek_v41.py), CPU only, against every implementation on
the path: the community transformers port and Unsloth's native module (tests/_dsv41_impl.py)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsv41_impl import available  # noqa: E402

IMPLS = available()
pytestmark = pytest.mark.skipif(
    not IMPLS, reason = "needs a DeepSeek-V4.1 implementation (transformers port, or unsloth's native module)"
)


@pytest.fixture(params = IMPLS or [None], ids = lambda impl: getattr(impl, "name", "none"))
def impl(request):
    request.param.register()
    return request.param


def _patches():
    from unsloth_zoo.temporary_patches.deepseek_v41 import (
        patch_deepseek_v41_fake_quant_ste,
        patch_deepseek_v41_gradient_checkpointing,
        patch_deepseek_v41_grouped_lora,
    )
    patch_deepseek_v41_gradient_checkpointing()
    patch_deepseek_v41_fake_quant_ste()
    patch_deepseek_v41_grouped_lora()


def _tiny_model(impl, seed = 0):
    DeepseekV41TextConfig = impl.TextConfig
    DeepseekV41ForCausalLM = impl.modeling.DeepseekV41ForCausalLM

    # Covers the real schedule kinds (sliding-only, ratio-1 and ratio-2 KV and index sources).
    config = DeepseekV41TextConfig(
        vocab_size = 256, hidden_size = 64, num_hidden_layers = 6, num_attention_heads = 2, head_dim = 64,
        qk_rope_head_dim = 16, q_lora_rank = 32, o_groups = 2, o_lora_rank = 16, moe_intermediate_size = 32,
        n_routed_experts = 4, num_experts_per_tok = 2, sliding_window = 8,
        compress_ratios = [0, 0, 2, 2, 1, 1], kv_source_layer_ids = [2, 4], index_source_layer_ids = [2, 3, 4, 5],
        candidate_source_layer_id = 4, candidate_topk_blocks = 2, candidate_block_size = 2, index_n_heads = 2,
        index_head_dim = 32, index_topk = 4, hc_mult = 2, hc_sinkhorn_iters = 3, engram_layer_ids = [],
        num_nextn_predict_layers = 0, rms_norm_eps = 1e-6,
    )
    torch.manual_seed(seed)
    model = DeepseekV41ForCausalLM(config).float()
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.normal_(0.0, 0.05)
    return model


def _grads(model, ids, gradient_checkpointing):
    model.zero_grad(set_to_none = True)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    out = model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_fake_quant_straight_through(impl):
    if impl.name != "port":
        pytest.skip("the native module has no fake-quant sites (it trains dequantized or bnb weights)")
    _patches()
    modeling = impl.modeling

    torch.manual_seed(0)
    x = torch.randn(3, 64, dtype = torch.bfloat16)
    for fn, args in ((modeling._fake_quant_fp8_block, (32,)), (modeling._fake_quant_fp4_block, (16, True))):
        reference = fn._unsloth_original(x, *args)
        leaf = x.clone().requires_grad_(True)
        out = fn(leaf, *args)
        assert torch.equal(out, reference)  # forward values unchanged
        (out.float() * 1e-4).sum().backward()
        # Before: the fp8 cast rounded the gradient to e4m3 (all zeros at this scale); fp4 only through the amax.
        assert torch.equal(leaf.grad, torch.full_like(leaf.grad, 1e-4))


def test_gradient_checkpointing_replays_group_state(impl):
    _patches()
    modeling = impl.modeling

    assert modeling.DeepseekV41PreTrainedModel.supports_gradient_checkpointing
    model = _tiny_model(impl)
    ids = torch.randint(0, 256, (2, 24), generator = torch.Generator().manual_seed(0))
    loss0, g0 = _grads(model, ids, False)
    loss1, g1 = _grads(model, ids, True)
    torch.testing.assert_close(loss0, loss1)
    assert g0.keys() == g1.keys()
    for name in g0:
        torch.testing.assert_close(g0[name], g1[name], rtol = 1e-4, atol = 1e-6, msg = name)
    # Every non-indexer weight learns from the LM loss (window KV included).
    zero = [n for n, g in g0.items() if g.abs().sum() == 0 and ".indexer." not in n]
    assert zero == []


def test_checkpointed_layers_keep_the_compile_stance(impl, monkeypatch):
    # Layers keep the caller's stance, not force_eager: a helper also used outside must compile
    # alike.
    _patches()
    from torch._dynamo import eval_frame
    modeling = impl.modeling

    seen = []
    original = modeling.DeepseekV41RMSNorm.forward

    def forward(self, *args, **kwargs):
        seen.append(getattr(eval_frame._stance, "stance", "default"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(modeling.DeepseekV41RMSNorm, "forward", forward)
    model = _tiny_model(impl)
    ids = torch.randint(0, 256, (2, 24), generator = torch.Generator().manual_seed(0))
    _grads(model, ids, True)
    assert seen and set(seen) == {"default"}


def test_prepare_compiled_checkpoint_specializes_floats(monkeypatch):
    from unsloth_zoo.temporary_patches.deepseek_v41 import prepare_compiled_checkpoint

    # Not monkeypatch: its undo deletes the ConfigModule attribute and breaks later torch.compile calls.
    saved = torch._dynamo.config.specialize_float
    try:
        torch._dynamo.config.specialize_float = False
        assert prepare_compiled_checkpoint() is True
        assert torch._dynamo.config.specialize_float is True
    finally:
        torch._dynamo.config.specialize_float = saved


def test_grouped_lora_matches_merged_weight(impl):
    _patches()
    from peft import LoraConfig, get_peft_model

    model = _tiny_model(impl)
    ids = torch.randint(0, 256, (2, 16), generator = torch.Generator().manual_seed(1))
    peft_model = get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["o_a_proj", "o_b_proj"], init_lora_weights = False)
    )
    wrapped = [m for n, m in peft_model.named_modules() if n.endswith("self_attn.o_a_proj")]
    assert wrapped and all(type(m).__name__ == "DeepseekV41GroupedLoraLinear" for m in wrapped)
    peft_model.eval()
    with torch.no_grad():
        adapted = peft_model(input_ids = ids, use_cache = False).logits
        merged = peft_model.merge_and_unload()
        reference = merged(input_ids = ids, use_cache = False).logits
    torch.testing.assert_close(adapted, reference, rtol = 1e-4, atol = 1e-4)


def test_patches_are_noops_without_port(monkeypatch):
    import unsloth_zoo.temporary_patches.deepseek_v41 as patch_module

    monkeypatch.setattr(patch_module, "_modeling", lambda: None)
    patch_module.patch_deepseek_v41_gradient_checkpointing()
    patch_module.patch_deepseek_v41_fake_quant_ste()
    patch_module.patch_deepseek_v41_grouped_lora()


def test_auto_targets_cover_engram_and_skip_indexer(impl):
    import re

    DeepseekV41TextConfig = impl.TextConfig
    DeepseekV41ForCausalLM = impl.modeling.DeepseekV41ForCausalLM

    from unsloth_zoo.peft_utils import get_peft_regex

    config = DeepseekV41TextConfig(
        vocab_size = 256, hidden_size = 64, num_hidden_layers = 4, num_attention_heads = 2, head_dim = 64,
        qk_rope_head_dim = 16, q_lora_rank = 32, o_groups = 2, o_lora_rank = 16, moe_intermediate_size = 32,
        n_routed_experts = 4, num_experts_per_tok = 2, compress_ratios = [0, 0, 2, 1],
        kv_source_layer_ids = [2, 3], index_source_layer_ids = [2, 3], candidate_source_layer_id = 3,
        index_n_heads = 2, index_head_dim = 32, hc_mult = 2, engram_layer_ids = [1],
        engram_num_embeddings = [64], engram_head_dim = 32, engram_vocab_size = 3, engram_n_heads = 2,
        engram_compressed_vocab_size = 8, num_nextn_predict_layers = 0,
    )
    model = DeepseekV41ForCausalLM(config)
    regex = get_peft_regex(model, True, True, True, True)
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    hit = {n for n in linears if re.fullmatch(regex, n)}
    assert "model.layers.1.engram.wkv" in hit
    assert not [n for n in hit if ".indexer." in n]
    expected = {
        "q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj", "gate_proj", "up_proj", "down_proj", "wkv",
    }
    assert {n.rsplit(".", 1)[-1] for n in hit} == expected
    assert "lm_head" not in hit
    # An explicit list still reaches the indexer.
    explicit = get_peft_regex(model, True, True, True, True, target_modules = ["q_b_proj"])
    assert any(re.fullmatch(explicit, n) for n in linears if ".indexer." in n)


def test_num_experts_alias_and_cpu_runtime_loads_the_patch(impl):
    from unsloth_zoo.temporary_patches.deepseek_v41 import TEMPORARY_PATCHES, patch_deepseek_v41_config_aliases

    DeepseekV41TextConfig = impl.TextConfig

    patch_deepseek_v41_config_aliases()
    config = DeepseekV41TextConfig(n_routed_experts = 7, num_hidden_layers = 4, num_nextn_predict_layers = 0)
    assert config.num_experts == 7 == config.num_local_experts

    # The CPU runtime loads the patch file standalone (no GPU kernels), so its imports must stay self-contained.
    import importlib.util
    import os
    import sys
    import types

    import unsloth_zoo

    path = os.path.join(os.path.dirname(unsloth_zoo.__file__), "temporary_patches", "deepseek_v41.py")
    common = types.ModuleType("_dsv41_standalone.common")
    common.TEMPORARY_PATCHES = []
    package = types.ModuleType("_dsv41_standalone")
    package.__path__ = []
    sys.modules["_dsv41_standalone"], sys.modules["_dsv41_standalone.common"] = package, common
    try:
        spec = importlib.util.spec_from_file_location("_dsv41_standalone.deepseek_v41", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("_dsv41_standalone", None)
        sys.modules.pop("_dsv41_standalone.common", None)
    registered = [f.__name__ for f in TEMPORARY_PATCHES if f.__module__.endswith("temporary_patches.deepseek_v41")]
    assert [f.__name__ for f in common.TEMPORARY_PATCHES] == registered and len(registered) == 6


def test_grouped_lora_defers_to_a_registered_custom_module(impl):
    _patches()
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora.layer import Linear as LoraLinear

    DeepseekV41GroupedLinear = impl.modeling.DeepseekV41GroupedLinear

    class CallerGroupedLoRA(LoraLinear):
        pass

    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["o_a_proj"])
    config._register_custom_module({DeepseekV41GroupedLinear: CallerGroupedLoRA})
    peft_model = get_peft_model(_tiny_model(impl), config)
    wrapped = [m for n, m in peft_model.named_modules() if n.endswith("self_attn.o_a_proj")]
    assert wrapped and all(type(m) is CallerGroupedLoRA for m in wrapped)


def test_no_placement_bare_names_are_relative_to_no_split_modules(impl):
    DeepseekV41TextConfig = impl.TextConfig
    DeepseekV41ForCausalLM = impl.modeling.DeepseekV41ForCausalLM

    from unsloth_zoo.device_map_planner import drop_no_placement_modules, no_placement_matcher

    config = DeepseekV41TextConfig(
        vocab_size = 256, hidden_size = 64, num_hidden_layers = 4, num_attention_heads = 2, head_dim = 64,
        qk_rope_head_dim = 16, q_lora_rank = 32, o_groups = 2, o_lora_rank = 16, moe_intermediate_size = 32,
        n_routed_experts = 4, num_experts_per_tok = 2, compress_ratios = [0, 0, 2, 1],
        kv_source_layer_ids = [2, 3], index_source_layer_ids = [2, 3], candidate_source_layer_id = 3,
        index_n_heads = 2, index_head_dim = 32, hc_mult = 2, engram_layer_ids = [1],
        engram_num_embeddings = [64], engram_head_dim = 32, engram_vocab_size = 3, engram_n_heads = 2,
        engram_compressed_vocab_size = 8, num_nextn_predict_layers = 0,
    )
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config)
    assert model._no_placement_params == ["weight", "weight_scale_inv"]
    match = no_placement_matcher(model)
    hit = sorted(n for n, _ in model.named_parameters() if match(n))
    if impl.name == "native" and not any(True for _ in model.model.engram_tables["1"].parameters()):
        assert hit == []
    else:
        assert hit == ["model.engram_tables.1.weight", "model.engram_tables.1.weight_scale_inv"]
        assert drop_no_placement_modules(model) == ["model.engram_tables.1"]
    # Dotted names (Qwen4Exp) keep the suffix match.
    dotted = no_placement_matcher(model, ["engram_tables.1.weight"])
    assert dotted("model.engram_tables.1.weight") and not dotted("lm_head.weight")


def test_pooled_compressor_lora_is_applied_and_trains(impl):
    _patches()
    from peft import LoraConfig, get_peft_model

    model = _tiny_model(impl)
    ids = torch.randint(0, 256, (2, 24), generator = torch.Generator().manual_seed(2))
    peft_model = get_peft_model(
        model,
        LoraConfig(r = 4, lora_alpha = 8, target_modules = ["compressor.kv_proj", "compressor.gate_proj"], init_lora_weights = False),
    )
    peft_model.train()
    out = peft_model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    # Layer 2 is the ratio-2 KV source (pooled, fp32 branch that reads `.weight`); before, its adapters got no grad.
    from unsloth_zoo.temporary_patches.deepseek_v41 import REPLAY_STATS

    if impl.name == "port":
        assert REPLAY_STATS["compressor_lora"] > 0
    for name in ("kv_proj", "gate_proj"):
        grads = [p.grad for n, p in peft_model.named_parameters() if f"layers.2.self_attn.compressor.{name}.lora_" in n]
        assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), name
    peft_model.eval()
    with torch.no_grad():
        adapted = peft_model(input_ids = ids, use_cache = False).logits
        reference = peft_model.merge_and_unload()(input_ids = ids, use_cache = False).logits
    torch.testing.assert_close(adapted, reference, rtol = 1e-4, atol = 1e-4)


def test_gradient_checkpointing_keeps_shared_state_grads_under_a_reentrant_shim(impl, monkeypatch):
    """Unsloth's smart checkpointer replaces torch.utils.checkpoint.checkpoint with a reentrant one: the patched layer
    still checkpoints non-reentrantly, so the layers reading another layer's shared KV still send it gradient."""
    _patches()
    import torch.utils.checkpoint as ckpt

    pristine = ckpt.checkpoint
    monkeypatch.setattr(ckpt, "_unsloth_pristine_checkpoint", pristine, raising = False)
    monkeypatch.setattr(ckpt, "checkpoint", lambda fn, *a, use_reentrant = None, **k: pristine(fn, *a, use_reentrant = True, **k))
    model = _tiny_model(impl)
    ids = torch.randint(0, 256, (2, 24), generator = torch.Generator().manual_seed(0))
    loss0, g0 = _grads(model, ids, False)
    model.zero_grad(set_to_none = True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model.train()
    out = model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    g1 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    torch.testing.assert_close(loss0, out.loss.detach())
    assert g0.keys() == g1.keys()
    for name in g0:
        torch.testing.assert_close(g0[name], g1[name], rtol = 1e-4, atol = 1e-6, msg = name)


def test_gradient_checkpointing_uses_the_bound_non_reentrant_wrapper(impl, monkeypatch):
    """Unsloth binds a non-reentrant wrapper (it offloads the saved inputs for use_gradient_checkpointing="unsloth")
    as each layer's _gradient_checkpointing_func; the replaying layer must checkpoint through it, not around it."""
    _patches()
    import functools
    import torch.utils.checkpoint as ckpt

    pristine = ckpt.checkpoint
    calls = []

    def wrapper(function, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        kwargs["use_reentrant"] = False
        return pristine(function, *args, **kwargs)

    wrapper._unsloth_original = pristine
    wrapper._unsloth_offloads = True
    monkeypatch.setattr(ckpt, "_unsloth_pristine_checkpoint", pristine, raising = False)
    model = _tiny_model(impl)
    ids = torch.randint(0, 256, (2, 24), generator = torch.Generator().manual_seed(0))
    loss0, g0 = _grads(model, ids, False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    for module in model.modules():
        if getattr(module, "_gradient_checkpointing_func", None) is not None:
            module._gradient_checkpointing_func = functools.partial(wrapper, use_reentrant = False)
    model.zero_grad(set_to_none = True)
    model.train()
    out = model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    assert calls == [False] * model.config.num_hidden_layers
    g1 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    torch.testing.assert_close(loss0, out.loss.detach())
    assert g0.keys() == g1.keys()
    for name in g0:
        torch.testing.assert_close(g0[name], g1[name], rtol = 1e-4, atol = 1e-6, msg = name)


@pytest.mark.parametrize("offload", [True, False], ids = ["unsloth", "true"])
def test_unsloth_gc_offloads_after_the_smart_patch_captured_the_pristine_checkpoint(impl, monkeypatch, offload):
    """use_gradient_checkpointing="unsloth" at max_seq_length >= 512: Unsloth patches its smart checkpointing first,
    which captures torch's pristine checkpoint, then forces DeepSeek-V4.1 onto its non-reentrant wrapper (the only
    entry to the activation-offload hooks). The replaying layer must still offload ("unsloth"), and not for True."""
    _patches()
    import functools
    import torch.utils.checkpoint as ckpt
    from unsloth_zoo import gradient_checkpointing as gc_module

    monkeypatch.setattr(ckpt, "_unsloth_pristine_checkpoint", None, raising = False)
    gc_module._capture_pristine_checkpoint_once()  # what patch_unsloth_smart_gradient_checkpointing does first
    assert ckpt._unsloth_pristine_checkpoint is not None
    pristine = ckpt.checkpoint

    def nonre_checkpoint(function, *args, **kwargs):  # unsloth/models/vision.py's _nonre_checkpoint
        kwargs["use_reentrant"] = False
        if not offload:
            return pristine(function, *args, **kwargs)
        with gc_module.non_reentrant_offload_hooks():
            return pristine(function, *args, **kwargs)

    nonre_checkpoint._unsloth_original = pristine
    nonre_checkpoint._unsloth_offloads = offload
    monkeypatch.setattr(ckpt, "checkpoint", nonre_checkpoint)
    # CPU stands in for the device: count CPU activations as offloadable.
    monkeypatch.setattr(gc_module.NonReentrantActivationOffloader, "_on_host", lambda self, x: x.numel() == 0)
    model = _tiny_model(impl)
    ids = torch.randint(0, 256, (2, 24), generator = torch.Generator().manual_seed(0))
    loss0, g0 = _grads(model, ids, False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    for module in model.modules():
        if getattr(module, "_gradient_checkpointing_func", None) is not None:
            module._gradient_checkpointing_func = functools.partial(nonre_checkpoint, use_reentrant = False)
    gc_module.get_activation_offload_stats(reset = True)
    model.zero_grad(set_to_none = True)
    model.train()
    out = model(input_ids = ids, labels = ids, use_cache = False)
    out.loss.backward()
    stats = gc_module.get_activation_offload_stats(reset = True)
    assert (stats["offloaded"] > 0) == offload, stats
    assert stats["restored"] == stats["offloaded"], stats
    g1 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    torch.testing.assert_close(loss0, out.loss.detach())
    assert g0.keys() == g1.keys()
    for name in g0:
        torch.testing.assert_close(g0[name], g1[name], rtol = 1e-4, atol = 1e-6, msg = name)
