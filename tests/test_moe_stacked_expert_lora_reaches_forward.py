"""Expert LoRA must reach the forward for a stacked-expert MoE, whatever forward runs.

`_patched_param_wrapper_forward` does not let PEFT fold an expert LoRA into the stacked
expert weight. It stashes the factors on the experts module and expects the experts forward
to apply them as a separate grouped GEMM. Unsloth installs such a forward for the MoE
families it patches; for a stacked-expert family it does not patch, transformers' own
experts forward runs and the stash is written and deleted unread, so the adapter has no
effect on the output while still looking healthy in `named_parameters`.

These tests pin both halves: the stash-reading forwards keep the separated path bit for bit,
and a forward that ignores the stash gets handed back to PEFT so the LoRA is applied anyway.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth_zoo.temporary_patches import moe_utils as MU


# ----------------------------------------------------------------------------- fixtures


def _moe_utils_copies():
    """Every loaded copy of moe_utils that can hold the saved PEFT forward.

    `install_to_cache` copies this module into unsloth_compiled_cache and
    `patch_param_wrapper_for_moe` prefers that copy, so the function PEFT ends up with, and
    the saved original, usually belong to `unsloth_cached_moe_utils` and not to the package
    module a test imported.
    """
    copies = [MU]
    for name in ("unsloth_cached_moe_utils", "moe_utils"):
        module = sys.modules.get(name)
        if module is not None and module is not MU and hasattr(module, "patch_param_wrapper_for_moe"):
            copies.append(module)
    return copies


@pytest.fixture
def restore_param_wrapper():
    """Start every test from PEFT's own `ParamWrapper.forward`, and put back what was there.

    Importing `unsloth_zoo.temporary_patches` applies the MoE patches as a side effect, so
    the wrapper may already be patched by the time a test runs. Without this reset the
    "before" half of a before-and-after comparison would be measured through the patch too,
    and the comparison would pass by being vacuous.
    """
    peft_layer = pytest.importorskip("peft.tuners.lora.layer")
    ParamWrapper = getattr(peft_layer, "ParamWrapper", None)
    if ParamWrapper is None:
        pytest.skip("this peft has no ParamWrapper (no target_parameters support)")

    installed = ParamWrapper.forward
    copies = _moe_utils_copies()
    saved = [(module, module._original_param_wrapper_forward) for module in copies]
    pristine = next((original for _, original in saved if original is not None), installed)

    ParamWrapper.forward = pristine
    for module, _ in saved:
        module._original_param_wrapper_forward = None
    try:
        yield ParamWrapper
    finally:
        ParamWrapper.forward = installed
        for module, original in saved:
            module._original_param_wrapper_forward = original


def _assert_wrapper_is_unpatched(ParamWrapper):
    """Guard against measuring the "before" half through the patch."""
    assert ParamWrapper.forward.__qualname__ != "_patched_param_wrapper_forward"


class _StashIgnoringExperts(nn.Module):
    """Stacked experts whose forward never looks at `_unsloth_lora_*`.

    This is what transformers' own experts forward does for every MoE family Unsloth has no
    patch for, Olmoe among them.
    """

    def __init__(self, num_experts=4, hidden=8, intermediate=6):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden, 2 * intermediate))
        self.down_proj = nn.Parameter(torch.randn(num_experts, intermediate, hidden))
        self.act_fn = F.silu

    def forward(self, hidden_states):
        gate_up = torch.einsum("teh,ehi->tei", hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.einsum("tei,eih->teh", self.act_fn(gate) * up, self.down_proj)


class _StashReadingExperts(_StashIgnoringExperts):
    """Same arithmetic, but it reads the stash the way Unsloth's own forwards do."""

    def forward(self, hidden_states):
        MU.take_moe_lora_stash(self, "gate_up_proj")
        MU.take_moe_lora_stash(self, "down_proj")
        return super().forward(hidden_states)


class _TinyMoE(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, hidden_states):
        return self.experts(hidden_states)


def _attach_expert_lora(model, rank=2, seed=0):
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(
        r=rank, lora_alpha=2 * rank, target_modules=[],
        target_parameters=["experts.gate_up_proj", "experts.down_proj"],
    )
    model = peft.get_peft_model(model, config)
    # lora_B is zero-initialised, which would hide any difference between the paths.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
    return model


def _build(experts_cls, seed=0, rank=2):
    torch.manual_seed(seed)
    return _attach_expert_lora(_TinyMoE(experts_cls()), rank=rank, seed=seed)


def _inputs(tokens=3, num_experts=4, hidden=8, seed=1):
    return torch.randn(tokens, num_experts, hidden,
                       generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------------------------- stash bookkeeping


def test_take_moe_lora_stash_records_the_read_and_returns_the_value():
    experts = _StashIgnoringExperts()
    assert MU.take_moe_lora_stash(experts, "gate_up_proj") is None
    assert MU._moe_lora_stash_was_read(experts, "gate_up_proj")
    assert not MU._moe_lora_stash_was_read(experts, "down_proj")

    setattr(experts, MU.moe_lora_stash_name("down_proj"), ("a", "b", 1.0, 4))
    assert MU.take_moe_lora_stash(experts, "down_proj") == ("a", "b", 1.0, 4)
    assert MU._moe_lora_stash_was_read(experts, "down_proj")


def test_reset_clears_only_the_named_parameter():
    experts = _StashIgnoringExperts()
    MU.take_moe_lora_stash(experts, "gate_up_proj")
    MU.take_moe_lora_stash(experts, "down_proj")
    MU._reset_moe_lora_stash_read(experts, "gate_up_proj")
    assert not MU._moe_lora_stash_was_read(experts, "gate_up_proj")
    assert MU._moe_lora_stash_was_read(experts, "down_proj")


def test_verdict_is_cached_per_parameter_and_invalidated_by_a_new_forward():
    experts = _StashIgnoringExperts()
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is None

    MU._record_moe_lora_forward_verdict(experts, "gate_up_proj", False)
    MU._record_moe_lora_forward_verdict(experts, "down_proj", True)
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is False
    assert MU.moe_lora_forward_applies_stash(experts, "down_proj") is True

    # Re-patching the forward must retire the verdict rather than answer for a function
    # that is no longer the one that will run.
    experts.forward = lambda hidden_states: hidden_states
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is None


def test_finding_a_wrapper_counts_as_reading_the_stash():
    """The MXFP4 GPT-OSS forward pulls LoRA straight off the wrapper, never off the stash."""
    experts = _StashIgnoringExperts()
    assert MU._get_lora_wrapper_for_param(experts, "gate_up_proj") is None
    assert not MU._moe_lora_stash_was_read(experts, "gate_up_proj")

    wrapper = nn.Module()
    wrapper.lora_A = nn.ModuleDict()
    experts.gate_up_proj_lora_wrapper = wrapper
    assert MU._get_lora_wrapper_for_param(experts, "gate_up_proj") is wrapper
    assert MU._moe_lora_stash_was_read(experts, "gate_up_proj")


# ------------------------------------------------------------------------- the defect


def test_expert_lora_reaches_a_forward_that_ignores_the_stash(restore_param_wrapper):
    """The regression. On main this output has no expert LoRA in it at all."""
    _assert_wrapper_is_unpatched(restore_param_wrapper)
    reference = _build(_StashIgnoringExperts)
    x = _inputs()
    with torch.no_grad():
        expected = reference(x)

    assert MU.patch_param_wrapper_for_moe()
    patched = _build(_StashIgnoringExperts)
    with torch.no_grad():
        got = patched(x)

    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


def test_expert_lora_changes_the_output_of_a_stash_ignoring_forward(restore_param_wrapper):
    """Blunter form of the same thing: zeroing lora_B has to change the output."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    x = _inputs()
    with torch.no_grad():
        with_lora = model(x)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
        without_lora = model(x)
    assert (with_lora - without_lora).abs().max() > 1e-6


def test_expert_lora_gradients_flow_for_a_stash_ignoring_forward(restore_param_wrapper):
    """Inert LoRA still reports requires_grad; the giveaway is that no gradient arrives."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    model(_inputs()).square().sum().backward()
    lora_B = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert lora_B
    assert all(p.grad is not None and p.grad.abs().max() > 0 for p in lora_B)


def test_a_stash_reading_forward_keeps_the_separated_path(restore_param_wrapper):
    """No regression for the families Unsloth does patch: PEFT's forward is never called."""
    assert MU.patch_param_wrapper_for_moe()
    calls = []
    original = MU._original_param_wrapper_forward

    def counting_original(self, x, *args, **kwargs):
        calls.append(getattr(self, "parameter_name", None))
        return original(self, x, *args, **kwargs)

    MU._original_param_wrapper_forward = counting_original
    model = _build(_StashReadingExperts)
    with torch.no_grad():
        model(_inputs())

    assert calls == []
    experts = model.base_model.model.experts.get_base_layer()
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is True
    assert MU.moe_lora_forward_applies_stash(experts, "down_proj") is True


def test_a_stash_ignoring_forward_is_recorded_and_not_reprobed(restore_param_wrapper):
    """The probe costs one extra call, once, and the verdict pins it afterwards."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    experts = model.base_model.model.experts.get_base_layer()
    x = _inputs()

    counter = {"n": 0}
    inner = type(experts).forward

    def counting_forward(self, hidden_states):
        counter["n"] += 1
        return inner(self, hidden_states)

    experts.forward = counting_forward.__get__(experts, type(experts))
    with torch.no_grad():
        model(x)
    first = counter["n"]
    with torch.no_grad():
        model(x)
    second = counter["n"] - first

    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is False
    assert MU.moe_lora_forward_applies_stash(experts, "down_proj") is False
    assert second == 1, "the verdict should make every later call a single forward"
    assert first > second, "the first call should have probed"


# --------------------------------------------------------------------- real model path


def _tiny_olmoe(dtype=torch.float32):
    olmoe = pytest.importorskip("transformers.models.olmoe.modeling_olmoe")
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig

    config = OlmoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
        num_experts_per_tok=2, max_position_embeddings=64, norm_topk_prob=False,
    )
    torch.manual_seed(0)
    model = olmoe.OlmoeForCausalLM(config).to(dtype).eval()
    if not hasattr(model.model.layers[0].mlp.experts, "gate_up_proj"):
        pytest.skip("this transformers keeps Olmoe experts as an nn.ModuleList")
    return model


def test_olmoe_expert_lora_matches_unpatched_peft(restore_param_wrapper):
    """Olmoe is the family in the wild that has stacked experts and no Unsloth forward."""
    peft = pytest.importorskip("peft")
    config = peft.LoraConfig(
        r=3, lora_alpha=6, target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(3))
    _assert_wrapper_is_unpatched(restore_param_wrapper)

    def logits():
        model = peft.get_peft_model(_tiny_olmoe(), config)
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
            return model(input_ids=ids).logits.float()

    expected = logits()
    assert MU.patch_param_wrapper_for_moe()
    torch.testing.assert_close(logits(), expected, rtol=0, atol=1e-6)


# ----------------------------------------------------------------------- drift guards


_STASH_READ = re.compile(r"""getattr\(\s*self\s*,\s*["']_unsloth_lora_(gate_up_proj|down_proj)["']""")
_STASH_ATTR = re.compile(r"""self\._unsloth_lora_(gate_up_proj|down_proj)\b""")


@pytest.mark.parametrize("filename", ["moe_utils.py", "moe_utils_fp8.py", "moe_utils_bnb4bit.py"])
def test_every_stash_read_goes_through_the_helper(filename):
    """A raw read would not be recorded, so the wrapper would think the LoRA was applied."""
    path = Path(MU.__file__).with_name(filename)
    source = path.read_text(encoding="utf-8")
    # The writer side lives in `_patched_param_wrapper_forward` and uses the experts module,
    # not `self`, so only `self`-qualified reads are the forward-side ones.
    offenders = _STASH_READ.findall(source) + _STASH_ATTR.findall(source)
    assert offenders == [], (
        f"{filename} reads the expert LoRA stash directly for {sorted(set(offenders))}; "
        "use take_moe_lora_stash so the read is recorded"
    )


def test_fp8_last_resort_loop_still_refuses_an_unappliable_lora():
    """Recording the read there must not turn the refusal into a silent PEFT fallback."""
    fp8 = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_fp8")
    experts = _StashIgnoringExperts()
    setattr(experts, MU.moe_lora_stash_name("down_proj"), ("a", "b", 1.0, 4))
    with pytest.raises(RuntimeError, match="separated LoRA delta"):
        fp8._forward_native_fp8_expert_loop(experts, torch.zeros(1, 8), None, None)
    assert MU._moe_lora_stash_was_read(experts, "down_proj")


# ----------------------------------------------------------- the stubbed bitsandbytes


def test_has_bnb_is_false_when_params4bit_is_not_a_class():
    """`HAS_BNB` must mean "isinstance against Params4bit is legal", not "the import ran".

    Where the real bitsandbytes is absent, unsloth_zoo injects a permissive stub whose every
    attribute is a placeholder object. `from bitsandbytes.nn import Params4bit` then succeeds
    and returns a non-class, and `_is_moe_experts_module` and `_get_base_weight` both go
    straight into `isinstance(param, Params4bit)`, which raises
    `TypeError: isinstance() arg 2 must be a type`. That is every expert forward on a Mac.
    """
    if MU.HAS_BNB:
        assert isinstance(MU.Params4bit, type)
    else:
        assert MU.Params4bit is None


@pytest.mark.parametrize("module_name", [
    "moe_utils", "moe_utils_bnb4bit", "moe_bnb", "moe_grouped_modulelist",
])
def test_every_module_level_has_bnb_implies_a_real_class(module_name):
    """The same guard in every sibling that keeps its own HAS_BNB and Params4bit pair.

    Each of these does `isinstance(w, Params4bit)` gated only on its own HAS_BNB, so one of
    them left honest and the others not is a half fix.
    """
    import importlib
    module = importlib.import_module(f"unsloth_zoo.temporary_patches.{module_name}")
    if getattr(module, "HAS_BNB", False):
        assert isinstance(module.Params4bit, type), (
            f"{module_name}.HAS_BNB is True but Params4bit is "
            f"{type(module.Params4bit).__name__}, so isinstance against it raises TypeError"
        )
    else:
        assert getattr(module, "Params4bit", None) is None


def test_is_moe_experts_module_survives_a_stubbed_params4bit(monkeypatch):
    """The call site itself, driven with the stub shape, whatever this runner has installed."""
    class _Noop:
        def __getattr__(self, name):
            return self

    monkeypatch.setattr(MU, "Params4bit", _Noop(), raising=False)
    monkeypatch.setattr(MU, "HAS_BNB", False, raising=False)
    experts = _StashIgnoringExperts()
    assert MU._is_moe_experts_module(experts) is True
