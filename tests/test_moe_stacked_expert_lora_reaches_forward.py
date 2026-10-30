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

"""Expert LoRA must reach stacked-expert MoE forwards; stash readers keep the separated path, while forwards that never
read the stash are handed back to PEFT, else the adapter has no effect."""

from __future__ import annotations

import contextlib
import json
import textwrap
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
    """Every loaded moe_utils copy that can hold the saved PEFT forward; patch_param_wrapper_for_moe prefers the
    unsloth_cached_moe_utils copy from install_to_cache over the imported module."""
    copies = [MU]
    for name in ("unsloth_cached_moe_utils", "moe_utils"):
        module = sys.modules.get(name)
        if module is not None and module is not MU and hasattr(module, "patch_param_wrapper_for_moe"):
            copies.append(module)
    return copies


@pytest.fixture
def restore_param_wrapper():
    """Starts each test from PEFT's own ParamWrapper.forward and restores it after; importing
    unsloth_zoo.temporary_patches already patches it, else before/after checks pass vacuously."""
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
    """Stacked experts whose forward never reads the _unsloth_lora_* stash, like transformers' own experts forward for MoE families Unsloth does not patch (Olmoe)."""

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

    # Re-patching the forward must retire the recorded verdict, not answer for a function that is no longer the one that will run.
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
    # The writer lives in _patched_param_wrapper_forward and uses the experts module, not self, so only self-qualified reads count as forward-side reads.
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
    """HAS_BNB must mean Params4bit is a real class: without bitsandbytes, the stub makes isinstance(param, Params4bit) raise TypeError on every Mac expert forward."""
    if MU.HAS_BNB:
        assert isinstance(MU.Params4bit, type)
    else:
        assert MU.Params4bit is None


@pytest.mark.parametrize("module_name", [
    "moe_utils", "moe_utils_bnb4bit", "moe_grouped_modulelist",
])
def test_every_module_level_has_bnb_implies_a_real_class(module_name):
    """Every sibling module with its own HAS_BNB and Params4bit pair needs the same real-class guard; one left unguarded is a half fix."""
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


# ------------------------------------------- what the measuring call is allowed to cost


def test_non_reentrant_gradient_checkpointing_survives_the_measurement(restore_param_wrapper):
    """The measuring call must not save tensors later calls do not: non-reentrant checkpoint (transformers 5 default) raises CheckpointError on a count mismatch."""
    from torch.utils.checkpoint import checkpoint

    assert MU.patch_param_wrapper_for_moe()
    for experts_cls in (_StashIgnoringExperts, _StashReadingExperts):
        model = _build(experts_cls)
        x = _inputs().requires_grad_(True)
        # The assertion is that this does not raise CheckpointError.
        out = checkpoint(lambda t: model(t), x, use_reentrant=False)
        out.sum().backward()
        if experts_cls is _StashIgnoringExperts:
            # This half is rerouted to PEFT, so the adapter is in the graph and trains.
            grads = [p.grad for name, p in model.named_parameters() if "lora_B" in name]
            assert grads and all(g is not None and g.abs().sum() > 0 for g in grads), (
                f"{experts_cls.__name__}: expert LoRA received no gradient"
            )


def test_the_measurement_happens_once_and_costs_one_forward_afterwards(restore_param_wrapper):
    """The stash measurement is one-off; after it each call runs one experts forward, else stash-ignoring families pay a
    double call every step and stash readers stop matching main byte for byte."""
    assert MU.patch_param_wrapper_for_moe()
    x = _inputs()
    for experts_cls in (_StashIgnoringExperts, _StashReadingExperts):
        model = _build(experts_cls)
        experts = model.base_model.model.experts
        while not isinstance(experts, experts_cls):
            experts = experts.get_base_layer()
        calls = {"n": 0}
        underlying = type(experts).forward

        def counting(self, hidden_states, _underlying=underlying):
            calls["n"] += 1
            return _underlying(self, hidden_states)

        type(experts).forward = counting
        try:
            with torch.no_grad():
                model(x)
                calls["n"] = 0
                first = model(x)
                assert calls["n"] == 1, (
                    f"{experts_cls.__name__}: {calls['n']} experts forwards per steady call"
                )
                calls["n"] = 0
                second = model(x)
                assert calls["n"] == 1
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        finally:
            type(experts).forward = underlying


def test_a_quantized_expert_weight_is_never_handed_to_the_peft_fold(monkeypatch):
    """PEFT folds by adding a delta, which needs a real float, so Params4bit, MXFP4 and FP8 expert weights must decline
    the reroute; MXFP4 GPT-OSS gets False because _get_lora_wrapper_for_param resolves to None under
    target_parameters."""
    # Set through monkeypatch: a saved forward left behind would be called by every later test, and a None stand-in would look like a model producing no output.
    monkeypatch.setattr(
        MU,
        "_original_param_wrapper_forward",
        MU._original_param_wrapper_forward or (lambda *a: None),
    )

    experts = _StashIgnoringExperts()
    assert MU._can_fold_moe_lora_through_peft(experts, "gate_up_proj") is True

    experts.gate_up_proj = nn.Parameter(
        torch.zeros(4, 8, 12, dtype=torch.uint8), requires_grad=False
    )
    assert MU._can_fold_moe_lora_through_peft(experts, "gate_up_proj") is False

    float8 = getattr(torch, "float8_e4m3fn", None)
    if float8 is not None:
        experts.gate_up_proj = nn.Parameter(
            torch.zeros(4, 8, 12).to(float8), requires_grad=False
        )
        assert MU._can_fold_moe_lora_through_peft(experts, "gate_up_proj") is False

    # bitsandbytes marks a packed 4-bit weight with quant_state, not with a dtype.
    packed = nn.Parameter(torch.zeros(4, 8, 12), requires_grad=False)
    packed.quant_state = object()
    experts.gate_up_proj = packed
    assert MU._can_fold_moe_lora_through_peft(experts, "gate_up_proj") is False


def test_the_probe_retains_no_autograd_graph_for_the_second_forward():
    """The probe runs under torch.no_grad() with its return value unbound, so no autograd graph outlives it; an in-band
    probe kept a graph alive and doubled peak memory on the first step."""
    import ast
    import inspect

    source = textwrap.dedent(inspect.getsource(MU._measure_moe_lora_stash_read))
    tree = ast.parse(source)

    no_grad_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "attr", None) == "no_grad"
            for item in node.items
        )
    ]
    assert no_grad_calls, "the probe forward is no longer under torch.no_grad()"

    probe_calls = [
        node
        for with_node in no_grad_calls
        for node in ast.walk(with_node)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "base_layer"
    ]
    assert probe_calls, "the probe no longer calls wrapper.base_layer under no_grad"

    bound = [
        node
        for with_node in no_grad_calls
        for node in ast.walk(with_node)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
    ]
    assert not bound, (
        "the probe forward's output is bound to a name inside the no_grad block; it must "
        f"stay unbound so nothing survives into the fallback call: {ast.dump(bound[0])}"
    )


def test_the_probe_result_never_reaches_the_peft_fallback():
    """When PEFT's forward runs, no local of the patched forward may hold a tensor that carries a grad_fn."""
    import inspect

    assert MU.patch_param_wrapper_for_moe()
    active = MU._load_cached_moe_utils_module() or MU
    original = active._original_param_wrapper_forward
    assert original is not None, "nothing recorded PEFT's original forward"
    retained = []

    def watching_original(self, x, *args, **kwargs):
        frame = inspect.currentframe().f_back
        if frame.f_code.co_name == "_patched_param_wrapper_forward":
            retained.append(
                sorted(
                    name
                    for name, value in frame.f_locals.items()
                    if isinstance(value, torch.Tensor) and value.grad_fn is not None
                )
            )
        return original(self, x, *args, **kwargs)

    active._original_param_wrapper_forward = watching_original
    try:
        model = _build(_StashIgnoringExperts)
        model(_inputs()).sum().backward()
    finally:
        active._original_param_wrapper_forward = original

    assert retained, "PEFT's forward was never reached, so this asserts nothing"
    for names in retained:
        assert names == [], (
            f"the patched forward still holds graph-carrying tensors while PEFT's forward "
            f"runs, so both forwards' activations are resident at once: {names}"
        )


# --------------------------------------------------------- the probe is RNG-invisible


class _RNGConsumingExperts(nn.Module):
    """An experts forward with dropout in it, which is all the probe needs to disturb."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *args, **kwargs):
        self.calls += 1
        return F.dropout(x, p = 0.5, training = True)


class _ProbeWrapper(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer


def _run_probe(monkeypatch, experts):
    """Runs _measure_moe_lora_stash_read with LoRA extraction, stash reset, read detection and verdict recording stubbed, so only the probe forward is observed."""
    monkeypatch.setattr(MU, "_extract_lora_from_wrapper", lambda wrapper: None)
    monkeypatch.setattr(MU, "_reset_moe_lora_stash_read", lambda module, name: None)
    monkeypatch.setattr(MU, "_moe_lora_stash_was_read", lambda module, name: True)
    monkeypatch.setattr(MU, "_record_moe_lora_forward_verdict", lambda module, name, read: None)
    return MU._measure_moe_lora_stash_read(
        _ProbeWrapper(experts), experts, "gate_up_proj", torch.ones(4, 8), (), {}
    )


def test_the_probe_forward_leaves_the_rng_where_it_found_it(monkeypatch):
    """no_grad turns off the graph, not RNG: the probe forks RNG, else its draws shift the counted forward and checkpoint recompute gets different dropout masks."""
    experts = _RNGConsumingExperts()

    torch.manual_seed(1234)
    expected = torch.rand(6)

    torch.manual_seed(1234)
    assert _run_probe(monkeypatch, experts) is True
    assert experts.calls == 1, "the probe did not run the forward it is supposed to measure"
    after_probe = torch.rand(6)

    assert torch.equal(after_probe, expected), (
        "the probe forward consumed random numbers the real forward was going to use"
    )


def test_the_probe_still_reports_a_raising_forward_as_no_evidence(monkeypatch):
    """NEGATIVE CONTROL: a probe whose forward raises still returns None, which caches no verdict and leaves the call on the stash path."""

    class _Raises(nn.Module):
        def forward(self, x, *args, **kwargs):
            raise RuntimeError("no")

    assert _run_probe(monkeypatch, _Raises()) is None


def test_the_probe_names_the_backend_rather_than_letting_fork_rng_guess(monkeypatch):
    """fork_rng's devices are within device_type, which otherwise comes from torch.accelerator.current_accelerator()
    (cuda fallback); unnamed, XPU forks ask the wrong module for RNG state."""
    seen = {}

    def _record(devices = None, enabled = True, device_type = None, **kwargs):
        seen["devices"] = devices
        seen["device_type"] = device_type
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.random, "fork_rng", _record)

    class _OnXPU(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    activation = torch.ones(2, 2)
    device = torch.device("xpu", 3)
    monkeypatch.setattr(type(activation), "device", property(lambda self: device), raising = False)

    with MU._preserved_rng_for_probe(activation):
        pass

    assert seen["device_type"] == "xpu"
    assert seen["devices"] == [device]


def test_a_fork_that_cannot_be_set_up_leaves_the_forward_running(monkeypatch):
    """The caller turns any exception into None, which caches no verdict and keeps the
    stash path, so a raising fork_rng would silently leave the LoRA unapplied on a family
    that ignores the stash. Failing to preserve is better than failing to probe."""

    def _explode(*args, **kwargs):
        raise RuntimeError("no generator for this device")

    monkeypatch.setattr(torch.random, "fork_rng", _explode)

    experts = _RNGConsumingExperts()
    assert _run_probe(monkeypatch, experts) is True
    assert experts.calls == 1, "the probe forward did not run"


def test_the_probe_does_not_run_at_all_while_dynamo_is_tracing(monkeypatch):
    """Under Dynamo the probe would be traced and run the experts forward three times per call (six einsums, not two;
    PyTorch 2.12 fullgraph), so it returns None while tracing."""
    experts = _RNGConsumingExperts()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert _run_probe(monkeypatch, experts) is None
    assert experts.calls == 0, "the probe forward was traced into the compiled graph"

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert _run_probe(monkeypatch, experts) is True
    assert experts.calls == 1, "the eager path must still measure"


def test_the_rng_guard_is_inert_while_tracing(monkeypatch):
    """The RNG guard keeps its own check: a generator-based context manager is a graph break inside a captured graph, and a hard error under fullgraph."""
    called = {"n": 0}

    def _count(*args, **kwargs):
        called["n"] += 1
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.random, "fork_rng", _count)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with MU._preserved_rng_for_probe(torch.ones(2, 2)):
        pass
    assert called["n"] == 0

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    with MU._preserved_rng_for_probe(torch.ones(2, 2)):
        pass
    assert called["n"] == 1, "the eager path must still preserve the RNG"


def test_a_compiled_first_call_applies_the_lora_instead_of_the_unread_stash(
    restore_param_wrapper, monkeypatch
):
    """A compiled first call cannot probe, so it folds the LoRA without PEFT's parametrization, since the stash path would let Dynamo reuse a graph with no LoRA."""
    assert MU.patch_param_wrapper_for_moe()
    x = _inputs()
    # The reference runs on its own copy, so the subject's first call is the compiled one with no verdict measured yet.
    with torch.no_grad():
        folded = _build(_StashIgnoringExperts)(x)

    model = _build(_StashIgnoringExperts)
    experts = model.base_model.model.experts.get_base_layer()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with torch.no_grad():
        traced = model(x)

    assert torch.allclose(traced, folded, atol = 1e-6), (
        "a compiled cold start left the call on a stash path nothing reads, so the "
        "expert LoRA is absent from the output"
    )
    assert MU._forward_statically_reads_stash(experts) is False
    # Nothing is recorded yet, so the first eager call still measures and later compiles use the real verdict rather than the tracing assumption.
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is None

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    with torch.no_grad():
        model(_inputs())
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is False


def test_a_compiled_first_call_keeps_the_stash_path_for_a_stash_reading_family(
    restore_param_wrapper, monkeypatch
):
    """NEGATIVE CONTROL: once an eager verdict exists, a compiled call follows it, not the tracing assumption, so stash-reading families keep their fast path."""
    assert MU.patch_param_wrapper_for_moe()
    calls = []

    def recording_original(self, x, *args, **kwargs):
        calls.append(getattr(self, "parameter_name", None))
        return self.base_layer(x, *args, **kwargs)

    for module in _moe_utils_copies():
        module._original_param_wrapper_forward = recording_original
    model = _build(_StashReadingExperts)
    experts = model.base_model.model.experts.get_base_layer()

    with torch.no_grad():
        model(_inputs())
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is True
    assert calls == []

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with torch.no_grad():
        model(_inputs())
    assert calls == [], "a recorded verdict was overridden by the tracing assumption"


def test_the_static_answer_recognises_a_stash_reading_forward():
    """_forward_statically_reads_stash finds take_moe_lora_stash in a forward's code object; assuming unread hard-fails
    fullgraph for supported families, assuming read drops the LoRA from stash-ignoring ones."""
    assert MU._forward_statically_reads_stash(_StashReadingExperts(4, 8, 4)) is True
    assert MU._forward_statically_reads_stash(_StashIgnoringExperts(4, 8, 4)) is False

    class _ReadsInAHelper(_StashIgnoringExperts):
        """Nested code objects count too, since a comprehension or inner function is often where a real forward makes the stash call."""

        def forward(self, hidden_states):
            def _inner():
                return MU.take_moe_lora_stash(self, "gate_up_proj")

            _inner()
            return super().forward(hidden_states)

    assert MU._forward_statically_reads_stash(_ReadsInAHelper(4, 8, 4)) is True

    # NEGATIVE CONTROL: nothing to read is "unknown", not "reads it".
    assert MU._forward_statically_reads_stash(object()) is None


def test_a_stash_ignoring_family_compiles_under_fullgraph(restore_param_wrapper):
    """The reroute must not cost full-graph compilation: PEFT _activate_lora registers a parametrization and calls set_
    on the stored parameter, which Dynamo refuses under fullgraph; the fold must match PEFT numbers and gradients."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    x = _inputs()

    expected = model(x)
    expected.sum().backward()
    expected_grads = {
        name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None
    }
    assert expected_grads, "the eager reference produced no gradients to compare against"
    model.zero_grad(set_to_none = True)

    compiled = torch.compile(model, fullgraph = True, dynamic = False)
    out = compiled(x)
    assert torch.allclose(out, expected, atol = 1e-5), (
        "the compiled fold does not match what PEFT's parametrization computes"
    )
    out.sum().backward()
    for name, param in model.named_parameters():
        if name in expected_grads:
            assert torch.allclose(param.grad, expected_grads[name], atol = 1e-5), name

    experts = model.base_model.model.experts.get_base_layer()
    for parameter_name in ("gate_up_proj", "down_proj"):
        assert isinstance(experts._parameters[parameter_name], torch.nn.Parameter), (
            "the swapped-in tensor outlived the call"
        )
        assert parameter_name not in experts.__dict__


def test_the_fold_swaps_inside_parameters_and_never_shadows_through_dunder_dict(restore_param_wrapper):
    """Swap the value inside _parameters, not the instance __dict__: Dynamo cannot trace removing the shadow from
    __dict__ (Unsupported on torch 2.10); _parameters traces on the whole range."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    wrapper = next(
        m for m in model.modules()
        if type(m).__name__ == "ParamWrapper" and m.parameter_name == "gate_up_proj"
    )
    experts = wrapper.get_base_layer()
    original = experts._parameters["gate_up_proj"]
    seen = {}

    def observe(x, *args, **kwargs):
        seen["read"] = getattr(experts, "gate_up_proj")
        seen["shadowed"] = "gate_up_proj" in experts.__dict__
        return x

    folded = MU._fold_moe_lora_without_parametrization(
        wrapper, observe, experts, "gate_up_proj", torch.zeros(1), (), {}
    )
    assert folded is not None, "the fold refused a plain float parameter"
    assert not seen["shadowed"], "the fold shadowed the parameter in the instance __dict__"
    assert seen["read"] is not original, "the forward read the unfolded weight"
    assert torch.allclose(
        seen["read"], MU._moe_lora_folded_weight(wrapper, original, list(wrapper.active_adapters)),
    )
    assert experts._parameters["gate_up_proj"] is original, "the parameter was not put back"
    assert "gate_up_proj" not in experts.__dict__


def test_the_fold_refuses_a_parameter_something_else_already_shadows(restore_param_wrapper):
    """NEGATIVE CONTROL: swapping _parameters under an instance-dict shadow folds into a value nothing reads, the silent no-op this file exists to rule out."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    wrapper = next(
        m for m in model.modules()
        if type(m).__name__ == "ParamWrapper" and m.parameter_name == "gate_up_proj"
    )
    experts = wrapper.get_base_layer()
    experts.__dict__["gate_up_proj"] = experts._parameters["gate_up_proj"].detach().clone()
    try:
        assert MU._fold_moe_lora_without_parametrization(
            wrapper, lambda x, *a, **k: x, experts, "gate_up_proj", torch.zeros(1), (), {}
        ) is None
    finally:
        experts.__dict__.pop("gate_up_proj", None)


def _install_peft_main_delta_factors(model):
    """Installs PEFT main's get_delta_factors verbatim, since the installed peft predates the factored API and cannot exercise the memory-relevant route."""
    import types

    def get_delta_factors(self, adapter_name):
        weight_A = self.lora_A[adapter_name].weight
        weight_B = self.lora_B[adapter_name].weight
        weight_A = weight_A.reshape(self.num_experts, -1, weight_A.shape[-1])
        weight_B = weight_B.reshape(weight_B.shape[0], -1, self.num_experts).permute(2, 0, 1)
        if not self._did_swap_in_out_features:
            lhs, rhs = weight_A.transpose(-2, -1), weight_B.transpose(-2, -1)
        else:
            lhs, rhs = weight_B, weight_A
        param = self.get_param()
        return lhs.to(param.dtype), rhs.to(param.dtype), self.scaling[adapter_name]

    wrappers = [m for m in model.modules() if type(m).__name__ == "ParamWrapper"]
    assert wrappers, "the model has no ParamWrapper to give the factored API to"
    for wrapper in wrappers:
        wrapper.get_delta_factors = types.MethodType(get_delta_factors, wrapper)
    return wrappers


def test_one_adapter_over_many_experts_folds_through_the_factors(restore_param_wrapper):
    """Fold via low-rank factors in one baddbmm as PEFT does; get_delta_weight plus the add allocate two dense
    expert-stack tensors per projection (64 experts on OlmoE)."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    wrappers = _install_peft_main_delta_factors(model)

    for wrapper in wrappers:
        param = wrapper.get_param()
        active = list(wrapper.active_adapters)
        dense = param + wrapper.get_delta_weight(active[0])
        factored = MU._moe_lora_folded_weight(wrapper, param, active)
        assert torch.allclose(dense, factored, atol = 1e-6), wrapper.parameter_name

    expected = model(_inputs())
    compiled = torch.compile(model, fullgraph = True, dynamic = False)
    assert torch.allclose(compiled(_inputs()), expected, atol = 1e-5)


def test_a_peft_without_the_factored_api_still_folds(restore_param_wrapper, monkeypatch):
    """NEGATIVE CONTROL: get_delta_factors is newer than the oldest PEFT with target_parameters, so its absence must
    fall back to the dense delta, not to nothing; 0.21.0 ships it, so the test simulates the absence."""
    assert MU.patch_param_wrapper_for_moe()
    model = _build(_StashIgnoringExperts)
    wrapper = next(m for m in model.modules() if type(m).__name__ == "ParamWrapper")
    monkeypatch.delattr(type(wrapper), "get_delta_factors", raising = False)
    assert not hasattr(wrapper, "get_delta_factors")

    param = wrapper.get_param()
    active = list(wrapper.active_adapters)
    folded = MU._moe_lora_folded_weight(wrapper, param, active)
    assert folded is not None
    assert torch.allclose(folded, param + wrapper.get_delta_weight(active[0]), atol = 1e-6)


def test_the_eager_path_still_goes_through_peft(restore_param_wrapper, monkeypatch):
    """NEGATIVE CONTROL: the parametrization-free fold is for the compiled path only; eagerly PEFT stays in charge, so its _activate_lora work is not bypassed."""
    assert MU.patch_param_wrapper_for_moe()
    calls = []

    def recording_original(self, x, *args, **kwargs):
        calls.append(getattr(self, "parameter_name", None))
        return self.base_layer(x, *args, **kwargs)

    for module in _moe_utils_copies():
        module._original_param_wrapper_forward = recording_original
    model = _build(_StashIgnoringExperts)
    with torch.no_grad():
        model(_inputs())
        model(_inputs())

    assert calls, "an eager call on a stash-ignoring family bypassed PEFT's own forward"


def test_the_static_answer_follows_a_delegating_forward():
    """The installed Qwen MoE forward dispatches: forward_moe_backend names only forward_native_grouped_mm,
    forward_triton_grouped_gemm and forward_native_moe_loop; those read the stash, so the scan must follow calls."""
    from unsloth_zoo.temporary_patches.qwen3_moe import _make_qwen_moe_experts_forward

    class _Dispatching:
        forward = staticmethod(_make_qwen_moe_experts_forward())

    assert MU._forward_statically_reads_stash(_Dispatching()) is True


def test_the_static_answer_does_not_follow_a_call_into_an_unrelated_helper():
    """NEGATIVE CONTROL: following calls must not make every forward a stash reader; a forward delegating to a helper that never reaches the stash is still False."""

    def _unrelated_helper(hidden_states):
        return hidden_states * 2

    def _delegating_forward(self, hidden_states, *args, **kwargs):
        return _unrelated_helper(hidden_states)

    class _Delegating:
        forward = _delegating_forward

    assert MU._forward_statically_reads_stash(_Delegating()) is False


def test_the_scan_terminates_on_a_recursive_call_graph():
    """A helper that calls itself, and a pair that call each other, must not loop."""

    def _self_recursive(hidden_states):
        return _self_recursive(hidden_states)

    def _ping(hidden_states):
        return _pong(hidden_states)

    def _pong(hidden_states):
        return _ping(hidden_states)

    class _Recursive:
        def forward(self, hidden_states, *args, **kwargs):
            _self_recursive(hidden_states)
            return _ping(hidden_states)

    assert MU._forward_statically_reads_stash(_Recursive()) is False


def test_every_float8_storage_dtype_is_refused_not_just_the_cuda_pair():
    """param + delta raises for all five float8 dtypes, including the fnuz pair ROCm uses; _FLOAT8_STORAGE_DTYPES must
    track PEFT's UPCAST_DTYPES, not a hand-written pair."""
    import torch
    from unsloth_zoo.temporary_patches.moe_utils import _FLOAT8_STORAGE_DTYPES

    try:
        from peft.utils import UPCAST_DTYPES
    except Exception:
        pytest.skip("this PEFT does not export UPCAST_DTYPES")

    expected = tuple(
        getattr(torch, name) for name in UPCAST_DTYPES
        if isinstance(getattr(torch, name, None), torch.dtype)
    )
    assert _FLOAT8_STORAGE_DTYPES == expected, (
        f"drifted from PEFT's UPCAST_DTYPES: {_FLOAT8_STORAGE_DTYPES} != {expected}"
    )
    assert len(_FLOAT8_STORAGE_DTYPES) >= 2

    # The reason each one is excluded, rather than an appeal to the list.
    for dtype in _FLOAT8_STORAGE_DTYPES:
        assert dtype.is_floating_point, f"{dtype} would already be refused by the float check"
        with pytest.raises(RuntimeError):
            torch.zeros(2, 2, dtype=dtype) + torch.zeros(2, 2, dtype=torch.float32)


def test_the_static_scan_does_not_wedge_dynamo(restore_param_wrapper):
    """The scan runs under Dynamo, so it must not install a guard that cannot match; id(forward) on a bound method,
    reallocated per access, failed its guard and blocked every unmeasured family from compiling."""
    import torch

    class _Experts(torch.nn.Module):
        def forward(self, x):
            return x * 2

    experts = _Experts()
    assert MU._forward_statically_reads_stash(experts) is False

    def scan(x):
        return x + (1.0 if MU._forward_statically_reads_stash(experts) else 0.0)

    for fullgraph in (True, False):
        torch._dynamo.reset()
        compiled = torch.compile(scan, fullgraph=fullgraph)
        # Two calls: the first creates the guards, the second has to match them.
        first = compiled(torch.zeros(4))
        second = compiled(torch.zeros(4))
        assert torch.equal(first, second)


def test_the_static_scan_takes_no_id_at_all():
    """The scan takes no id(): id(forward) wedges Dynamo, id(code) is rejected on some torch versions, and code-object
    keys collapse equal ones; identity via is survives all three."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(MU._forward_statically_reads_stash)))
    id_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "id"
    ]
    assert not id_calls, (
        "the scan runs while Dynamo traces, and id() is unsupported there on some torch "
        "versions; compare code objects with `is` instead"
    )


def test_a_deep_visit_does_not_suppress_a_shallower_one():
    """The scan must not depend on PYTHONHASHSEED; a plain visited set can discard a later shallower revisit after a
    deep pop at the depth limit (5 of 14 seeds gave a wrong False)."""
    import torch.nn as nn

    source = (
        'def stash_reader(m): return take_moe_lora_stash(m, "gate_up_proj")\n'
        "def helper(m): return stash_reader(m)\n"
        "def h3(m): return helper(m)\n"
        "def h2(m): return h3(m)\n"
        "def h1(m): return h2(m)\n"
        "def fwd(self, x):\n"
        "    h1(self)\n"       # deep route: helper lands at the depth limit
        "    helper(self)\n"   # shallow route: helper at depth 1, reaches the stash
        "    return x\n"
    )
    namespace = {"take_moe_lora_stash": MU.take_moe_lora_stash}
    exec(compile(source, "<depth-case>", "exec"), namespace)
    experts = type("T", (nn.Module,), {"forward": namespace["fwd"]})()

    assert MU._forward_statically_reads_stash(experts) is True


def test_the_scan_terminates_on_mutual_recursion():
    """Revisiting a node at a shallower depth must not make the walk unbounded: the depth limit caps how many times one code object is re-entered."""
    import torch.nn as nn

    source = (
        "def a(m): return b(m)\n"
        "def b(m): return a(m)\n"
        "def fwd(self, x):\n"
        "    a(self)\n"
        "    return x\n"
    )
    namespace = {}
    exec(compile(source, "<cycle>", "exec"), namespace)
    experts = type("T", (nn.Module,), {"forward": namespace["fwd"]})()
    assert MU._forward_statically_reads_stash(experts) is False


def test_equal_but_distinct_code_objects_are_both_scanned():
    """The visited map is keyed by identity; equal-but-distinct code objects (same source, different __globals__) must
    both be scanned or a stash reader answers False."""
    import torch.nn as nn

    source = "def helper(m): return target(m)\n"
    unrelated = {"target": lambda m: None}
    reaches_stash = {"target": MU.take_moe_lora_stash}
    exec(compile(source, "<same>", "exec"), unrelated)
    exec(compile(source, "<same>", "exec"), reaches_stash)

    first = unrelated["helper"].__code__
    second = reaches_stash["helper"].__code__
    assert first == second and hash(first) == hash(second), (
        "the fixture needs code objects that compare and hash equal, or it proves nothing"
    )
    assert first is not second

    namespace = {"h_unrelated": unrelated["helper"], "h_stash": reaches_stash["helper"]}
    exec(
        compile(
            "def fwd(self, x):\n    h_unrelated(self)\n    h_stash(self)\n    return x\n",
            "<fwd>", "exec",
        ),
        namespace,
    )
    experts = type("T", (nn.Module,), {"forward": namespace["fwd"]})()
    assert MU._forward_statically_reads_stash(experts) is True


@pytest.mark.parametrize("experts_cls", [_StashIgnoringExperts, _StashReadingExperts])
@pytest.mark.parametrize("fullgraph", [True, False])
def test_a_real_cold_compile_reaches_the_static_probe_without_a_guard_on_a_bound_method(
    restore_param_wrapper, experts_cls, fullgraph
):
    """REGRESSION: a real cold compile must not guard on the bound method module.forward; id() of it raised
    AssertionError (guard failed on the same frame) under both fullgraph settings, for supported families too."""
    assert MU.patch_param_wrapper_for_moe()
    x = _inputs()

    # The reference gets its own model, so the subject's first call is the compiled one and no eagerly measured verdict can stand in for the probe.
    with torch.no_grad():
        expected = _build(experts_cls)(x)

    model = _build(experts_cls)
    experts = model.base_model.model.experts.get_base_layer()
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is None, (
        "this test is only meaningful while no verdict has been recorded"
    )

    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend = "eager", fullgraph = fullgraph, dynamic = False)
        with torch.no_grad():
            out = compiled(x)
            again = compiled(_inputs())
    finally:
        torch._dynamo.reset()

    assert torch.allclose(out, expected, atol = 1e-5), (
        "a real cold compiled start does not agree with the eager fold"
    )
    assert again.shape == out.shape


@pytest.mark.parametrize(
    "experts_cls, expected",
    [
        (_StashIgnoringExperts, MU.LORA_B_LAYOUT_RANK_MAJOR),
        (_StashReadingExperts, MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT),
    ],
)
@pytest.mark.parametrize("fullgraph", [True, False])
def test_a_compiled_only_run_records_the_layout_its_forward_used(
    restore_param_wrapper, monkeypatch, tmp_path, experts_cls, expected, fullgraph
):
    """A compiled-only run never reaches the eager probe; a stash-ignoring family folds rank-major, so the marker must
    say rank_major despite the legacy UNSLOTH_MOE_LORA_B_LAYOUT declaration."""
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT)
    assert MU.patch_param_wrapper_for_moe()
    model = _build(experts_cls)
    experts = model.base_model.model.experts.get_base_layer()
    wrappers = [
        module for module in model.modules()
        if getattr(module, "parameter_name", None) in ("gate_up_proj", "down_proj")
        and hasattr(module, "lora_A")
    ]
    assert len(wrappers) == 2
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is None

    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend = "eager", fullgraph = fullgraph, dynamic = False)
        with torch.no_grad():
            compiled(_inputs())
    finally:
        torch._dynamo.reset()

    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is None
    for wrapper in wrappers:
        assert MU.moe_lora_b_layout_for_wrapper(wrapper, "default") == expected, (
            wrapper.parameter_name
        )
    model.save_pretrained(str(tmp_path))
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config.get("lora_B_layout") == expected
