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

import contextlib
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


# ------------------------------------------- what the measuring call is allowed to cost


def test_non_reentrant_gradient_checkpointing_survives_the_measurement(restore_param_wrapper):
    """The measuring call must not save tensors the later calls do not save.

    `torch.utils.checkpoint` with `use_reentrant = False` replays the region in the
    backward and refuses, with `CheckpointError: A different number of tensors was saved
    during the original forward and recomputation`, if the replay does not save what the
    forward saved. Measuring in band cannot satisfy that: the first call runs the experts
    forward to find out and then again through PEFT when the answer is no, while the
    recompute already has the verdict and runs it once.

    This is the configuration transformers 5 chooses by default, from
    `modeling_utils.gradient_checkpointing_enable`:
    `gradient_checkpointing_kwargs = {"use_reentrant": False}`, and the one Unsloth's own
    vision path selects whenever the run is distributed. Both halves of the family split
    are checked, because a fix that only reroutes differently would still leave the
    stash-reading half saving a different number of tensors on its first call.
    """
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
    """Steady state is one experts forward per call, for both halves of the family split.

    The measurement is a one-off. If it were not, a family that ignores the stash would pay
    the double call on every step, and a family that reads it would stop being byte for
    byte what main runs.
    """
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
    """PEFT folds by adding a delta to the stored parameter, which needs a real float.

    A stacked expert weight held as `Params4bit`, MXFP4 blocks or FP8 is not one, so the
    reroute must decline and leave that case as it is on main. The MXFP4 GPT-OSS experts
    forward is the live one: it reads its LoRA through `_get_lora_wrapper_for_param`, which
    resolves to None against PEFT's `target_parameters` layout, so nothing records a read
    and the verdict for that forward is False.
    """
    # Through monkeypatch: a saved forward left behind here is the one every later test
    # ends up calling, and a stand-in that returns None would make those look like a model
    # producing no output at all.
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
    """The probe must not keep a graph alive while PEFT's forward allocates its own.

    An earlier shape of this code ran the probe in-band and held its output in a local
    across the fallback call, so that forward's expert activations stayed resident, by way
    of the result's grad_fn, while the second forward built its own set. MoE training sits
    near the memory limit and this happens once per newly probed layer on the FIRST step,
    so the doubling reads as a first-step OOM that later steps never reproduce.

    `_measure_moe_lora_stash_read` answers it structurally rather than by releasing
    afterwards: the probe runs under `torch.no_grad()` and its return value is not bound at
    all, so there is no graph to retain. Both halves are asserted, because either one alone
    can be lost in a refactor and neither is visible in a passing functional test.
    """
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
    """The same property from the outside: when PEFT's forward runs, no local in the
    patched forward is holding a tensor that carries a grad_fn."""
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
    """Drive `_measure_moe_lora_stash_read` with the surrounding machinery stubbed out,
    so what the test observes is the probe forward and nothing else."""
    monkeypatch.setattr(MU, "_extract_lora_from_wrapper", lambda wrapper: None)
    monkeypatch.setattr(MU, "_reset_moe_lora_stash_read", lambda module, name: None)
    monkeypatch.setattr(MU, "_moe_lora_stash_was_read", lambda module, name: True)
    monkeypatch.setattr(MU, "_record_moe_lora_forward_verdict", lambda module, name, read: None)
    return MU._measure_moe_lora_stash_read(
        _ProbeWrapper(experts), experts, "gate_up_proj", torch.ones(4, 8), (), {}
    )


def test_the_probe_forward_leaves_the_rng_where_it_found_it(monkeypatch):
    """`no_grad` turns off the graph, it does not preserve RNG.

    Every random draw in the throwaway forward advances the generator, so without a fork
    the call that counts gets different numbers than it would have without the probe. The
    sharp edge is gradient checkpointing: non-reentrant checkpointing restores the RNG
    state at the start of the region and replays it, and the original pass runs probe plus
    real forward while the recompute has the verdict cached and runs the real forward
    alone, so the two draw different dropout masks for the same region and the gradients
    are computed against a mask the forward never used.
    """
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
    """NEGATIVE CONTROL: the fork must not swallow the failure path. A probe that raises
    still returns None, which caches nothing and leaves the call on the stash path."""

    class _Raises(nn.Module):
        def forward(self, x, *args, **kwargs):
            raise RuntimeError("no")

    assert _run_probe(monkeypatch, _Raises()) is None


def test_the_probe_names_the_backend_rather_than_letting_fork_rng_guess(monkeypatch):
    """`fork_rng`'s `devices` identifies devices WITHIN `device_type`, which it otherwise
    resolves from `torch.accelerator.current_accelerator()` and falls back to "cuda" for.
    An XPU activation forked without naming the backend asks the wrong module for a
    generator state."""
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
    """Dynamo has no notion of a one-time measurement.

    It traces the throwaway forward into the graph beside the call that counts, so the
    compiled region runs the experts forward three times on EVERY invocation rather than
    once during a warm-up: six expert einsums in one graph where two are correct, measured
    on PyTorch 2.12 with fullgraph. Returning None caches no verdict and leaves the call on
    the stash path, which is what this module did before the probe existed.
    """
    experts = _RNGConsumingExperts()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert _run_probe(monkeypatch, experts) is None
    assert experts.calls == 0, "the probe forward was traced into the compiled graph"

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert _run_probe(monkeypatch, experts) is True
    assert experts.calls == 1, "the eager path must still measure"


def test_the_rng_guard_is_inert_while_tracing(monkeypatch):
    """The helper is usable on its own, so it keeps its own check: a generator-based
    context manager in a captured graph is a break, and under fullgraph a hard error."""
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
    """A model whose FIRST invocation is compiled has no verdict and no way to get one.

    The probe cannot run inside a captured graph without being captured with it and re-run
    on every call. Leaving the call inconclusive took the separated stash path, and on a
    family whose experts forward ignores the stash Dynamo would reuse that graph forever
    with the expert LoRA absent from every output and gradient. The fold has to happen
    instead, and while tracing it happens without PEFT's parametrization.
    """
    assert MU.patch_param_wrapper_for_moe()
    x = _inputs()
    # The reference runs on its own copy, so the subject's FIRST call is the compiled one
    # and no verdict has been measured for it.
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
    # Nothing recorded, so the first eager call still measures and every later compile
    # uses the real verdict rather than this assumption.
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is None

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    with torch.no_grad():
        model(_inputs())
    assert MU.moe_lora_forward_applies_stash(experts, "gate_up_proj") is False


def test_a_compiled_first_call_keeps_the_stash_path_for_a_stash_reading_family(
    restore_param_wrapper, monkeypatch
):
    """NEGATIVE CONTROL: the assumption must not cost the supported families their fast
    path when a verdict already exists. Once measured eagerly, a compiled call follows the
    recorded verdict, not the tracing assumption."""
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
    """The thing the compiled cold start is decided on, on its own.

    Every forward that applies the separated expert LoRA calls take_moe_lora_stash, so the
    name in its code object is what tells an Unsloth-installed forward from transformers'
    own. Both wrong answers are costly, which is why the decision is not a fixed
    assumption: assuming unread sends a supported family into PEFT's ParamWrapper.forward,
    which registers a parametrization while tracing and hard-fails under fullgraph;
    assuming read leaves the LoRA out of every output on a family that ignores the stash.
    """
    assert MU._forward_statically_reads_stash(_StashReadingExperts(4, 8, 4)) is True
    assert MU._forward_statically_reads_stash(_StashIgnoringExperts(4, 8, 4)) is False

    class _ReadsInAHelper(_StashIgnoringExperts):
        """Nested code objects count too: a comprehension or an inner function is where a
        real forward often makes the call."""

        def forward(self, hidden_states):
            def _inner():
                return MU.take_moe_lora_stash(self, "gate_up_proj")

            _inner()
            return super().forward(hidden_states)

    assert MU._forward_statically_reads_stash(_ReadsInAHelper(4, 8, 4)) is True

    # NEGATIVE CONTROL: nothing to read is "unknown", not "reads it".
    assert MU._forward_statically_reads_stash(object()) is None


def test_a_stash_ignoring_family_compiles_under_fullgraph(restore_param_wrapper):
    """The reroute must not cost these families full-graph compilation.

    PEFT's `_activate_lora` registers a parametrization for the call, and the `set_` that
    performs on the stored parameter is a graph-input mutation Dynamo refuses with
    "Getting an inplace view on a graph input is not supported", which is a hard error
    under fullgraph. Folding the same delta ourselves has to give PEFT's numbers, PEFT's
    gradients, and a model that still compiles.
    """
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
    """The swap has to be one Dynamo can trace, and `__dict__` is not one.

    Shadowing the registered parameter with an entry in the instance `__dict__` reads
    correctly and restores correctly, and it is still wrong here: `experts_module.__dict__`
    is an unknown type to Dynamo, so taking the shadow back out raises `Unsupported:
    Dynamo does not know how to trace method 'pop' of class '<unknown type>'` on torch
    2.10, which is inside the range this package supports. Swapping the value inside
    `_parameters`, which `nn.Module.__getattr__` reads, traces on the whole range.
    """
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
    """NEGATIVE CONTROL: swapping `_parameters` under a shadow folds into a value nothing
    reads, which is the silent no-op this whole file exists to rule out."""
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
    """PEFT main's `get_delta_factors` on this model's wrappers, for a PEFT without it.

    The installed peft predates the factored API, so the route that matters for memory
    cannot be exercised through it. This is upstream's implementation verbatim.
    """
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
    """PEFT keeps the low-rank factors for this case, and so must this fold.

    `get_delta_weight` materialises a delta the size of the whole expert stack, and the
    add then allocates a second one. On OlmoE that is 64 experts of dense weight twice per
    projection per layer, which is what a single `baddbmm` over the factors avoids. The
    result has to be the same weight either way.
    """
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
    """NEGATIVE CONTROL: `get_delta_factors` is newer than the oldest PEFT that has
    `target_parameters`, so its absence must fall back to the dense delta, not to nothing.

    The absence is simulated rather than asserted, since peft>=0.18.0 is unbounded here and
    0.21.0 already ships the factored API, so asserting it would fail on a resolution the
    package allows.
    """
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
    """NEGATIVE CONTROL: the parametrization-free fold is for the compiled path only.
    Eagerly PEFT stays in charge, so nothing it does inside `_activate_lora` is bypassed
    on a path where it runs perfectly well."""
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
    """The installed Qwen MoE forward is a dispatcher, so the scan has to follow the call.

    `_make_qwen_moe_experts_forward` installs `forward_moe_backend`, whose own code object
    names only `forward_native_grouped_mm`, `forward_triton_grouped_gemm` and
    `forward_native_moe_loop`. Each of those reads the stash, so stopping at the
    dispatcher calls these supported families stash-ignorant and sends their first
    compiled call into PEFT's parametrization path, which is the fullgraph failure this
    patch exists to remove.
    """
    from unsloth_zoo.temporary_patches.qwen3_moe import _make_qwen_moe_experts_forward

    class _Dispatching:
        forward = staticmethod(_make_qwen_moe_experts_forward())

    assert MU._forward_statically_reads_stash(_Dispatching()) is True


def test_the_static_answer_does_not_follow_a_call_into_an_unrelated_helper():
    """NEGATIVE CONTROL: following calls must not turn every forward into a stash reader.
    A forward that delegates to a helper which never reaches the stash is still False."""

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
    """`param + delta` raises for all five float8 dtypes, not the two named originally.

    Naming `float8_e4m3fn` and `float8_e5m2` by hand let the `fnuz` pair that ROCm uses
    through, so the compiled fold was permitted on a weight it cannot fold, turning a
    quantized path that used to be left alone into a hard failure. PEFT lists the whole set
    in `UPCAST_DTYPES`; this pins that we track it rather than restate it.
    """
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
