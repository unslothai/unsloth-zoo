# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fused MoE expert LoRA: recording the lora_B packing, and the legacy merge (#6930).

PEFT's ParamWrapper packs lora_A grouped by expert and lora_B rank-major, and pairs them
that way in both get_delta_factors (its forward) and get_delta_weight (its merge). That is
also what vLLM and every other consumer of a saved adapter reads, so it is the layout an
Unsloth adapter has to be written in, and the separated MoE forward is being moved onto it
(unsloth_zoo#1269).

What is tested here is the other half: recording which packing an adapter was trained
with, in adapter_config.json, since Unsloth stamps no version into that file and a
marker-less adapter cannot be dated from its bytes; and the merge for an adapter the
caller has explicitly declared legacy with UNSLOTH_MOE_LORA_B_LAYOUT=grouped_by_expert,
where PEFT's own reconstruction would disagree with the forward that switch selects.

Unset, that variable leaves PEFT's merge completely alone, which is what
`test_get_delta_weight_is_pefts_own_unless_the_legacy_layout_is_declared` pins.

CPU only.
"""

from __future__ import annotations

import gc
import json
import os
import sys

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402

from unsloth_zoo.temporary_patches import moe_utils as MU  # noqa: E402


NUM_EXPERTS = 4
INTERMEDIATE = 8
HIDDEN = 12
TWO_INTER = 2 * INTERMEDIATE
# The two packings differ whenever both of these are above one; equal values do NOT make
# them agree (test_the_two_packings_differ_whenever_experts_and_rank_both_exceed_one).
# Kept unequal anyway so a reshape that confused the two axes cannot pass on shape alone.
RANK = 3
TOTAL_RANK = NUM_EXPERTS * RANK


def _peft_supports_target_parameters() -> bool:
    try:
        LoraConfig(r=1, target_parameters=["dummy"])
        return True
    except TypeError:
        return False
    except Exception:
        return True


requires_target_parameters = pytest.mark.skipif(
    not _peft_supports_target_parameters(),
    reason="PEFT < 0.18 lacks target_parameters",
)

# The lora_B packing fix (unsloth_zoo#1269) is what makes rank_major the layout the
# separated forward trains. Everything here works with or without it, except the two
# tests that are about the DEFAULT layout specifically: without the fix the separated
# forward is unconditionally grouped by expert, so there is no default to test.
requires_packing_fix = pytest.mark.skipif(
    not callable(getattr(MU, "moe_lora_b_layout", None)),
    reason="needs the lora_B packing fix (unsloth_zoo#1269)",
)


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------


@pytest.fixture
def moe_param_wrapper_patch():
    """Apply the MoE ParamWrapper patches for one test and put PEFT back afterwards, so
    the patch cannot leak into the rest of the session."""
    from peft import PeftModel
    from peft.tuners.lora.layer import ParamWrapper

    saved = (
        ParamWrapper.forward,
        ParamWrapper.get_delta_weight,
        PeftModel.save_pretrained,
    )
    assert MU.patch_param_wrapper_for_moe(), "the MoE ParamWrapper patch did not apply"
    try:
        yield
    finally:
        ParamWrapper.forward, ParamWrapper.get_delta_weight, PeftModel.save_pretrained = saved


@pytest.fixture
def legacy_lora_b_layout(monkeypatch):
    """Declare that the fused expert adapters in this process are packed the pre-fix,
    grouped-by-expert way. That declaration is the only thing that redirects PEFT's own
    get_delta_weight, and it is also what makes the separated forward read them that way
    once the packing fix is in, so with it set the forward and the merge agree whether or
    not unsloth_zoo#1269 has landed."""
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT)


@pytest.fixture
def default_lora_b_layout(monkeypatch):
    """No declaration: PEFT's rank_major, the layout every other consumer reads."""
    monkeypatch.delenv("UNSLOTH_MOE_LORA_B_LAYOUT", raising=False)


class _FusedExperts(nn.Module):
    """A fused expert stack, shaped like the real ones: one 3D Parameter for all experts."""

    num_experts = NUM_EXPERTS

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, TWO_INTER, HIDDEN))

    def forward(self, x):
        return torch.einsum("bh,eih->bei", x, self.gate_up_proj)


class _UnfusedExperts(nn.Module):
    """NemotronH keeps the projections unfused, as separate 3D Parameters, which PEFT can
    target since unsloth#11014 and which the separated forward does not claim."""

    num_experts = NUM_EXPERTS

    def __init__(self):
        super().__init__()
        self.up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, INTERMEDIATE, HIDDEN))
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE))

    def forward(self, x):
        return torch.einsum("bh,eih->bei", x, self.up_proj)


class _ToyMoE(nn.Module):
    """The parameter has to be nested: PEFT refuses to target an nn.Parameter on the
    top-level module, and the real models keep it at ...mlp.experts.gate_up_proj."""

    num_experts = NUM_EXPERTS

    def __init__(self, experts_cls=_FusedExperts):
        super().__init__()
        self.experts = experts_cls()

    def forward(self, x):
        return self.experts(x)


class _ToyMoEWithDense(_ToyMoE):
    """A fused expert stack next to an ordinary Linear, so one model can carry a fused
    expert adapter and a dense adapter at the same time."""

    def __init__(self, experts_cls=_FusedExperts):
        super().__init__(experts_cls)
        self.dense = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return self.experts(self.dense(x))


def _fused_wrappers(model):
    wrappers = {}
    for module in model.modules():
        name = getattr(module, "parameter_name", None)
        if name and hasattr(module, "lora_A"):
            wrappers[name] = module
    return wrappers


def _wrap(target_parameters, experts_cls=_FusedExperts, seed=0, model=None,
          adapter_name="default"):
    torch.manual_seed(seed)
    model = get_peft_model(
        _ToyMoE(experts_cls) if model is None else model,
        LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, bias="none",
                   target_modules=[], target_parameters=target_parameters),
        adapter_name=adapter_name,
    )
    wrappers = _fused_wrappers(model)
    assert wrappers, "PEFT wrapped no expert parameter"
    return model, wrappers


def _pristine_get_delta_weight():
    """PEFT's own ParamWrapper.get_delta_weight, whether or not the patch is installed.
    `import unsloth` applies the MoE ParamWrapper patches on transformers 5 before a
    single test runs, so reading the class attribute is not enough. The patch is built
    with functools.wraps, so the original is one hop away."""
    from peft.tuners.lora.layer import ParamWrapper

    function = ParamWrapper.get_delta_weight
    return getattr(function, "__wrapped__", function)


def _seed_lora_b(wrapper, seed=1, adapter_name="default"):
    """lora_B is initialised to zero, which hides every packing error."""
    weight = wrapper.lora_B[adapter_name].weight
    with torch.no_grad():
        weight.copy_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(seed)))
    return weight


def _separated_forward_delta(wrapper, adapter_name="default"):
    """The delta the separated MoE forward applies, in the base parameter's own layout.
    Built from the extractor the forward itself calls, so this is not a second opinion
    about the packing, it is the packing."""
    weight_A = wrapper.lora_A[adapter_name].weight
    weight_B = wrapper.lora_B[adapter_name].weight
    first, second, scaling, _ = MU.extract_moe_lora_weights_for_grouped_mm(
        wrapper, weight_A, weight_B, wrapper.scaling[adapter_name],
        int(wrapper.num_experts),
    )
    delta = torch.bmm(first.float(), second.float()) * scaling
    param = wrapper.get_param()
    if tuple(delta.shape) != tuple(param.shape):
        delta = delta.transpose(1, 2)
    return delta


# ---------------------------------------------------------------------------------------
# What PEFT does, so we notice if it ever changes
# ---------------------------------------------------------------------------------------


@requires_target_parameters
def test_peft_still_packs_lora_b_rank_major():
    """PEFT reads lora_B's E*R axis as (out, R, E), unchanged in 0.18.0, 0.19.0, 0.19.1,
    0.20.0 and 0.21.0, and that reading is what the separated forward is being aligned
    onto. If a PEFT release ever switches to (out, E, R), rank_major stops being the
    ecosystem layout and both the packing fix and the legacy merge below have to be
    revisited rather than left inverting a correct reconstruction."""
    model, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    weight_B = _seed_lora_b(wrapper)
    weight_A = wrapper.lora_A["default"].weight
    scaling = wrapper.scaling["default"]
    got = _pristine_get_delta_weight()(wrapper, "default").float()

    # PEFT #3165 (0.19.1) flipped which 3D axis counts as in_features, so which of the
    # two einsums runs depends on the release. That is the in/out orientation only; the
    # expert-vs-rank flatten order it does not touch. Read the flag off the wrapper.
    dim_B = weight_B.shape[0]
    subscript = "e o i" if getattr(wrapper, "_did_swap_in_out_features", False) else "e i o"
    A = weight_A.reshape(NUM_EXPERTS, RANK, -1).float()
    rank_major = torch.einsum(
        f"o r e, e r i -> {subscript}",
        weight_B.reshape(dim_B, RANK, NUM_EXPERTS).float(), A,
    ) * scaling
    grouped = torch.einsum(
        f"o e r, e r i -> {subscript}",
        weight_B.reshape(dim_B, NUM_EXPERTS, RANK).float(), A,
    ) * scaling

    assert torch.allclose(got, rank_major, atol=1e-5), (
        f"PEFT {peft.__version__} no longer packs lora_B rank-major; revisit "
        f"moe_utils.moe_lora_b_layout and "
        f"moe_utils._patched_param_wrapper_get_delta_weight (#6930)"
    )
    assert not torch.allclose(got, grouped, atol=1e-3), (
        "the two packings coincide in this fixture, so it cannot tell them apart"
    )


# ---------------------------------------------------------------------------------------
# The fix: the delta the merge reconstructs is the delta the forward applied
# ---------------------------------------------------------------------------------------


def test_a_bitsandbytes_without_the_params4bit_class_is_treated_as_absent():
    """The macOS bitsandbytes imports but does not expose Params4bit as a class, so the
    `isinstance(param, Params4bit)` checks in this module raised TypeError instead of
    answering False, and every fused expert layout question died with it. Measured on a
    macOS 15 runner, where 10 of these tests failed that way.

    Two halves: the invariant on whatever bitsandbytes this machine has, and the macOS
    combination itself, in a subprocess because the normalisation runs at import."""
    assert MU.Params4bit is None or isinstance(MU.Params4bit, type)
    if MU.Params4bit is None:
        assert MU.HAS_BNB is False, (
            "HAS_BNB with no Params4bit class is the combination that raises"
        )

    import subprocess
    import sys

    program = (
        "import sys, types, importlib.machinery\n"
        "mod = types.ModuleType('bitsandbytes'); nn = types.ModuleType('bitsandbytes.nn')\n"
        "nn.Params4bit = object()\n"          # imports fine, is not a class
        "mod.nn = nn\n"
        # A real install has a spec, and transformers 4.57's _is_package_available runs
        # importlib.util.find_spec on it, which raises ValueError on a spec-less module.
        # Without these two lines the stand-in fails for a reason macOS never had.
        "mod.__spec__ = importlib.machinery.ModuleSpec('bitsandbytes', None)\n"
        "nn.__spec__ = importlib.machinery.ModuleSpec('bitsandbytes.nn', None)\n"
        "mod.__version__ = '0.48.0'\n"
        "sys.modules['bitsandbytes'] = mod; sys.modules['bitsandbytes.nn'] = nn\n"
        "from unsloth_zoo.temporary_patches import moe_utils as MU\n"
        "import torch, torch.nn as tnn\n"
        "class Experts(tnn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.gate_up_proj = tnn.Parameter(torch.zeros(4, 8, 6))\n"
        "print('HAS_BNB', MU.HAS_BNB)\n"
        "print('is_moe', MU._is_moe_experts_module(Experts()))\n"
    )
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "HAS_BNB False" in done.stdout, done.stdout
    assert "is_moe True" in done.stdout, done.stdout


@requires_target_parameters
def test_layout_of_a_fused_expert_wrapper_follows_the_declared_layout(
    moe_param_wrapper_patch, legacy_lora_b_layout,
):
    _, wrappers = _wrap(["experts.gate_up_proj"])
    assert MU.moe_lora_b_layout_for_wrapper(wrappers["gate_up_proj"]) == \
        MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT


@requires_target_parameters
@requires_packing_fix
def test_layout_of_a_fused_expert_wrapper_is_rank_major_by_default(
    moe_param_wrapper_patch, default_lora_b_layout,
):
    """With the packing fix in place and nothing declared, the separated forward packs
    lora_B exactly as PEFT does, so a fused expert wrapper is rank_major like every other
    wrapper and there is nothing for the marker or the merge to correct."""
    _, wrappers = _wrap(["experts.gate_up_proj"])
    assert MU.moe_lora_b_layout_for_wrapper(wrappers["gate_up_proj"]) == \
        MU.LORA_B_LAYOUT_RANK_MAJOR


@requires_target_parameters
def test_get_delta_weight_is_pefts_own_unless_the_legacy_layout_is_declared(
    moe_param_wrapper_patch, default_lora_b_layout,
):
    """The default has to be no patch at all. PEFT's reconstruction is the correct one for
    every adapter packed the standard way, and a wrapper-routing test would redirect it
    for any fused expert adapter merged inside an Unsloth process, including a genuinely
    rank-major one that never came from Unsloth."""
    _, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    _seed_lora_b(wrapper)

    got = wrapper.get_delta_weight("default").float()
    expected = _pristine_get_delta_weight()(wrapper, "default").float()
    assert expected.abs().max().item() > 1e-3, "the fixture's delta is too small to test"
    torch.testing.assert_close(got, expected, atol=0.0, rtol=0.0)


@requires_target_parameters
def test_get_delta_weight_reconstructs_the_separated_forward_delta(
    moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """Declared legacy: the merge has to reconstruct the delta the forward that switch
    selects applies. PEFT's own reconstruction pairs expert e's lora_A rows with the wrong
    lora_B columns for such an adapter, so merge_and_unload, merge_adapter and unmerge
    would all bake a scrambled delta."""
    _, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    _seed_lora_b(wrapper)

    expected = _separated_forward_delta(wrapper)
    got = wrapper.get_delta_weight("default").float()
    assert expected.abs().max().item() > 1e-3, "the fixture's delta is too small to test"
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


@requires_target_parameters
def test_merge_then_unmerge_round_trips_the_fused_parameter(
    moe_param_wrapper_patch, legacy_lora_b_layout,
):
    _, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    _seed_lora_b(wrapper)

    before = wrapper.get_param().detach().clone()
    expected = _separated_forward_delta(wrapper)
    wrapper.merge()
    merged = wrapper.get_param().detach().clone()
    torch.testing.assert_close(merged.float(), (before.float() + expected),
                               atol=1e-6, rtol=1e-5)
    wrapper.unmerge()
    torch.testing.assert_close(wrapper.get_param().detach().float(), before.float(),
                               atol=1e-5, rtol=1e-5)


@requires_target_parameters
def test_merge_and_unload_writes_the_separated_forward_delta(
    moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """The user-facing path: `merge_and_unload` strips PEFT off and leaves one plain
    model, and the fused expert Parameter it leaves behind has to be the base weight plus
    exactly the delta the separated forward was applying."""
    model, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    _seed_lora_b(wrapper)

    before = wrapper.get_param().detach().clone()
    expected = _separated_forward_delta(wrapper)
    assert expected.abs().max().item() > 1e-3, "the fixture's delta is too small to test"

    unloaded = model.merge_and_unload()
    merged = unloaded.experts.gate_up_proj.detach()
    assert not hasattr(merged, "lora_A"), "merge_and_unload left the wrapper on"
    torch.testing.assert_close(merged.float(), before.float() + expected,
                               atol=1e-6, rtol=1e-5)


@requires_target_parameters
def test_merge_adapter_and_unmerge_adapter_go_through_the_same_delta(
    moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """`merge_adapter` and `unmerge_adapter` on the PeftModel itself, rather than the one
    wrapper: the same reconstruction has to reach them, and the round trip has to land
    back on the weight the model started with."""
    model, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    _seed_lora_b(wrapper)

    before = wrapper.get_param().detach().clone()
    expected = _separated_forward_delta(wrapper)
    assert expected.abs().max().item() > 1e-3, "the fixture's delta is too small to test"

    model.base_model.merge_adapter()
    assert wrapper.merged, "merge_adapter did not merge the fused expert wrapper"
    torch.testing.assert_close(wrapper.get_param().detach().float(),
                               before.float() + expected, atol=1e-6, rtol=1e-5)

    model.base_model.unmerge_adapter()
    assert not wrapper.merged
    torch.testing.assert_close(wrapper.get_param().detach().float(), before.float(),
                               atol=1e-5, rtol=1e-5)


@requires_target_parameters
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_delta_comes_back_in_the_parameters_dtype(
    dtype, moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """PEFT hands the merge a delta already cast to the base parameter, and merge adds it
    in place, so a reconstruction that returns float32 for a bf16 expert stack would
    either raise or silently upcast."""
    _, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    wrapper.to(dtype)
    _seed_lora_b(wrapper)
    delta = wrapper.get_delta_weight("default")
    param = wrapper.get_param()
    assert delta.dtype == param.dtype
    assert delta.device == param.device
    assert tuple(delta.shape) == tuple(param.shape)


@requires_target_parameters
def test_unfused_expert_parameters_keep_pefts_own_packing(moe_param_wrapper_patch):
    """The separated forward only claims gate_up_proj and down_proj on a fused expert
    stack. The unfused 3D pair that unsloth#11014 lets PEFT target is trained by PEFT's
    own forward, so its delta must stay PEFT's, untouched."""
    from peft.tuners.lora.layer import ParamWrapper

    _, wrappers = _wrap(["experts.up_proj", "experts.down_proj"],
                        experts_cls=_UnfusedExperts)
    for name, wrapper in wrappers.items():
        assert MU.moe_lora_b_layout_for_wrapper(wrapper) == MU.LORA_B_LAYOUT_RANK_MAJOR, name
        _seed_lora_b(wrapper)
        original = _pristine_get_delta_weight()
        torch.testing.assert_close(
            wrapper.get_delta_weight("default").float(),
            original(wrapper, "default").float(),
            atol=0.0, rtol=0.0,
        )
    assert getattr(ParamWrapper.get_delta_weight, "_unsloth_moe_layout_patched", False)


@requires_target_parameters
def test_the_patches_are_idempotent(moe_param_wrapper_patch):
    from peft import PeftModel
    from peft.tuners.lora.layer import ParamWrapper

    before = (ParamWrapper.forward, ParamWrapper.get_delta_weight, PeftModel.save_pretrained)
    assert MU.patch_param_wrapper_for_moe()
    assert MU.patch_param_wrapper_for_moe()
    after = (ParamWrapper.forward, ParamWrapper.get_delta_weight, PeftModel.save_pretrained)
    assert before[1] is after[1], "get_delta_weight was wrapped twice"
    assert before[2] is after[2], "save_pretrained was wrapped twice"
    assert not getattr(
        _pristine_get_delta_weight(), "_unsloth_moe_layout_patched", False,
    ), "the patch wrapped itself, so the original is no longer reachable"
    assert not getattr(
        getattr(PeftModel.save_pretrained, "__wrapped__", PeftModel.save_pretrained),
        "_unsloth_moe_layout_patched", False,
    ), "save_pretrained wrapped itself"


# ---------------------------------------------------------------------------------------
# The marker in adapter_config.json
# ---------------------------------------------------------------------------------------


@requires_target_parameters
def test_layout_marker_is_written_to_adapter_config(
    tmp_path, moe_param_wrapper_patch, legacy_lora_b_layout,
):
    model, _ = _wrap(["experts.gate_up_proj"])
    model.save_pretrained(str(tmp_path))

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["lora_B_layout"] == "grouped_by_expert"
    detail = config["unsloth_fused_expert_lora"]
    assert detail["lora_A_layout"] == "grouped_by_expert"
    assert detail["parameters"]["gate_up_proj"] == {
        "lora_B_layout": "grouped_by_expert",
        "num_experts": NUM_EXPERTS,
    }


@requires_target_parameters
@requires_packing_fix
def test_the_marker_records_rank_major_for_an_adapter_trained_after_the_fix(
    tmp_path, moe_param_wrapper_patch, default_lora_b_layout,
):
    """The marker is a record of how THIS adapter was trained, which is the whole of its
    use as a migration vehicle: an adapter trained with the packing fix in place says
    rank_major and needs no conversion, and one saved from a legacy run says
    grouped_by_expert and is a column permutation away (flat column e*r + j becomes
    j*E + e). A marker that always said grouped_by_expert would carry no information."""
    model, _ = _wrap(["experts.gate_up_proj"])
    model.save_pretrained(str(tmp_path))

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["lora_B_layout"] == "rank_major"
    assert config["unsloth_fused_expert_lora"]["parameters"]["gate_up_proj"] == {
        "lora_B_layout": "rank_major",
        "num_experts": NUM_EXPERTS,
    }


@requires_target_parameters
def test_the_marker_records_pefts_packing_when_peft_owns_the_wrapper(
    tmp_path, moe_param_wrapper_patch,
):
    model, _ = _wrap(["experts.up_proj", "experts.down_proj"],
                     experts_cls=_UnfusedExperts)
    model.save_pretrained(str(tmp_path))

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["lora_B_layout"] == "rank_major"
    parameters = config["unsloth_fused_expert_lora"]["parameters"]
    assert set(parameters) == {"up_proj", "down_proj"}
    assert all(entry["lora_B_layout"] == "rank_major" for entry in parameters.values())


@requires_target_parameters
def test_a_non_main_process_save_writes_no_marker(
    tmp_path, moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """is_main_process=False means another rank owns the files. PEFT takes it positionally
    as well as by keyword, so the marker has to read it the same way or every rank would
    rewrite the same adapter_config.json."""
    import inspect

    from peft import PeftModel

    original = getattr(PeftModel.save_pretrained, "__wrapped__", PeftModel.save_pretrained)
    parameters = list(inspect.signature(original).parameters)
    assert parameters[:2] == ["self", "save_directory"], parameters[:2]
    positional = [
        {"safe_serialization": True, "selected_adapters": None,
         "save_embedding_layers": "auto", "is_main_process": False}[name]
        for name in parameters[2:6]
    ]

    model, _ = _wrap(["experts.gate_up_proj"])
    path = tmp_path / "adapter_config.json"

    model.save_pretrained(str(tmp_path))
    config = json.loads(path.read_text())
    assert config["lora_B_layout"] == "grouped_by_expert"

    # Strip the marker, then save as a non-main rank. PEFT writes nothing at all in that
    # case, and neither may we: the file has to come back untouched.
    for key in ("lora_B_layout", "unsloth_fused_expert_lora"):
        config.pop(key)
    path.write_text(json.dumps(config, indent=2, sort_keys=True))
    model.save_pretrained(str(tmp_path), *positional)
    reread = json.loads(path.read_text())
    assert "lora_B_layout" not in reread
    assert "unsloth_fused_expert_lora" not in reread

    model.save_pretrained(str(tmp_path))
    assert json.loads(path.read_text())["lora_B_layout"] == "grouped_by_expert"


def test_a_dense_adapter_gets_no_marker(tmp_path, moe_param_wrapper_patch):
    torch.manual_seed(0)
    model = get_peft_model(
        nn.Sequential(nn.Linear(HIDDEN, HIDDEN)),
        LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=["0"]),
    )
    model.save_pretrained(str(tmp_path))
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert "lora_B_layout" not in config
    assert "unsloth_fused_expert_lora" not in config


@requires_target_parameters
def test_a_dense_adapter_beside_a_fused_one_gets_no_marker(
    tmp_path, moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """A PeftModel can carry both. The layout scan used to walk the whole model with no
    notion of which adapter it was describing, so the fused adapter's expert packing was
    written into the DENSE adapter's adapter_config.json as well, where it describes
    tensors that adapter does not have. A converter reading those two keys off the dense
    adapter would repack weights that are not packed that way at all."""
    model, wrappers = _wrap(["experts.gate_up_proj"], model=_ToyMoEWithDense())
    model.add_adapter(
        "dense",
        LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, bias="none",
                   target_modules=["dense"]),
    )
    wrapper = wrappers["gate_up_proj"]
    assert "dense" not in wrapper.lora_A, (
        "the dense adapter reached the fused expert wrapper, so this fixture is not the "
        "two-adapter case it means to be"
    )

    model.save_pretrained(str(tmp_path))

    fused_config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert fused_config["lora_B_layout"] == "grouped_by_expert"
    assert "gate_up_proj" in fused_config["unsloth_fused_expert_lora"]["parameters"]

    dense_config = json.loads((tmp_path / "dense" / "adapter_config.json").read_text())
    assert "lora_B_layout" not in dense_config, (
        "the fused adapter's expert packing was written into the dense adapter's config"
    )
    assert "unsloth_fused_expert_lora" not in dense_config


@requires_target_parameters
def test_the_layout_scan_is_scoped_to_one_adapter(moe_param_wrapper_patch):
    """The same defect at the source, without going through a save."""
    model, _ = _wrap(["experts.gate_up_proj"], model=_ToyMoEWithDense())
    model.add_adapter(
        "dense",
        LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, bias="none",
                   target_modules=["dense"]),
    )

    fused = MU.fused_expert_lora_layout(model, "default")
    assert fused is not None
    assert set(fused["parameters"]) == {"gate_up_proj"}
    assert MU.fused_expert_lora_layout(model, "dense") is None
    assert MU.fused_expert_lora_layout(model, "no-such-adapter") is None

    # Backwards compatibility: no adapter name still answers for the active adapter.
    assert MU.fused_expert_lora_layout(model) == fused
    model.set_adapter("dense")
    try:
        assert MU.fused_expert_lora_layout(model) is None
    finally:
        model.set_adapter("default")


@requires_target_parameters
def test_a_save_leaves_a_dense_adapters_own_keys_alone(tmp_path, moe_param_wrapper_patch):
    """The writer must not strip from an adapter it has nothing to say about. Whatever is
    in that file is PEFT's, or somebody else's, not this function's to remove."""
    model, _ = _wrap(["experts.gate_up_proj"], model=_ToyMoEWithDense())
    model.add_adapter(
        "dense",
        LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, bias="none",
                   target_modules=["dense"]),
    )
    model.save_pretrained(str(tmp_path))

    # PEFT itself rewrites adapter_config.json from the LoraConfig on every save, so plant
    # the key and call the writer directly: this is about what the writer does, not about
    # what survives a save.
    path = tmp_path / "dense" / "adapter_config.json"
    config = json.loads(path.read_text())
    config["lora_B_layout"] = "planted_by_someone_else"
    planted = json.dumps(config, indent=2, sort_keys=True)
    path.write_text(planted)

    written = MU.write_fused_expert_lora_layout(model, str(tmp_path))
    assert str(path) not in written, "the writer touched an adapter it has nothing to say about"
    assert path.read_text() == planted, "the writer rewrote a dense adapter's config"
    assert written == [str(tmp_path / "adapter_config.json")], written


@requires_target_parameters
def test_the_flat_key_is_never_the_mixed_sentinel(
    tmp_path, monkeypatch, moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """`mixed` is a report, not a layout. Written as the flat `lora_B_layout` it reads as
    rank_major to any converter that branches on `== "grouped_by_expert"`, which is
    exactly the misleading answer the flat key exists to avoid. The stale flat key an
    earlier save left behind does have to go, but only because this wrote it: a value
    this could not have written is somebody else's and stays."""
    model, _ = _wrap(["experts.gate_up_proj"])
    model.save_pretrained(str(tmp_path))
    path = tmp_path / "adapter_config.json"
    assert json.loads(path.read_text())["lora_B_layout"] == "grouped_by_expert"

    monkeypatch.setattr(MU, "fused_expert_lora_layout", lambda *args, **kwargs: {
        "lora_A_layout": "grouped_by_expert",
        "parameters": {"gate_up_proj": {"lora_B_layout": "mixed",
                                        "num_experts": NUM_EXPERTS}},
    })
    MU.write_fused_expert_lora_layout(model, str(tmp_path))

    config = json.loads(path.read_text())
    assert config["unsloth_fused_expert_lora"]["parameters"]["gate_up_proj"][
        "lora_B_layout"] == "mixed"
    assert "lora_B_layout" not in config, (
        "the mixed sentinel was written as the flat key a converter branches on"
    )

    config["lora_B_layout"] = "planted_by_someone_else"
    path.write_text(json.dumps(config, indent=2, sort_keys=True))
    MU.write_fused_expert_lora_layout(model, str(tmp_path))
    assert json.loads(path.read_text())["lora_B_layout"] == "planted_by_someone_else"


@requires_target_parameters
def test_moe_lora_b_layout_claims_nothing_for_a_wrapper_with_no_adapters(
    moe_param_wrapper_patch, legacy_lora_b_layout,
):
    """delete_adapter empties lora_A and leaves the wrapper on the model. There is no
    packing left to describe, so the unnamed form must not report one."""
    _, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    assert MU.moe_lora_b_layout_for_wrapper(wrapper) == MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT

    del wrapper.lora_A["default"]
    assert MU.moe_lora_b_layout_for_wrapper(wrapper) == MU.LORA_B_LAYOUT_RANK_MAJOR
    assert MU.moe_lora_b_layout_for_wrapper(wrapper, "default") == MU.LORA_B_LAYOUT_RANK_MAJOR


@requires_target_parameters
def test_the_marker_leaves_the_config_loadable_by_peft(tmp_path, moe_param_wrapper_patch):
    """PEFT ignores keys it does not know, so the marker must not cost a load."""
    model, _ = _wrap(["experts.gate_up_proj"])
    model.save_pretrained(str(tmp_path))
    reloaded = LoraConfig.from_pretrained(str(tmp_path))
    assert reloaded.r == RANK
    assert list(reloaded.target_parameters) == ["experts.gate_up_proj"]


@requires_target_parameters
def test_a_save_that_writes_no_config_is_not_an_error(tmp_path, moe_param_wrapper_patch):
    """The marker writer runs after PEFT's save and must never turn a completed save into
    a failure, whatever it finds on disk."""
    model, _ = _wrap(["experts.gate_up_proj"])
    written = MU.write_fused_expert_lora_layout(model, str(tmp_path / "nothing-here"))
    assert written == []


# ---------------------------------------------------------------------------------------
# The condition under which any of this matters
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_experts,rank", [
    (1, 1), (1, 4), (4, 1),          # one axis is a singleton: the two readings coincide
    (2, 2), (3, 3), (4, 4),          # equal, and still different: E*R == R*E is not it
    (4, 3), (3, 4), (6, 2),
])
def test_the_two_packings_differ_whenever_experts_and_rank_both_exceed_one(
    num_experts, rank,
):
    """The condition under which this whole patch matters. Both readings of lora_B's
    E*R axis always fit, since E*R == R*E, so nothing raises and a merge on the wrong one
    is silent. They send flat column r*E + e and flat column e*R + r to the same place,
    which agree for every (e, r) only when E == 1 or R == 1. Equal num_experts and rank
    does not make them agree."""
    out_features, in_features = 5, 7
    generator = torch.Generator().manual_seed(0)
    weight_A = torch.randn(num_experts * rank, in_features, generator=generator,
                           dtype=torch.float64)
    weight_B = torch.randn(out_features, num_experts * rank, generator=generator,
                           dtype=torch.float64)
    A = weight_A.reshape(num_experts, rank, in_features)

    # Neither reshape may raise, whatever E and R are.
    rank_major = torch.einsum("o r e, e r i -> e i o",
                              weight_B.reshape(out_features, rank, num_experts), A)
    grouped = torch.einsum("o e r, e r i -> e i o",
                           weight_B.reshape(out_features, num_experts, rank), A)

    difference = (rank_major - grouped).abs().max().item()
    if num_experts == 1 or rank == 1:
        assert difference == 0.0, (num_experts, rank, difference)
    else:
        assert difference > 1e-6, (
            f"E={num_experts}, R={rank}: the two packings agreed, so the condition in "
            f"moe_utils is wrong"
        )


# ---------------------------------------------------------------------------------------
# End to end on a real architecture, with the separated forward actually running
# ---------------------------------------------------------------------------------------


def _transformers_v5() -> bool:
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


@requires_target_parameters
@pytest.mark.skipif(not _transformers_v5(),
                    reason="the separated grouped_mm LoRA path is transformers 5 only")
@pytest.mark.parametrize("declared_layout", [
    MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT,
    pytest.param(None, marks=requires_packing_fix),
])
def test_merge_matches_the_adapter_forward_on_a_tiny_fused_moe(
    tmp_path, monkeypatch, declared_layout,
):
    """The reported symptom, end to end: a merged fused-MoE LoRA model must produce the
    logits the attached adapter produced. Uses gpt-oss because it is the fused MoE family
    a tiny config can build offline on both transformers 4 and 5.

    Both ways round. Declared legacy, the forward reads grouped by expert and the patched
    get_delta_weight has to follow it. Declared nothing, with the packing fix in place
    both sides are PEFT's rank_major and the patch must stay out of the way; that arm is
    the one that fails if the patch ever fires on routing rather than on the
    declaration."""
    if declared_layout is None:
        monkeypatch.delenv("UNSLOTH_MOE_LORA_B_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", declared_layout)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _merge_e2e_helpers as H

    if not H.family_available("gpt_oss"):
        pytest.skip("this transformers has no gpt_oss")
    H.set_offline_cpu_env()
    # patch_gpt_oss_moe_for_lora only patches the model it was loaded for.
    monkeypatch.setenv("UNSLOTH_MODEL_NAME", "unsloth/gpt-oss-20b-BF16")

    import transformers.models.gpt_oss.modeling_gpt_oss as modeling_gpt_oss
    from peft import PeftModel
    from peft.tuners.lora.layer import ParamWrapper
    from unsloth_zoo.temporary_patches.gpt_oss import patch_gpt_oss_moe_for_lora

    # patch_gpt_oss_moe_for_lora rebinds GptOssExperts.forward for the whole process.
    # Left in place it makes inspect.getsource on that class return Unsloth's wrapper,
    # which is what the upstream signature drift detectors read, so restore it here.
    experts_cls = modeling_gpt_oss.GptOssExperts
    saved = (ParamWrapper.forward, ParamWrapper.get_delta_weight, PeftModel.save_pretrained)
    saved_experts_forward = experts_cls.forward
    was_patched = hasattr(experts_cls, "_unsloth_lora_patched")
    try:
        patch_gpt_oss_moe_for_lora()
        if ParamWrapper.forward.__name__ != "_patched_param_wrapper_forward":
            pytest.skip("the separated MoE LoRA forward did not apply on this stack")

        spec = H.make_spec("gpt_oss")
        torch.manual_seed(0)
        model = H.build_and_save_base(spec, str(tmp_path / "base"), dtype=torch.float32)
        peft_model = H.attach_lora(model, spec, "expert_only", r=6)
        peft_model.eval()
        fused_wrappers = [
            module for module in peft_model.modules()
            if getattr(module, "parameter_name", None) and hasattr(module, "lora_A")
            and int(getattr(module, "num_experts", 1) or 1) > 1
        ]
        ids = torch.randint(0, 32, (1, 6))
        with torch.no_grad():
            adapter_logits = peft_model(input_ids=ids).logits.float()

        # Not a pass if the expert LoRA does not reach the forward at all.
        saved_B = {}
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if "lora_B" in name:
                    saved_B[name] = parameter.detach().clone()
                    parameter.zero_()
            zeroed_logits = peft_model(input_ids=ids).logits.float()
            for name, parameter in peft_model.named_parameters():
                if name in saved_B:
                    parameter.copy_(saved_B[name])
        effect = (adapter_logits - zeroed_logits).abs().max().item()
        if not fused_wrappers:
            pytest.skip(
                "this transformers builds the gpt-oss experts in a layout PEFT's "
                "target_parameters does not reach, so there is no fused expert LoRA to "
                "merge here"
            )
        if effect <= 1e-3:
            # Not an assertion: with no expert LoRA reaching the forward there is no
            # forward to be at parity with, and calling that a merge failure would point
            # at the wrong defect. The arithmetic itself is pinned by the contract tests
            # above, which need no forward at all.
            pytest.skip(
                f"the expert LoRA does not reach the forward on this stack (effect "
                f"{effect}), so merge parity cannot be measured here"
            )

        merged = peft_model.merge_and_unload()
        with torch.no_grad():
            merged_logits = merged(input_ids=ids).logits.float()
        torch.testing.assert_close(merged_logits, adapter_logits, atol=1e-4, rtol=1e-3)
    finally:
        ParamWrapper.forward, ParamWrapper.get_delta_weight, PeftModel.save_pretrained = saved
        if not was_patched:
            experts_cls.forward = saved_experts_forward
            for attribute in ("_unsloth_lora_patched", "_original_forward"):
                if attribute in experts_cls.__dict__:
                    delattr(experts_cls, attribute)
        # PEFT's parametrization leaves weakproxies behind, and a later test counts live
        # tensors with gc.get_objects(), where a dead weakproxy raises.
        peft_model = merged = model = None
        gc.collect()


# ---------------------------------------------------------------------------------------
# The fix has to reach the user, and it travels through unsloth_compiled_cache
# ---------------------------------------------------------------------------------------


def _write_cache_copy(directory, text):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "moe_utils.py"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_cache_copy_that_matches_this_module_is_used():
    """The control: install_to_cache writes an exact copy, which must stay usable."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        import pathlib
        path = _write_cache_copy(
            pathlib.Path(directory),
            pathlib.Path(MU.__file__).read_text(encoding="utf-8"),
        )
        assert MU._cached_copy_is_current(str(path), MU.__file__)


def test_a_cache_copy_from_another_version_is_ignored():
    """`install_to_cache` swallows a failed copy, so a cache that cannot be rewritten
    keeps an older unsloth_zoo, and every caller here prefers the cached module. That
    would install the OLD module's patches over this release's, with no error anywhere:
    the merge fix simply would not be there. Byte equality is the whole test, because
    the copy is a byte copy."""
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        stale = _write_cache_copy(
            pathlib.Path(directory),
            "# an older unsloth_zoo\ndef patch_param_wrapper_for_moe():\n    return True\n",
        )
        assert not MU._cached_copy_is_current(str(stale), MU.__file__)

        previous = os.environ.get("UNSLOTH_COMPILE_LOCATION")
        os.environ["UNSLOTH_COMPILE_LOCATION"] = directory
        try:
            assert MU._load_cached_moe_utils_module() is None, (
                "a stale cache copy must not be loaded and asked to patch"
            )
        finally:
            if previous is None:
                os.environ.pop("UNSLOTH_COMPILE_LOCATION", None)
            else:
                os.environ["UNSLOTH_COMPILE_LOCATION"] = previous


def test_an_unwritable_stale_cache_still_gets_this_releases_patches(tmp_path):
    """End to end, in a fresh interpreter, which is the only place import time
    behaviour can be measured. A pre-#6930 moe_utils.py is planted in the cache and made
    read only, so `install_to_cache` cannot refresh it, which is a container image or a
    read only mount. get_delta_weight must still end up patched."""
    import pathlib
    import subprocess
    import textwrap

    cache = tmp_path / "unsloth_compiled_cache"
    cache.mkdir()
    stale = cache / "moe_utils.py"
    # A plausible pre-#6930 module: it patches the forward and nothing else.
    stale.write_text(
        textwrap.dedent(
            """
            def patch_param_wrapper_for_moe():
                return True
            def forward_moe_backend(*args, **kwargs):
                raise NotImplementedError
            """
        ),
        encoding="utf-8",
    )
    os.chmod(stale, 0o444)

    program = textwrap.dedent(
        """
        import json, sys
        from unsloth_zoo.temporary_patches.moe_utils import patch_param_wrapper_for_moe
        patch_param_wrapper_for_moe()
        try:
            from peft.tuners.lora.layer import ParamWrapper
        except Exception:
            print(json.dumps({"skip": "no ParamWrapper"})); sys.exit(0)
        print(json.dumps({
            "get_delta_weight_patched": bool(
                getattr(ParamWrapper.get_delta_weight, "_unsloth_moe_layout_patched", False)
            ),
        }))
        """
    )
    environment = dict(os.environ)
    environment["UNSLOTH_COMPILE_LOCATION"] = str(cache)
    environment["UNSLOTH_IS_PRESENT"] = "1"
    root = str(pathlib.Path(MU.__file__).resolve().parents[3])
    environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program], env=environment,
            capture_output=True, text=True, timeout=900,
        )
    finally:
        os.chmod(stale, 0o644)

    assert completed.returncode == 0, completed.stderr[-4000:]
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    if "skip" in payload:
        pytest.skip(payload["skip"])
    assert payload["get_delta_weight_patched"], (
        "a stale unwritable unsloth_compiled_cache/moe_utils.py silently kept the "
        "pre-#6930 merge"
    )
