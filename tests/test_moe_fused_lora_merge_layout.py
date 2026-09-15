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

"""Fused MoE expert LoRA: lora_B packing, and the merge that has to honour it (#6930).

PEFT's ParamWrapper packs lora_A grouped by expert and lora_B rank-major, and pairs them
that way in get_delta_weight. Unsloth does not use PEFT's forward for fused experts; the
separated forward reads lora_B grouped by expert. Both are self-consistent, they disagree
whenever num_experts != rank, and the merge has to reconstruct whichever one trained the
adapter. CPU only, except for one GPU-gated equivalence check.
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
# Deliberately different from NUM_EXPERTS: the two packings coincide when they are equal.
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


def _wrap(target_parameters, experts_cls=_FusedExperts, seed=0):
    torch.manual_seed(seed)
    model = get_peft_model(
        _ToyMoE(experts_cls),
        LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, bias="none",
                   target_modules=[], target_parameters=target_parameters),
    )
    wrappers = {}
    for module in model.modules():
        name = getattr(module, "parameter_name", None)
        if name and hasattr(module, "lora_A"):
            wrappers[name] = module
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


def _seed_lora_b(wrapper, seed=1):
    """lora_B is initialised to zero, which hides every packing error."""
    weight = wrapper.lora_B["default"].weight
    with torch.no_grad():
        weight.copy_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(seed)))
    return weight


def _separated_forward_delta(wrapper):
    """The delta the separated MoE forward applies, in the base parameter's own layout.
    Built from the extractor the forward itself calls, so this is not a second opinion
    about the packing, it is the packing."""
    weight_A = wrapper.lora_A["default"].weight
    weight_B = wrapper.lora_B["default"].weight
    first, second, scaling, _ = MU.extract_moe_lora_weights_for_grouped_mm(
        wrapper, weight_A, weight_B, wrapper.scaling["default"], int(wrapper.num_experts),
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
    """The whole patch exists because PEFT reads lora_B's E*R axis as (out, R, E). If a
    PEFT release ever switches to (out, E, R), the two stacks agree and the patch must be
    dropped rather than left to invert a correct reconstruction."""
    model, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    weight_B = _seed_lora_b(wrapper)
    weight_A = wrapper.lora_A["default"].weight
    scaling = wrapper.scaling["default"]
    got = _pristine_get_delta_weight()(wrapper, "default").float()

    # PEFT 0.18 gives lora_A and lora_B the opposite roles from 0.19+, and picks the
    # einsum accordingly, so read both from the wrapper rather than assuming either.
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
        f"moe_utils._patched_param_wrapper_get_delta_weight (#6930)"
    )
    assert not torch.allclose(got, grouped, atol=1e-3), (
        "the two packings coincide in this fixture, so it cannot tell them apart"
    )


# ---------------------------------------------------------------------------------------
# The fix: the delta the merge reconstructs is the delta the forward applied
# ---------------------------------------------------------------------------------------


@requires_target_parameters
def test_layout_of_a_fused_expert_wrapper_is_grouped_by_expert(moe_param_wrapper_patch):
    _, wrappers = _wrap(["experts.gate_up_proj"])
    assert MU.moe_lora_b_layout(wrappers["gate_up_proj"]) == \
        MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT


@requires_target_parameters
def test_get_delta_weight_reconstructs_the_separated_forward_delta(moe_param_wrapper_patch):
    """The bug: PEFT's reconstruction pairs expert e's lora_A rows with the wrong lora_B
    columns, so merge_and_unload, merge_adapter and unmerge all bake a scrambled delta."""
    _, wrappers = _wrap(["experts.gate_up_proj"])
    wrapper = wrappers["gate_up_proj"]
    _seed_lora_b(wrapper)

    expected = _separated_forward_delta(wrapper)
    got = wrapper.get_delta_weight("default").float()
    assert expected.abs().max().item() > 1e-3, "the fixture's delta is too small to test"
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)


@requires_target_parameters
def test_merge_then_unmerge_round_trips_the_fused_parameter(moe_param_wrapper_patch):
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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_delta_comes_back_in_the_parameters_dtype(dtype, moe_param_wrapper_patch):
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
        assert MU.moe_lora_b_layout(wrapper) == MU.LORA_B_LAYOUT_RANK_MAJOR, name
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
def test_layout_marker_is_written_to_adapter_config(tmp_path, moe_param_wrapper_patch):
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
def test_a_non_main_process_save_writes_no_marker(tmp_path, moe_param_wrapper_patch):
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
# The writer in saving_utils, which reconstructs the same delta from tensors on disk
# ---------------------------------------------------------------------------------------


def _reference_delta(lora_A, lora_B, layout, num_experts, rank, alpha):
    out = []
    for expert_idx in range(num_experts):
        rows = lora_A[expert_idx * rank : (expert_idx + 1) * rank, :]
        if layout == "rank_major":
            columns = lora_B[:, expert_idx :: num_experts]
        else:
            columns = lora_B[:, expert_idx * rank : (expert_idx + 1) * rank]
        out.append(alpha * (columns @ rows))
    return torch.stack(out, 0)


@pytest.mark.parametrize("layout", ["grouped_by_expert", "rank_major"])
def test_apply_fused_expert_lora_delta_honours_the_layout(layout):
    from unsloth_zoo.saving_utils import _apply_fused_expert_lora_delta

    torch.manual_seed(0)
    dim_A, dim_B, alpha = HIDDEN, TWO_INTER, 2.0
    base = torch.randn(NUM_EXPERTS, dim_B, dim_A)
    lora_A = torch.randn(TOTAL_RANK, dim_A)
    lora_B = torch.randn(dim_B, TOTAL_RANK)

    got = _apply_fused_expert_lora_delta(
        base.clone(), lora_A, lora_B, NUM_EXPERTS, RANK, dim_A, dim_B, alpha,
        False, lora_b_layout=layout,
    )
    expected = base + _reference_delta(lora_A, lora_B, layout, NUM_EXPERTS, RANK, alpha)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    other = "rank_major" if layout == "grouped_by_expert" else "grouped_by_expert"
    assert not torch.allclose(
        got, base + _reference_delta(lora_A, lora_B, other, NUM_EXPERTS, RANK, alpha),
        atol=1e-3,
    ), "the fixture cannot tell the two layouts apart"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("layout", ["grouped_by_expert", "rank_major"])
@pytest.mark.parametrize("use_transpose", [False, True])
def test_the_batched_and_looped_fused_merges_agree(layout, use_transpose):
    """The batched bmm branch only runs on an accelerator, so CPU-only tests never
    compare it against the loop it is meant to be equivalent to."""
    from unsloth_zoo.saving_utils import _apply_fused_expert_lora_delta

    torch.manual_seed(0)
    dim_A, dim_B, alpha = HIDDEN, TWO_INTER, 2.0
    shape = (NUM_EXPERTS, dim_A, dim_B) if use_transpose else (NUM_EXPERTS, dim_B, dim_A)
    base = torch.randn(*shape)
    lora_A = torch.randn(TOTAL_RANK, dim_A)
    lora_B = torch.randn(dim_B, TOTAL_RANK)

    def _run(device):
        return _apply_fused_expert_lora_delta(
            base.clone().to(device), lora_A.to(device), lora_B.to(device),
            NUM_EXPERTS, RANK, dim_A, dim_B, alpha, use_transpose,
            lora_b_layout=layout,
        ).cpu()

    torch.testing.assert_close(_run("cuda"), _run("cpu"), atol=1e-5, rtol=1e-5)


def test_the_two_modules_agree_on_the_layout_names():
    """saving_utils keeps its own copy of the names so it never has to import the patch
    module at module scope. They have to stay the same two strings."""
    from unsloth_zoo import saving_utils

    assert saving_utils.LORA_B_LAYOUT_GROUPED_BY_EXPERT == MU.LORA_B_LAYOUT_GROUPED_BY_EXPERT
    assert saving_utils.LORA_B_LAYOUT_RANK_MAJOR == MU.LORA_B_LAYOUT_RANK_MAJOR


# ---------------------------------------------------------------------------------------
# End to end on a real architecture, with the separated forward actually running
# ---------------------------------------------------------------------------------------


def _transformers_v5() -> bool:
    import transformers
    return int(transformers.__version__.split(".")[0]) >= 5


@requires_target_parameters
@pytest.mark.skipif(not _transformers_v5(),
                    reason="the separated grouped_mm LoRA path is transformers 5 only")
def test_merge_matches_the_adapter_forward_on_a_tiny_fused_moe(tmp_path, monkeypatch):
    """The reported symptom, end to end: a merged fused-MoE LoRA model must produce the
    logits the attached adapter produced. Uses gpt-oss because it is the fused MoE family
    a tiny config can build offline on both transformers 4 and 5."""
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
