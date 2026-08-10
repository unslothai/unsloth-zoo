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

"""Head-aware device map planner: introspection and arithmetic.

These tests use meta-device models only, so they need no GPU and no weights.
They pin the parts that must stay model agnostic: finding the output head,
finding the decoder block class, honouring tied embeddings, the headroom
arithmetic, and refusing rather than silently spilling when a request cannot
fit.
"""
import sys

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import (
    DeviceMapInfeasible,
    _auto_class_for,
    _compute_module_sizes,
    _from_config_remote_aware,
    _split_units,
    _keep_in_fp32_modules,
    logit_headroom_bytes,
    plan_device_map,
    resolve_head_width,
    resolve_no_split_classes,
    resolve_output_head,
)

_GiB = 1024 ** 3


class _Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden, bias = False)


class _Tiny(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self, hidden = 64, vocab = 512, layers = 8, tie = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias = False)
        if tie:
            self.lm_head.weight = self.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def _meta(**kw):
    with torch.device("meta"):
        return _Tiny(**kw)


def test_finds_head_and_block_class():
    model = _meta()
    name, head = resolve_output_head(model)
    assert name == "lm_head"
    assert head is model.lm_head
    assert resolve_head_width(model, head) == 512
    assert "_Block" in resolve_no_split_classes(model)


def test_headroom_grows_with_rows_and_vocab():
    small = logit_headroom_bytes(1000, 128, softcapped = False)
    wide = logit_headroom_bytes(200000, 128, softcapped = False)
    assert wide > small
    more_rows = logit_headroom_bytes(200000, 256, softcapped = False)
    assert more_rows > wide


def test_headroom_retained_term_scales_with_total_rows():
    # The retained term is what makes the backward pass expensive: it depends on
    # the total number of rows, not on the chunk size, so chunking alone cannot
    # bound it.
    a = logit_headroom_bytes(200000, 128, retained_rows = 1024)
    b = logit_headroom_bytes(200000, 128, retained_rows = 4096)
    assert b > a


def test_single_device_returns_none_shaped_plan():
    model = _meta()
    plan = plan_device_map(model, max_memory = {0: "8GiB"})
    assert plan is None or set(plan.device_map.values()) == {0}


def test_two_devices_place_every_module_on_a_gpu():
    model = _meta(layers = 8)
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    assert plan is not None
    assert set(plan.device_map.values()) <= {0, 1}
    assert all(isinstance(v, int) for v in plan.device_map.values()), "no cpu or disk spill"
    assert plan.head_device in (0, 1)


def test_tied_embedding_lands_with_the_head():
    model = _meta(tie = True)
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    assert plan is not None
    assert plan.device_map["embed_tokens"] == plan.head_device


def test_refuses_instead_of_spilling_when_it_cannot_fit():
    model = _meta(hidden = 256, vocab = 4096, layers = 16)
    with pytest.raises(DeviceMapInfeasible):
        plan_device_map(
            model,
            max_memory = {0: "1MiB", 1: "1MiB"},
            headroom_bytes = 8 * _GiB,
        )


# --------------------------------------------------------------------------- #
# quantised sizing
# --------------------------------------------------------------------------- #
def test_quantised_sizes_use_the_storage_dtype():
    """`preprocess_model` leaves a meta `Params4bit` at the full unpacked shape
    in float32, so measuring the modules directly reports a 4-bit checkpoint at
    several times its real size and the planner refuses the very models it
    exists for. The quantiser turns the load dtype into the storage dtype the
    loader will really allocate; this is what transformers does before it calls
    `infer_auto_device_map`."""
    pytest.importorskip("accelerate")
    from accelerate.utils import CustomDtype

    class Q:
        """Both quantiser protocols at once.

        transformers' own `compute_module_sizes` asks for `param_element_size`;
        accelerate's has no quantiser argument and takes a storage dtype plus the
        not-converted modules. Which one the planner reaches depends on the
        installed transformers, so the double implements both and the test pins
        the same answer either way."""
        modules_to_not_convert = ["lm_head"]

        def _skipped(self, name):
            return any(m in name for m in self.modules_to_not_convert)

        # transformers >= the release that added hf_quantizer support
        def param_element_size(self, model, name, param):
            return 2 if self._skipped(name) else 0.5

        # accelerate
        def adjust_target_dtype(self, dtype):
            return CustomDtype.INT4

        def get_special_dtypes_update(self, model, dtype):
            return {n: dtype for n, _ in model.named_parameters() if self._skipped(n)}

    class Cfg:
        dtype = torch.bfloat16

    model = _meta(hidden = 64, vocab = 512, layers = 8)
    model.config = Cfg()
    plain = _compute_module_sizes(model)[""]
    quant = _compute_module_sizes(model, Q())[""]
    assert quant < plain, (quant, plain)
    # lm_head is not converted, so it keeps the load dtype rather than shrinking
    # with everything else.
    assert _compute_module_sizes(model, Q())["lm_head"] == 512 * 64 * 2


def test_no_quantizer_leaves_sizes_untouched():
    model = _meta()
    assert _compute_module_sizes(model, None) == _compute_module_sizes(model)


# --------------------------------------------------------------------------- #
# placement units
# --------------------------------------------------------------------------- #
class _RootState(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(8))
        self.register_buffer("shift", torch.zeros(8))
        self.embed_tokens = nn.Embedding(64, 16)
        self.layers = nn.ModuleList([_Block(16) for _ in range(4)])
        self.lm_head = nn.Linear(16, 64, bias = False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def test_state_owned_by_the_root_module_is_placed():
    """A parameter or buffer registered on the root has no module name to hang
    it on, and accelerate reads "" as a catch-all default, so each tensor gets
    its own entry. Without one the coverage check called it unplaced and raised
    an internal error even with memory to spare."""
    with torch.device("meta"):
        model = _RootState()
    plan = plan_device_map(model, max_memory = {0: "1GiB", 1: "1GiB"})
    assert plan.device_map["scale"] in (0, 1)
    assert plan.device_map["shift"] in (0, 1)


class _CompositeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 64, bias = False)


class _CompositeHeadModel(nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 16)
        self.layers = nn.ModuleList([_Block(16) for _ in range(4)])
        self.lm_head = _CompositeHead()

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def test_a_composite_head_is_pinned_whole():
    """`_split_units` descends into a head that has children, so the head's own
    name never appears as a unit and an exact-name filter dropped it from the
    pinned list. The greedy walk was then free to place it away from
    `head_device`, leaving the logit headroom reserved on the wrong card."""
    with torch.device("meta"):
        model = _CompositeHeadModel()
    plan = plan_device_map(model, max_memory = {0: "1GiB", 1: "1GiB"})
    assert plan.device_map["lm_head.proj"] == plan.head_device


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #
class _Sized(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))


class _Bins(nn.Module):
    _no_split_modules = ["_Sized"]

    def __init__(self, sizes):
        super().__init__()
        self.parts = nn.ModuleList([_Sized(n) for n in sizes])
        self.lm_head = nn.Linear(1, 1, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def test_differently_sized_units_still_find_a_packing():
    """Next-fit with a first-fit rescue builds loads of 9 and 9 out of
    7, 6, 3, 2, 2 into capacities 10 and 10 and then rejects the last unit,
    although 7+3 and 6+2+2 both fit. float32 makes each unit 4 bytes wide, and
    the 4-byte head is pinned, so device 1 gets 4 bytes more."""
    with torch.device("meta"):
        model = _Bins([7, 6, 3, 2, 2])
    plan = plan_device_map(
        model,
        max_memory = {0: 4 * 10, 1: 4 * 10 + 4},
        headroom_bytes = 0,
        activation_reserve_bytes = 0,
        prefer_head_device = 1,
    )
    assert plan.weight_bytes == {0: 40, 1: 44}
    assert plan.device_map["lm_head"] == 1


def test_in_order_layout_is_kept_when_it_fits():
    """The repack is a last resort. As long as the in-order walk succeeds the
    layout must stay in definition order, which keeps consecutive layers on one
    card and costs fewer cross-device hops."""
    model = _meta(layers = 8)
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    layers = [plan.device_map[f"layers.{i}"] for i in range(8)]
    assert layers == sorted(layers), layers


# --------------------------------------------------------------------------- #
# reserves
# --------------------------------------------------------------------------- #
def test_an_explicit_reserve_is_a_hard_constraint():
    """The relaxation loop walks the non-head reserve down to zero. Doing that
    to a reserve the caller measured is how a run OOMs on a card the plan still
    reports as having that reserve free, so an explicit reserve now either fits
    or the plan is refused."""
    model = _meta(hidden = 16, vocab = 64, layers = 8)
    total = _compute_module_sizes(model)[""]
    reserve = 4096
    with pytest.raises(DeviceMapInfeasible):
        plan_device_map(
            model,
            max_memory = {0: total // 2 + reserve // 2, 1: total // 2 + reserve},
            headroom_bytes = 0,
            activation_reserve_bytes = {0: reserve, 1: reserve},
        )


def test_the_auto_reserve_is_still_relaxable():
    """The default reserve is a guess, so it may shrink to make a placement fit;
    only that keeps a model that only just fits from being refused."""
    model = _meta(hidden = 16, vocab = 64, layers = 8)
    total = _compute_module_sizes(model)[""]
    plan = plan_device_map(
        model,
        max_memory = {0: total // 2 + 4096, 1: total // 2 + 4096},
        headroom_bytes = 0,
    )
    assert set(plan.device_map.values()) <= {0, 1}


# --------------------------------------------------------------------------- #
# remote code
# --------------------------------------------------------------------------- #
def test_remote_code_config_picks_the_auto_class_the_repo_registered():
    """A dynamic config is in no `_model_mapping`, so the mapping walk falls
    through to AutoModel and `from_config` then fails instead of honouring the
    `trust_remote_code` the caller passed. The repo says which Auto class it
    registered; `from_config` looks the model class up under exactly that name."""
    transformers = pytest.importorskip("transformers")

    class RemoteConfig:
        architectures = ["MyRemoteForCausalLM"]
        auto_map = {"AutoConfig": "x--MyConfig",
                    "AutoModelForCausalLM": "x--MyRemoteForCausalLM"}

    cfg = RemoteConfig()
    assert _auto_class_for(cfg, trust_remote_code = True) is transformers.AutoModelForCausalLM
    # Without permission nothing changes: the old fall-through is preserved.
    assert _auto_class_for(cfg) is transformers.AutoModel


def test_a_normal_config_is_unaffected_by_the_auto_map_branch():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.AutoConfig.for_model("llama", vocab_size = 32, hidden_size = 8,
                                            num_hidden_layers = 1, num_attention_heads = 1)
    assert _auto_class_for(cfg) is transformers.AutoModelForCausalLM
    assert _auto_class_for(cfg, trust_remote_code = True) is transformers.AutoModelForCausalLM


class _BigHead(nn.Module):
    """Four small blocks and an output head larger than half the weights."""
    _no_split_modules = ["_Block"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(32) for _ in range(4)])
        self.output = nn.Linear(32, 256, bias = False)


def test_a_model_that_trivially_fits_is_not_refused():
    """The auto reserve is capped by the smallest budget less the headroom and
    less the weight the head's device has to hold. Using only an average share
    of the weights understates that whenever the pinned units are bigger than
    the share, and since `attempt` relaxes the reserve on the OTHER cards only,
    the head's budget stays negative and every step fails. Four 4 KiB blocks and
    a 32 KiB head (share 24 KiB) were refused on 2 x 8 GiB."""
    with torch.device("meta"):
        model = _BigHead()
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    assert set(plan.device_map.values()) <= {0, 1}
    assert plan.device_map["output"] == plan.head_device
    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        assert any(name == k or name.startswith(k + ".") for k in plan.device_map), name


def test_an_unusable_prefer_head_device_says_so():
    """It used to fall through the candidate loop and report a memory
    shortfall, which points at entirely the wrong problem."""
    model = _meta()
    with pytest.raises(ValueError, match = "prefer_head_device"):
        plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"}, prefer_head_device = 7)


def test_each_backend_is_called_with_the_arguments_it_accepts(monkeypatch):
    """transformers' `compute_module_sizes(model, hf_quantizer=...)` and
    accelerate's `compute_module_sizes(model, dtype=..., special_dtypes=...)`
    do not accept each other's arguments, and the wrong pair does not fail
    loudly: it falls through to the plain walk, which measures a 4-bit
    checkpoint at several times its real size. So dispatch on the signature."""
    import types

    seen = {}

    def transformers_style(model, hf_quantizer = None, buffers_only = False, only_modules = True):
        seen["transformers"] = hf_quantizer
        return {"": 1234}, {}

    def accelerate_style(model, dtype = None, special_dtypes = None, buffers_only = False):
        seen["accelerate"] = (dtype, special_dtypes)
        return {"": 5678}

    class Q:
        modules_to_not_convert = ["lm_head"]
        def adjust_target_dtype(self, dtype):
            return torch.int8
        def get_special_dtypes_update(self, model, dtype):
            return {}

    class Cfg:
        dtype = torch.bfloat16

    model = _meta()
    model.config = Cfg()

    fake = types.ModuleType("transformers.integrations.accelerate")
    fake.compute_module_sizes = transformers_style
    monkeypatch.setitem(sys.modules, "transformers.integrations.accelerate", fake)
    assert _compute_module_sizes(model, Q())[""] == 1234
    assert isinstance(seen["transformers"], Q)

    # Same call with only accelerate's older signature available.
    seen.clear()
    fake2 = types.ModuleType("transformers.integrations.accelerate")
    fake2.compute_module_sizes = accelerate_style
    monkeypatch.setitem(sys.modules, "transformers.integrations.accelerate", fake2)
    assert _compute_module_sizes(model, Q())[""] == 5678
    assert seen["accelerate"][0] is torch.int8


# --------------------------------------------------------------------------- #
# budgets, packing and shared parameters
# --------------------------------------------------------------------------- #
_MiB = 1024 ** 2


def test_quantizer_budget_haircut_is_applied():
    """`transformers.modeling_utils._get_device_map` runs
    `hf_quantizer.adjust_max_memory(...)` before it infers a device map, and the
    bitsandbytes quantisers hold 10% of every budget back for the buffers they
    allocate while quantising. An explicit map skips that path, so without this
    the planner hands out memory the loader still needs."""

    class Q:
        def adjust_max_memory(self, max_memory):
            return {k: v * 0.90 for k, v in max_memory.items()}

    model = _meta()
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"}, hf_quantizer = Q())
    assert plan.raw_budgets[0] == int(8 * _GiB * 0.90)
    assert plan.raw_budgets[1] == int(8 * _GiB * 0.90)
    assert any("adjust_max_memory" in n for n in plan.notes)

    # A quantiser that asks for more must not talk the planner into overcommitting.
    class Greedy:
        def adjust_max_memory(self, max_memory):
            return {k: v * 4 for k, v in max_memory.items()}

    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"}, hf_quantizer = Greedy())
    assert plan.raw_budgets == {0: 8 * _GiB, 1: 8 * _GiB}


class _Uneven(nn.Module):
    """Unit sizes 2, 2, 2, 3, 5, 6 MiB plus a 20 MiB head."""

    def __init__(self):
        super().__init__()
        self.u1 = nn.Linear(256, 2048, bias = False)
        self.u2 = nn.Linear(256, 2048, bias = False)
        self.u3 = nn.Linear(256, 2048, bias = False)
        self.u4 = nn.Linear(256, 3072, bias = False)
        self.u5 = nn.Linear(256, 5120, bias = False)
        self.u6 = nn.Linear(256, 6144, bias = False)
        self.lm_head = nn.Linear(256, 20480, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def test_heterogeneous_units_are_packed_exactly():
    """Both heuristics reject this one: the in-order walk builds 2+2+2+3 then
    stalls, and best-fit packs 6+3 and 5+2+2 before rejecting the last 2, even
    though 6+2+2 and 5+3+2 fill both cards exactly."""
    with torch.device("meta"):
        model = _Uneven()
    plan = plan_device_map(
        model,
        max_memory = {0: 10 * _MiB, 1: 10 * _MiB, 2: 20 * _MiB},
        headroom_bytes = 0,
        activation_reserve_bytes = 0,
    )
    assert plan.device_map["lm_head"] == 2
    for device, budget in plan.raw_budgets.items():
        assert plan.weight_bytes[device] <= budget
    for name, _ in model.named_parameters():
        assert any(name == k or name.startswith(k + ".") for k in plan.device_map), name


class _FatBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(256, 6144, bias = False)   # 6 MiB
        self.b = nn.Linear(256, 6144, bias = False)   # 6 MiB


class _FatBlockModel(nn.Module):
    _no_split_modules = ["_FatBlock"]

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FatBlock()])
        self.lm_head = nn.Linear(256, 1024, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def test_empty_no_split_override_is_honoured():
    """`[]` means "nothing is atomic, split anywhere" and is not the same
    request as `None` ("detect the block classes"). A truthiness test threw the
    override away, so a model whose whole block is larger than any single card
    was reported infeasible even though its children fit."""
    with torch.device("meta"):
        model = _FatBlockModel()
    budgets = {0: 7 * _MiB, 1: 7 * _MiB}
    with pytest.raises(DeviceMapInfeasible):
        plan_device_map(model, max_memory = budgets, headroom_bytes = 0,
                        activation_reserve_bytes = 0)
    plan = plan_device_map(model, max_memory = budgets, headroom_bytes = 0,
                           activation_reserve_bytes = 0,
                           no_split_module_classes = [])
    assert plan.no_split_module_classes == []
    assert plan.device_map["layers.0.a"] != plan.device_map["layers.0.b"]


class _SharedProj(nn.Module):
    """A parameter shared by two modules outside the embedding pair."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(1024, 256)
        self.projA = nn.Linear(256, 4096, bias = False)
        self.filler = nn.Linear(256, 4096, bias = False)
        self.projB = nn.Linear(256, 4096, bias = False)
        self.projB.weight = self.projA.weight
        self.lm_head = nn.Linear(256, 1024, bias = False)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def test_every_shared_parameter_is_co_located():
    """Module sizing counts a shared tensor once, so splitting its owners across
    devices makes accelerate materialise a copy that `weight_bytes` never
    accounted for and a "feasible" plan OOMs during dispatch."""
    with torch.device("meta"):
        model = _SharedProj()
    plan = plan_device_map(
        model,
        max_memory = {0: 8 * _MiB, 1: 8 * _MiB},
        headroom_bytes = 0,
        activation_reserve_bytes = 0,
    )
    assert plan.device_map["projA"] == plan.device_map["projB"]


def test_reported_budgets_cover_the_pinned_head_weights():
    """`budgets` means "budget available to weights after every reserve", and
    `weight_bytes` counts the pinned head units, so the head device must not be
    reported holding more weight than its own budget allows."""
    model = _meta(tie = True)
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    for device, budget in plan.budgets.items():
        assert plan.weight_bytes[device] <= budget, (device, plan.describe())
    assert plan.budgets[plan.head_device] <= plan.raw_budgets[plan.head_device]


class _OwnStateBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(hidden))
        self.mlp = nn.Linear(hidden, hidden, bias = False)


class _OwnState(nn.Module):
    _no_split_modules = []

    def __init__(self, hidden = 64, vocab = 512):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_OwnStateBlock(hidden) for _ in range(4)])
        self.lm_head = nn.Linear(hidden, vocab, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def test_directly_owned_tensors_use_the_quantizer_aware_sizes():
    """A composite module's own parameters were measured with the live tensor's
    `element_size()`, which is exactly the number the quantiser-aware table
    exists to replace: the units then stop summing to the total and the planner
    refuses a model that fits (or overcommits one that does not)."""
    with torch.device("meta"):
        model = _OwnState()
    # A quantiser that halves every parameter, the shape of a 4-bit checkpoint
    # whose meta tensors still report the unpacked float32 size.
    sizes = {k: v // 2 for k, v in _compute_module_sizes(model).items()}
    units = _split_units(model, [], sizes)
    assert sum(size for _, size in units) == sizes[""]
    own = dict(units)["layers.0"]
    assert own == sizes["layers.0"] - sizes["layers.0.mlp"]


def test_remote_code_construction_gets_the_hub_options(monkeypatch):
    """Resolving a dynamic model class is a second Hub lookup. Handing the Hub
    options to `from_config` is not an option -- it forwards leftovers to
    `cls(config, **kwargs)`, so `token=...` is a TypeError on every ordinary
    checkpoint -- so the class is resolved explicitly instead."""
    import transformers.dynamic_module_utils as dyn

    seen = {}

    class _Remote:
        @classmethod
        def _from_config(cls, config):
            return "remote-model"

    def fake_get_class(class_reference, repo, **kwargs):
        seen["ref"] = (class_reference, repo)
        seen["kwargs"] = kwargs
        return _Remote

    monkeypatch.setattr(dyn, "get_class_from_dynamic_module", fake_get_class)

    class Cfg:
        auto_map = {"AutoModelForCausalLM": "org/repo--modeling_x.XForCausalLM"}
        _name_or_path = "org/repo"

    class AutoModelForCausalLM:
        @staticmethod
        def from_config(config, **kwargs):
            seen["fallback"] = kwargs
            return "auto-model"

    out = _from_config_remote_aware(
        AutoModelForCausalLM, Cfg(),
        {"trust_remote_code": True, "token": "t", "code_revision": "abc", "dtype": "bfloat16"},
    )
    assert out == "remote-model"
    assert seen["ref"] == ("org/repo--modeling_x.XForCausalLM", "org/repo")
    assert seen["kwargs"] == {"token": "t", "code_revision": "abc"}
    assert "fallback" not in seen

    # No auto_map entry, or no Hub options to carry: plain from_config.
    class Plain:
        auto_map = {}

    assert _from_config_remote_aware(AutoModelForCausalLM, Plain(), {"token": "t"}) == "auto-model"
    assert seen["fallback"] == {"trust_remote_code": True}


class _Composite(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.inner = nn.Linear(hidden, hidden, bias = False)


class _TiedDirectState(nn.Module):
    """A composite module whose own parameter is tied to one counted earlier."""

    def __init__(self, hidden = 16, vocab = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.trunk = _Composite(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias = False)
        # Registered after `embed_tokens`, so sizing counts the tensor there and
        # this module's own share of the size table is zero.
        self.trunk.shadow = self.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def test_zero_byte_direct_state_still_gets_a_unit():
    """Sizing counts a shared tensor once, so a direct parameter tied to one
    counted elsewhere weighs nothing. Its unit still has to exist or the
    parameter is uncovered and the coverage check raises."""
    with torch.device("meta"):
        model = _TiedDirectState()
    assert _compute_module_sizes(model)["trunk"] == 16 * 16 * 4
    assert ("trunk", 0) in _split_units(model, [], _compute_module_sizes(model))
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    # accelerate's own `check_device_map` walks `state_dict()`, which lists a
    # tied tensor under every name it has, so the map has to cover those too.
    for name in model.state_dict():
        assert any(name == k or name.startswith(k + ".") for k in plan.device_map), name


def test_recurrentgemma_softcap_alias_is_detected():
    """`logits_soft_cap` is the same knob under a different name, and the
    repository's own `_detect_logit_softcap` already treats them as aliases.
    Missing it drops the tanh temporary and the retained soft-cap buffer, so the
    head's card is under-reserved."""
    class Cfg:
        logits_soft_cap = 30.0

    model = _meta()
    model.config = Cfg()
    capped = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"},
                             retained_rows = 256)
    uncapped = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"},
                               retained_rows = 256, softcapped = False)
    assert capped.headroom_bytes > uncapped.headroom_bytes


def test_keep_in_fp32_modules_mirror_the_loader():
    """`from_pretrained` hands this list to `preprocess_model`; passing `[]`
    quantises modules the loader keeps in float32 and undercounts the weights."""
    model = _meta()

    class Cfg:
        dtype = torch.bfloat16

    model.config = Cfg()
    model._keep_in_fp32_modules = ["norm"]
    model._keep_in_fp32_modules_strict = ["lm_head"]

    class Bnb:
        use_keep_in_fp32_modules = True

        def update_dtype(self, dtype):
            return dtype

    class Other:
        use_keep_in_fp32_modules = False

        def update_dtype(self, dtype):
            return dtype

    # bfloat16: the strict list always applies, the plain one only because the
    # quantiser asks for it.
    assert _keep_in_fp32_modules(model, Bnb()) == ["norm", "lm_head"]
    assert _keep_in_fp32_modules(model, Other()) == ["lm_head"]
    assert _keep_in_fp32_modules(model, None) == ["lm_head"]


def test_seq2seq_checkpoints_get_their_conditional_generation_class():
    """T5 and mT5 are registered only under `AutoModelForSeq2SeqLM`, so the walk
    fell through to the bare `AutoModel`: a meta model with no output head, an
    undercounted weight total and the headroom reserved around some unrelated
    linear."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.T5Config(vocab_size = 32, d_model = 8, num_layers = 1,
                                num_heads = 1, d_ff = 16)
    assert _auto_class_for(cfg) is transformers.AutoModelForSeq2SeqLM

    # A config in two mappings is decided by the checkpoint's `architectures`.
    bart = transformers.BartConfig(vocab_size = 32, d_model = 8,
                                   encoder_layers = 1, decoder_layers = 1,
                                   encoder_attention_heads = 1,
                                   decoder_attention_heads = 1)
    assert _auto_class_for(bart) is transformers.AutoModelForCausalLM
    bart.architectures = ["BartForConditionalGeneration"]
    assert _auto_class_for(bart) is transformers.AutoModelForSeq2SeqLM


def test_model_declaring_an_empty_no_split_list_is_believed():
    """`[]` is the model saying nothing is atomic, which camembert, colpali,
    colqwen2, efficientnet and fuyu all do. Only `None` means "not declared"."""
    model = _meta()
    model._no_split_modules = []
    assert resolve_no_split_classes(model) == []
    model._no_split_modules = None
    assert "_Block" in resolve_no_split_classes(model)


def test_a_failed_dynamic_resolution_keeps_the_hub_restrictions(monkeypatch):
    """Retrying through `from_config` after a failed lookup would go to the
    network under `local_files_only`, drop the token, or take a different code
    revision, so the failure has to surface instead."""
    import transformers.dynamic_module_utils as dyn

    def boom(class_reference, repo, **kwargs):
        raise OSError("not found in the local cache")

    monkeypatch.setattr(dyn, "get_class_from_dynamic_module", boom)

    class Cfg:
        auto_map = {"AutoModelForCausalLM": "org/repo--modeling_x.XForCausalLM"}
        _name_or_path = "org/repo"

    class AutoModelForCausalLM:
        @staticmethod
        def from_config(config, **kwargs):
            raise AssertionError("must not retry without the Hub options")

    with pytest.raises(OSError, match = "local cache"):
        _from_config_remote_aware(
            AutoModelForCausalLM, Cfg(),
            {"trust_remote_code": True, "local_files_only": True},
        )


class _Wrapped(nn.Module):
    """The head lives inside a module the caller declares atomic."""

    def __init__(self, hidden = 64, vocab = 512):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(4)])
        self.language_model = nn.Module()
        self.language_model.norm = nn.LayerNorm(hidden)
        self.language_model.lm_head = nn.Linear(hidden, vocab, bias = False)

    def get_output_embeddings(self):
        return self.language_model.lm_head


def test_an_atomic_ancestor_of_the_head_is_pinned():
    """`_split_units` emits the atomic ancestor, not the nested head name, so a
    descendants-only filter left `pinned` empty and the logit headroom was
    reserved on a card that need not hold the head at all."""
    with torch.device("meta"):
        model = _Wrapped()
    plan = plan_device_map(
        model,
        max_memory = {0: "8GiB", 1: "8GiB"},
        no_split_module_classes = ["Module", "_Block"],
    )
    assert plan.head_module == "language_model.lm_head"
    assert plan.device_map["language_model"] == plan.head_device


def test_logit_scaling_adds_one_chunk_buffer():
    """`chunk_logits * logit_scale_multiply` and `/ logit_scale_divide` are both
    out of place, so one more buffer of the chunk shape in the logit dtype is
    live before the float32 copy."""
    plain = logit_headroom_bytes(200000, 128, softcapped = False, safety_bytes = 0)
    scaled = logit_headroom_bytes(200000, 128, softcapped = False, logit_scaled = True,
                                  safety_bytes = 0)
    s = torch.empty((), dtype = torch.bfloat16).element_size()
    assert scaled - plain == 128 * 200000 * s

    model = _meta()
    budgets = {0: "8GiB", 1: "8GiB"}
    assert (plan_device_map(model, max_memory = budgets, logit_scaled = True).headroom_bytes >
            plan_device_map(model, max_memory = budgets).headroom_bytes)


def test_the_plan_reports_the_reserve_each_device_kept():
    """A per-device mapping is not one number, and the auto reserve is relaxed on
    the non-head cards, so a single scalar promised headroom the accepted
    packing had not kept."""
    model = _meta()
    plan = plan_device_map(
        model,
        max_memory = {0: "8GiB", 1: "8GiB"},
        activation_reserve_bytes = {0: 1 * _GiB, 1: 3 * _GiB},
    )
    assert plan.activation_reserve_by_device == {0: 1 * _GiB, 1: 3 * _GiB}
    for device, kept in plan.activation_reserve_by_device.items():
        head_extra = plan.headroom_bytes if device == plan.head_device else 0
        assert plan.weight_bytes[device] + kept + head_extra <= plan.raw_budgets[device]
    assert "1.000 GiB" in plan.describe()


class _SharedLayerStack(nn.Module):
    """The same block object registered under several names."""

    def __init__(self, hidden = 64, vocab = 512):
        super().__init__()
        block = _Block(hidden)
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([block] * 4)
        self.lm_head = nn.Linear(hidden, vocab, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def test_repeated_module_registrations_are_planned():
    """`named_children()` drops repeated registrations of the same object, so
    the aliases never became placement units while the coverage check still
    enumerated them, and a model that fits was reported as an internal error."""
    with torch.device("meta"):
        model = _SharedLayerStack()
    assert len(model.layers) == 4
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    for name, _ in model.named_parameters(remove_duplicate = False):
        assert any(name == k or name.startswith(k + ".") for k in plan.device_map), name
    # One shared object, so every alias has to land on the same device.
    assert len({plan.device_map[f"layers.{i}"] for i in range(4)}) == 1


def test_infeasible_reports_each_devices_reserve():
    """The message promises exact numbers, so a per-device mapping cannot be
    collapsed to its maximum and multiplied by the device count: 1 GiB and
    3 GiB reserved is 4 GiB, not 6 GiB."""
    model = _meta(hidden = 256, vocab = 4096, layers = 16)
    with pytest.raises(DeviceMapInfeasible) as excinfo:
        plan_device_map(
            model,
            max_memory = {0: "2GiB", 1: "2GiB"},
            headroom_bytes = 8 * _GiB,
            activation_reserve_bytes = {0: 1 * _GiB, 1: 3 * _GiB},
        )
    message = str(excinfo.value)
    assert "cuda:0=1.000 GiB, cuda:1=3.000 GiB" in message
    assert "(4.000 GiB total)" in message


def test_head_max_fills_the_other_cards_before_a_low_index_head():
    """`head_max` exists to push weight off the head's card. Walking plain
    device order filled the head's own card first whenever the head was not the
    last device."""
    model = _meta(layers = 8)
    plan = plan_device_map(
        model,
        max_memory = {0: "8GiB", 1: "8GiB"},
        free_space_policy = "head_max",
        prefer_head_device = 0,
    )
    assert plan.head_device == 0
    assert plan.weight_bytes[1] > 0
    assert plan.weight_bytes[1] >= plan.weight_bytes[0]


def test_fp32_loader_exceptions_apply_without_a_quantizer():
    """An fp16 checkpoint whose class declares `_keep_in_fp32_modules` (the T5
    family keeps `wo` that way) has those tensors loaded as float32, while the
    meta model still holds them in the config dtype."""
    pytest.importorskip("accelerate")

    class Cfg:
        dtype = torch.float16

    with torch.device("meta"):
        model = _Tiny(hidden = 64, vocab = 512, layers = 4)
    model.half()
    model.config = Cfg()
    plain = _compute_module_sizes(model)
    model._keep_in_fp32_modules = ["norm"]
    upcast = _compute_module_sizes(model)
    # The norm doubles from float16 to float32; nothing else moves.
    assert upcast["norm"] == plain["norm"] * 2
    assert upcast["layers"] == plain["layers"]
    assert upcast[""] == plain[""] + plain["norm"]
