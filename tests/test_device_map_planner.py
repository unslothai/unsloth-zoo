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

"""Head-aware device map planner tests on meta-device models (no GPU or weights); the parts that must stay model
agnostic: output head and decoder block discovery, tied embeddings, headroom, refusal over spilling."""
import sys

import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import (
    DeviceMapInfeasible,
    detect_logit_transforms,
    _auto_class_for,
    _compute_module_sizes,
    _from_config_remote_aware,
    _split_units,
    _keep_in_fp32_modules,
    _runtime_quantization_config,
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
    # The retained term scales with total rows, not chunk size, so chunking alone cannot bound it.
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


def test_quantised_sizes_use_the_storage_dtype():
    """preprocess_model leaves meta Params4bit at the full unpacked float32 shape, so sizes must use the quantiser's
    storage dtype (as transformers does) or 4-bit models are overstated."""
    pytest.importorskip("accelerate")
    from accelerate.utils import CustomDtype

    class Q:
        """Fake quantiser implementing both protocols: transformers' compute_module_sizes asks for param_element_size,
        accelerate's takes a storage dtype and not-converted modules; the answer must match either way."""
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
    # lm_head is not converted, so it keeps the load dtype.
    assert _compute_module_sizes(model, Q())["lm_head"] == 512 * 64 * 2


def test_no_quantizer_leaves_sizes_untouched():
    model = _meta()
    assert _compute_module_sizes(model, None) == _compute_module_sizes(model)


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
    """Root-owned params or buffers have no module name and accelerate reads '' as a catch-all, so each tensor gets its
    own entry, else coverage raises despite free memory."""
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
    """_split_units descends into a head with children, so its own name is never a unit; an exact-name filter drops it
    from the pinned list and the walk can place it off head_device."""
    with torch.device("meta"):
        model = _CompositeHeadModel()
    plan = plan_device_map(model, max_memory = {0: "1GiB", 1: "1GiB"})
    assert plan.device_map["lm_head.proj"] == plan.head_device


class _Sized(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n))


class _Bins(nn.Module):
    _no_split_modules = ["_Sized"]

    def __init__(self, sizes, head = 1):
        super().__init__()
        self.parts = nn.ModuleList([_Sized(n) for n in sizes])
        self.lm_head = nn.Linear(head, 1, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def test_differently_sized_units_still_find_a_packing():
    """Next-fit with first-fit rescue builds loads of 9 and 9 from 7, 6, 3, 2, 2 in capacities 10 and 10 and rejects the
    last unit though 7+3 and 6+2+2 fit; float32 makes each unit 4 bytes wide, and the pinned 4-byte head adds 4 bytes
    to device 1."""
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
    """The repack is a last resort: while the in-order walk succeeds the layout stays in definition order, keeping
    consecutive layers on one card with fewer cross-device hops."""
    model = _meta(layers = 8)
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    layers = [plan.device_map[f"layers.{i}"] for i in range(8)]
    assert layers == sorted(layers), layers


def test_an_explicit_reserve_is_a_hard_constraint():
    """The relaxation loop must not lower an explicit, caller-measured reserve; doing so OOMs a card the plan still
    reports as free, so an explicit reserve fits or is refused."""
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
    """The default reserve is a guess and may shrink to make a placement fit, which keeps a model that only just fits from being refused."""
    model = _meta(hidden = 16, vocab = 64, layers = 8)
    total = _compute_module_sizes(model)[""]
    plan = plan_device_map(
        model,
        max_memory = {0: total // 2 + 4096, 1: total // 2 + 4096},
        headroom_bytes = 0,
    )
    assert set(plan.device_map.values()) <= {0, 1}


def test_remote_code_config_picks_the_auto_class_the_repo_registered():
    """Dynamic configs are in no _model_mapping, so from_config falls to AutoModel and fails instead of honouring
    trust_remote_code; use the Auto class the repo registered."""
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
    """Auto reserve is capped by the smallest budget less headroom and the head device's own weight, not an average
    share; attempt relaxes only the other cards, so the head budget stays negative."""
    with torch.device("meta"):
        model = _BigHead()
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    assert set(plan.device_map.values()) <= {0, 1}
    assert plan.device_map["output"] == plan.head_device
    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        assert any(name == k or name.startswith(k + ".") for k in plan.device_map), name


def test_an_unusable_prefer_head_device_says_so():
    """An unusable prefer_head_device must raise its own ValueError, not fall through the candidate loop and report a misleading memory shortfall."""
    model = _meta()
    with pytest.raises(ValueError, match = "prefer_head_device"):
        plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"}, prefer_head_device = 7)


def test_each_backend_is_called_with_the_arguments_it_accepts(monkeypatch):
    """transformers and accelerate compute_module_sizes take different kwargs, and a mismatched pair silently falls back
    to the plain walk (4-bit sized several times too big), so dispatch on signature."""
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


_MiB = 1024 ** 2


def test_quantizer_budget_haircut_is_applied():
    """transformers' _get_device_map calls hf_quantizer.adjust_max_memory first, and bitsandbytes quantisers hold 10% of
    each budget back; an explicit map skips that, so the planner must apply it too."""

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
    """Both heuristics reject this case: in-order packing stalls at 2+2+2+3 and best-fit packs 6+3 and 5+2+2, yet 6+2+2 and 5+3+2 fill both cards exactly."""
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
    """`[]` (nothing atomic) differs from `None` (detect block classes); a truthiness test dropped `[]`, so a block
    bigger than any card was wrongly reported infeasible."""
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
    """Sizing counts a shared tensor once, so splitting its owners across devices makes accelerate copy it uncounted in `weight_bytes`, OOMing at dispatch."""
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
    """`budgets` is the room left for weights after every reserve; `weight_bytes` includes the pinned head, so the head device must not exceed its budget."""
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
    """Direct params of a composite need quantiser-aware sizes, not the live `element_size()`; otherwise units stop summing to the total."""
    with torch.device("meta"):
        model = _OwnState()
    # A 4-bit checkpoint whose meta tensors still report the unpacked float32 size.
    sizes = {k: v // 2 for k, v in _compute_module_sizes(model).items()}
    units = _split_units(model, [], sizes)
    assert sum(size for _, size in units) == sizes[""]
    own = dict(units)["layers.0"]
    assert own == sizes["layers.0"] - sizes["layers.0.mlp"]


def test_remote_code_construction_gets_the_hub_options(monkeypatch):
    """Hub options cannot go via from_config, which forwards leftovers to cls(config, **kwargs) so token=... is a TypeError; resolve the class explicitly."""
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
        # Registered after `embed_tokens`, so sizing counts the tensor there.
        self.trunk.shadow = self.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def test_zero_byte_direct_state_still_gets_a_unit():
    """A direct parameter tied to a tensor counted elsewhere weighs zero, but its unit must still exist or the coverage check raises for it."""
    with torch.device("meta"):
        model = _TiedDirectState()
    assert _compute_module_sizes(model)["trunk"] == 16 * 16 * 4
    assert ("trunk", 0) in _split_units(model, [], _compute_module_sizes(model))
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    # accelerate's `check_device_map` walks `state_dict()`, which lists a tied tensor under every name, so the map must cover all of them.
    for name in model.state_dict():
        assert any(name == k or name.startswith(k + ".") for k in plan.device_map), name


def test_recurrentgemma_softcap_alias_is_detected():
    """`logits_soft_cap` is an alias that `_detect_logit_softcap` already handles; missing it drops the tanh temporary and retained buffer, under-reserving the head."""
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
    """`from_pretrained` passes this list to `preprocess_model`; `[]` would quantise modules the loader keeps in float32, undercounting weights."""
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

    # bfloat16: the strict list always applies, the plain one only on request.
    assert _keep_in_fp32_modules(model, Bnb()) == ["norm", "lm_head"]
    assert _keep_in_fp32_modules(model, Other()) == ["lm_head"]
    assert _keep_in_fp32_modules(model, None) == ["lm_head"]


def test_seq2seq_checkpoints_get_their_conditional_generation_class():
    """T5 and mT5 are registered only under `AutoModelForSeq2SeqLM`; falling through to `AutoModel` drops the output head and undercounts weights."""
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
    """`[]` means the model declares nothing atomic (camembert, colpali, colqwen2, efficientnet and fuyu do this); only `None` means not declared."""
    model = _meta()
    model._no_split_modules = []
    assert resolve_no_split_classes(model) == []
    model._no_split_modules = None
    assert "_Block" in resolve_no_split_classes(model)


def test_a_failed_dynamic_resolution_keeps_the_hub_restrictions(monkeypatch):
    """Retrying via `from_config` after a failed lookup would hit the network under `local_files_only`, drop the token or change revision; the error must surface."""
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
    """`_split_units` emits the atomic ancestor, not the nested head name; a descendants-only filter left `pinned` empty and misplaced the logit headroom."""
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
    """Both `* logit_scale_multiply` and `/ logit_scale_divide` run out of place, so one extra chunk-shaped logit-dtype buffer is live before the float32 copy."""
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
    """The auto reserve is relaxed on non-head cards, so the plan must report a per-device reserve; one scalar would promise headroom the packing did not keep."""
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
    """`named_children()` drops repeated registrations, so aliases were never placement units while coverage still counted them, causing a false internal error."""
    with torch.device("meta"):
        model = _SharedLayerStack()
    assert len(model.layers) == 4
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    for name, _ in model.named_parameters(remove_duplicate = False):
        assert any(name == k or name.startswith(k + ".") for k in plan.device_map), name
    # One shared object, so every alias has to land on the same device.
    assert len({plan.device_map[f"layers.{i}"] for i in range(4)}) == 1


def test_infeasible_reports_each_devices_reserve():
    """The infeasible message must sum each device's reserve, not take the maximum times device count: 1 GiB and 3 GiB reserved is 4 GiB, not 6 GiB."""
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
    """`head_max` pushes weight off the head's card; plain device order filled the head's card first whenever the head was not the last device."""
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
    """fp16 checkpoints load `_keep_in_fp32_modules` tensors (T5's `wo`) as float32 even without a quantizer, so the meta sizing must honour them."""
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
    assert upcast["norm"] == plain["norm"] * 2
    assert upcast["layers"] == plain["layers"]
    assert upcast[""] == plain[""] + plain["norm"]


def test_fp32_exceptions_survive_a_transformers_that_only_takes_a_quantizer(monkeypatch):
    """Signature support must pick the order: transformers' compute_module_sizes takes only hf_quantizer, so using it
    unquantised dropped fp32 exceptions and undercharged T5 `wo`."""
    import types

    pytest.importorskip("accelerate")
    calls = []

    def quantizer_only(model, hf_quantizer = None, buffers_only = False, only_modules = True):
        """Newer transformers: no dtype, no special_dtypes."""
        calls.append("transformers")
        sizes = {}
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            parts = name.split(".")
            for i in range(len(parts) + 1):
                key = ".".join(parts[:i])
                sizes[key] = sizes.get(key, 0) + tensor.numel() * tensor.element_size()
        sizes.setdefault("", 0)
        return sizes, {}

    fake = types.ModuleType("transformers.integrations.accelerate")
    fake.compute_module_sizes = quantizer_only
    monkeypatch.setitem(sys.modules, "transformers.integrations.accelerate", fake)

    class Cfg:
        dtype = torch.float16

    with torch.device("meta"):
        model = _Tiny(hidden = 64, vocab = 512, layers = 4)
    model.half()
    model.config = Cfg()
    plain = _compute_module_sizes(model)
    model._keep_in_fp32_modules = ["norm"]
    upcast = _compute_module_sizes(model)

    assert calls, "the preferred implementation should still be tried first"
    assert upcast["norm"] == plain["norm"] * 2
    assert upcast["layers"] == plain["layers"]
def test_the_balanced_reserve_is_derived_per_head_candidate():
    """The balanced cap depends on which card pays the headroom; deriving it once against the smallest budget clamped the reserve wrongly and stopped balancing."""
    model = _meta(hidden = 512, vocab = 8192, layers = 8)
    plan = plan_device_map(
        model,
        max_memory = {0: 3 * _GiB, 1: 16 * _GiB},
        headroom_bytes = (5 * _GiB) // 2,
        prefer_head_device = 1,
    )
    assert plan.head_device == 1
    kept = plan.activation_reserve_by_device
    # cuda:0 holds no head, so its cap is 3 GiB less the weight share (2.98 GiB); the shared cap also charged it the 2.5 GiB headroom, leaving under 0.5 GiB.
    assert kept[0] > 1 * _GiB
    for device, reserve in kept.items():
        head_extra = plan.headroom_bytes if device == plan.head_device else 0
        assert plan.weight_bytes[device] + reserve + head_extra <= plan.raw_budgets[device]


def test_runtime_quantization_options_build_a_quantizer(monkeypatch):
    """`load_in_4bit=True` writes no `config.quantization_config` on full-precision checkpoints, so config-only sizing skipped the quantiser's budget adjustment."""
    kwargs = {"load_in_4bit": True, "trust_remote_code": False}
    qcfg = _runtime_quantization_config(kwargs)
    assert qcfg is not None and qcfg.load_in_4bit is True
    # Popped, so they never reach AutoConfig as stray attributes.
    assert kwargs == {"trust_remote_code": False}

    explicit = object()
    assert _runtime_quantization_config({"quantization_config": explicit}) is explicit
    assert _runtime_quantization_config({}) is None
    assert _runtime_quantization_config({"load_in_8bit": True}).load_in_8bit is True

    # `llm_int8_skip_modules` becomes `modules_to_not_convert`; ignoring it sizes whole modules at the quantised width the loader keeps at full precision.
    kwargs = {"load_in_8bit": True, "llm_int8_skip_modules": ["lm_head"],
              "trust_remote_code": True}
    qcfg = _runtime_quantization_config(kwargs)
    assert qcfg.llm_int8_skip_modules == ["lm_head"]
    assert kwargs.get("trust_remote_code") is True
    assert "llm_int8_skip_modules" not in kwargs

    kwargs = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4"}
    assert _runtime_quantization_config(kwargs).bnb_4bit_quant_type == "nf4"

    # The same contradiction `from_pretrained` refuses.
    with pytest.raises(ValueError, match = "not both"):
        _runtime_quantization_config({"load_in_4bit": True, "quantization_config": explicit})


def test_the_pretrained_helper_forwards_the_no_split_override(monkeypatch):
    """Passing the no-split override via `**config_kwargs` sends it to AutoConfig, so the plan silently used detected
    classes and refused a model the planner can place."""
    import unsloth_zoo.device_map_planner as planner

    with torch.device("meta"):
        model = _FatBlockModel()
    monkeypatch.setattr(planner, "build_meta_model", lambda *a, **k: (model, None, None))
    monkeypatch.setattr(planner, "_usable_devices", lambda max_memory: [0, 1])

    budgets = {0: 7 * _MiB, 1: 7 * _MiB}
    with pytest.raises(DeviceMapInfeasible):
        planner.plan_device_map_for_pretrained(
            "x", max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
        )
    plan = planner.plan_device_map_for_pretrained(
        "x", max_memory = budgets, headroom_bytes = 0, activation_reserve_bytes = 0,
        no_split_module_classes = [],
    )
    assert plan.no_split_module_classes == []


class _OneFatBlock(nn.Module):
    """A single atomic block that fits on no card except beside the head."""

    _no_split_modules = ["_Block"]

    def __init__(self, hidden = 1024):
        super().__init__()
        self.layers = nn.ModuleList([_Block(hidden)])
        self.lm_head = nn.Linear(hidden, 64, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def test_the_auto_reserve_relaxes_on_the_head_as_a_last_resort():
    """An auto reserve may relax on the head card as a last resort, since an atomic unit can miss it by less than the reserve; logit headroom is never relaxed."""
    with torch.device("meta"):
        model = _OneFatBlock()
    block = 1024 * 1024 * 4          # 4 MiB
    head = 1024 * 64 * 4             # 0.25 MiB
    headroom = 1 * _MiB
    plan = plan_device_map(
        model,
        max_memory = {0: block - 1, 1: block + head + headroom + (_MiB // 2)},
        headroom_bytes = headroom,
        prefer_head_device = 1,
    )
    assert plan.head_device == 1
    assert plan.device_map["layers.0"] == 1
    # The headroom survives in full; only the activation reserve gave way.
    assert plan.raw_budgets[1] - plan.weight_bytes[1] >= headroom


def _muse_shaped_budgets(model):
    """Solve B and headroom from the real sizes to trigger the Muse Glimmer bug: headroom > B - W/2, headroom < 2B - W, and B >= head + headroom."""
    sizes = {n: p.numel() * p.element_size()
             for n, p in model.named_parameters()}
    total = sum(sizes.values())
    head = sizes["lm_head.weight"]
    budget = 3 * total // 2
    # Inside (budget - total/2, budget - head], non-empty iff head < total/2.
    assert head < total // 2, "fixture no longer has a head under half the weights"
    headroom = (budget - total // 2 + budget - head) // 2
    assert budget - total // 2 < headroom <= budget - head
    assert headroom < 2 * budget - total
    return budget, headroom


def test_a_negative_cap_on_the_head_does_not_zero_the_other_cards():
    """Muse Glimmer shape on 2 x 14.56 GiB: a min cap zeroed both cards and OOMed cuda:0; a card that does not pay the headroom must keep its own reserve."""
    model = _meta(hidden = 256, vocab = 8192, layers = 16)
    budget, headroom = _muse_shaped_budgets(model)
    plan = plan_device_map(
        model,
        max_memory = {0: budget, 1: budget},
        headroom_bytes = headroom,
    )
    assert plan is not None
    reserves = plan.activation_reserve_by_device
    head_device = plan.device_map[
        next(n for n in plan.device_map if n.endswith("lm_head"))
    ]
    others = [d for d in reserves if d != head_device]
    assert others, "expected a non-head device to exist"
    assert any(reserves[d] > 0 for d in others), (
        f"every non-head card was given a zero activation reserve "
        f"({reserves}); the head's negative cap has leaked onto them again"
    )


def test_the_head_card_reserve_still_cannot_go_negative():
    """A negative reserve on the head's card would make every relaxation step infeasible and refuse the plan, so each device's reserve is clamped at zero."""
    model = _meta(hidden = 256, vocab = 8192, layers = 16)
    budget, headroom = _muse_shaped_budgets(model)
    plan = plan_device_map(
        model,
        max_memory = {0: budget, 1: budget},
        headroom_bytes = headroom,
    )
    assert plan is not None
    assert all(v >= 0 for v in plan.activation_reserve_by_device.values())


class _WideBlock(nn.Module):
    """Wide decoder block so packing has real granularity; the single-Linear _Block fits on the first greedy try and never exercises the relaxation ladder."""
    def __init__(self, hidden, ffn, dtype):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden, bias = False, dtype = dtype)
        self.mlp = nn.Linear(hidden, ffn, bias = False, dtype = dtype)
        self.down = nn.Linear(ffn, hidden, bias = False, dtype = dtype)


class _Wide(nn.Module):
    _no_split_modules = ["_WideBlock"]

    def __init__(self, hidden, ffn, vocab, layers, dtype = torch.bfloat16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden, dtype = dtype)
        self.layers = nn.ModuleList(
            [_WideBlock(hidden, ffn, dtype) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden, dtype = dtype)
        self.lm_head = nn.Linear(hidden, vocab, bias = False, dtype = dtype)

    def get_output_embeddings(self):
        return self.lm_head

    def get_input_embeddings(self):
        return self.embed_tokens


def test_the_relaxed_reserve_is_never_below_the_old_shared_cap():
    """Non-head reserves relax from the top of the range in 5% steps, which on identical cards can fall below the old
    shared cap; the relaxed reserve must stay at or above it."""
    kw = dict(hidden = 4096, ffn = 16384, vocab = 152064, layers = 24)
    with torch.device("meta"):
        model = _Wide(**kw)
    n, budget = 4, 6086993920

    plan = plan_device_map(model, max_memory = {d: budget for d in range(n)})
    assert plan is not None

    # What a single shared `min` cap across the devices would have produced.
    total = plan.total_weight_bytes
    headroom = plan.headroom_bytes
    head_bytes = model.lm_head.weight.numel() * model.lm_head.weight.element_size()
    value = max(0, (n * budget - total - headroom) // n)
    share = max(-(-total // n), head_bytes)
    shared_cap = max(0, min(value, budget - share - headroom))
    assert shared_cap > 0, "fixture no longer exercises the cap"

    kept = plan.activation_reserve_by_device
    assert min(kept.values()) >= shared_cap, (
        f"the relaxation ladder landed below the old shared cap: kept {kept}, "
        f"shared cap {shared_cap}"
    )


def test_the_head_relaxation_ladder_also_keeps_the_old_shared_floor():
    """The last-resort ladder that also relaxes the head's reserve must keep the old shared floor too; scaling from the top alone can undercut the shared cap."""
    with torch.device("meta"):
        model = _Bins([5, 100], head = 20)
    plan = plan_device_map(
        model,
        max_memory = {0: 300, 1: 590},
        headroom_bytes = 22,
    )
    assert plan is not None
    assert plan.weight_bytes == {0: 20, 1: 480}
    kept = plan.activation_reserve_by_device
    assert min(kept.values()) >= 50, (
        f"the last-resort ladder landed below the old shared cap of 50: {kept}"
    )


def _held_bytes(model, plan):
    """Weights each device really holds, walked from the map, not the plan."""
    per = {d: 0 for d in plan.raw_budgets}
    for name, device in plan.device_map.items():
        module = model.get_submodule(name) if name else model
        per[device] += sum(
            p.numel() * p.element_size() for p in module.parameters(recurse = True)
        )
    return per


def test_the_smaller_card_of_an_asymmetric_pair_is_not_packed_to_zero():
    """Unequal cards: the flat average weight exceeds what a small card holds, zeroing its reserve; a card not paying the headroom must reserve for its own weight."""
    with torch.device("meta"):
        model = _Wide(hidden = 8192, ffn = 8192, vocab = 128000, layers = 110)

    for small, big in [(16, 80), (24, 48), (40, 80)]:
        max_memory = {0: small * _GiB, 1: big * _GiB}
        plan = plan_device_map(model, max_memory = max_memory)
        assert plan is not None, f"{small}+{big} GiB was refused"

        kept = plan.activation_reserve_by_device
        assert kept[0] > 0, (
            f"{small}+{big} GiB: the smaller card was given a zero activation "
            f"reserve ({kept}); it is being charged the flat average weight again"
        )

        # Reserving the small card's WHOLE budget leaves it empty, not a plan.
        assert plan.weight_bytes[0] > 0, (
            f"{small}+{big} GiB: the smaller card was left holding nothing "
            f"({plan.weight_bytes})"
        )

        # The reported free space is the truth of the packing, not the ask.
        held = _held_bytes(model, plan)
        assert held == plan.weight_bytes, f"{small}+{big} GiB: {held} vs {plan.weight_bytes}"
        free = plan.free_bytes
        assert free[0] >= kept[0], (
            f"{small}+{big} GiB: kept {kept[0]} but only {free[0]} is free"
        )

        small_frac = free[0] / max_memory[0]
        big_frac = free[1] / max_memory[1]
        assert small_frac >= big_frac / 2, (
            f"{small}+{big} GiB: the small card has {100 * small_frac:.1f}% free "
            f"against {100 * big_frac:.1f}% on the big one; free {free}"
        )


def test_identical_cards_still_get_the_flat_average_share():
    """Proration must be an exact no-op on identical cards (Muse Glimmer arithmetic); both sides need ceiling division,
    since a floored `total * b // (n * b)` drifts off `-(-total // n)`."""
    with torch.device("meta"):
        model = _Wide(hidden = 2048, ffn = 8192, vocab = 32768, layers = 20)
    budget = int(14.56 * _GiB)

    head_bytes = (
        model.lm_head.weight.numel() * model.lm_head.weight.element_size()
    )
    for n in (2, 3, 4, 5):
        plan = plan_device_map(model, max_memory = {d: budget for d in range(n)})
        assert plan is not None, f"{n} x 14.56 GiB was refused"

        # Pre-proration: one flat average share, pinned head as its floor.
        total = plan.total_weight_bytes
        headroom = plan.headroom_bytes
        value = max(0, (n * budget - total - headroom) // n)
        share = max(-(-total // n), head_bytes)
        old = {
            d: int(max(0, min(
                value,
                budget - share - (headroom if d == plan.head_device else 0),
            )))
            for d in range(n)
        }
        assert value > 0 and old[plan.head_device] < value, (
            f"{n} cards: fixture no longer exercises the head's cap"
        )
        assert plan.activation_reserve_by_device == old, (
            f"{n} x 14.56 GiB moved: got {plan.activation_reserve_by_device}, "
            f"the flat-average formula gives {old}"
        )


def _flat_average_ask(plan, pinned_bytes):
    """What the pre-proration planner asked every device to keep."""
    n = len(plan.raw_budgets)
    total, headroom = plan.total_weight_bytes, plan.headroom_bytes
    value = max(0, (sum(plan.raw_budgets.values()) - total - headroom) // n)
    share = max(-(-total // n), pinned_bytes)
    return {
        d: int(max(0, min(
            value,
            budget - share - (headroom if d == plan.head_device else 0),
        )))
        for d, budget in plan.raw_budgets.items()
    }


def test_prorating_never_charges_the_head_card_more_than_the_flat_average():
    """The head sits on the bigger card, which proration would charge most; the prorated share is only taken when
    smaller than the flat average, so no card loses reserve versus the old planner."""
    with torch.device("meta"):
        model = _Wide(hidden = 8192, ffn = 8192, vocab = 128000, layers = 4)
    pinned = model.lm_head.weight.numel() * model.lm_head.weight.element_size()

    for small, big, hr in [(4, 12, 8), (10, 12, 2)]:
        plan = plan_device_map(
            model,
            max_memory = {0: small * _GiB, 1: big * _GiB},
            headroom_bytes = hr * _GiB,
        )
        assert plan is not None, f"{small} + {big} GiB was refused"

        flat = _flat_average_ask(plan, pinned)
        head = plan.head_device
        assert plan.raw_budgets[head] > min(plan.raw_budgets.values()), (
            f"{small} + {big} GiB: the head is not on the bigger card any more"
        )
        assert flat[head] > 0, "fixture no longer exercises the head's cap"

        kept = plan.activation_reserve_by_device
        assert kept[head] > 0, (
            f"{small} + {big} GiB: the head card was given a zero activation "
            f"reserve ({kept}); it is being charged its full proportional "
            f"share AND the logit headroom"
        )
        assert kept[head] >= flat[head], (
            f"{small} + {big} GiB: the head kept {kept[head]} where the "
            f"flat-average planner kept {flat[head]}; proration must only "
            f"loosen the cap, never tighten it"
        )
        assert all(v > 0 for v in kept.values()), kept


def test_the_relaxation_ladder_still_offers_the_flat_average_rungs():
    """Relaxation rungs are 5% of the mapping they scale from, so a prorated start can step past a flat-average rung
    that fits; walk both ladders to keep every old rung."""
    with torch.device("meta"):
        model = _Bins([68, 192], head = 82)
    plan = plan_device_map(
        model,
        max_memory = {0: 1264, 1: 1354},
        headroom_bytes = 3,
    )
    assert plan is not None
    assert plan.total_weight_bytes == 1368
    assert plan.weight_bytes == {0: 768, 1: 600}
    assert plan.activation_reserve_by_device == {0: 493, 1: 623}, (
        f"the prorated ladder skipped the flat-average rung: "
        f"{plan.activation_reserve_by_device}"
    )


def test_a_merged_ladder_rung_never_undercuts_the_flat_average_plan():
    """Ordering candidates by the smallest reserve kept is not monotone; pooling prorated and flat-average ladders can
    leave a card with less than the flat-average planner kept."""
    with torch.device("meta"):
        model = _Bins([249, 230, 144, 203, 223, 68], head = 32)
    plan = plan_device_map(
        model,
        max_memory = {0: 3050, 1: 3977},
        headroom_bytes = 504,
    )
    assert plan is not None
    assert plan.head_device == 1
    assert plan.total_weight_bytes == 4596
    legacy = {0: 752, 1: 963}
    kept = plan.activation_reserve_by_device
    assert all(kept[d] >= legacy[d] for d in legacy), (
        f"a merged rung kept less than the flat-average planner {legacy}: {kept}"
    )
    assert kept == {0: 866, 1: 963}, kept


def test_a_small_card_is_not_charged_the_pinned_head_it_never_holds():
    """The pinned head's weight is charged only to the head card; charging every card zeroes the small card's reserve and packs it full."""
    with torch.device("meta"):
        model = _Bins([100, 50], head = 150)
    plan = plan_device_map(
        model,
        max_memory = {0: 400, 1: 2000},
        headroom_bytes = 0,
    )
    assert plan is not None
    assert plan.head_device == 1
    assert plan.total_weight_bytes == 1200
    kept = plan.activation_reserve_by_device
    assert kept[0] > 0, (
        f"cuda:0 was charged the pinned head it does not hold: {kept}"
    )
    assert kept == {0: 200, 1: 600}, kept
    free = {d: plan.raw_budgets[d] - plan.weight_bytes.get(d, 0) for d in (0, 1)}
    assert free[0] >= kept[0], f"cuda:0 was packed below its own reserve: {free}"


class _Cfg:
    """A model config stands in as a plain attribute holder."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


class _MethodOnlyTextConfig:
    """The T5Gemma shape: no `text_config` attribute, so a reader checking only the top level and `.text_config` sees no soft cap."""

    def __init__(self, inner):
        self._inner = inner

    def get_text_config(self):
        return self._inner


@pytest.mark.parametrize(
    "config, expected",
    [
        # Gemma 2/3/3n, T5Gemma, VaultGemma.
        (_Cfg(final_logit_softcapping = 30.0), (30.0, 0.0, 0.0)),
        # RecurrentGemma spells the same knob differently.
        (_Cfg(logits_soft_cap = 30.0), (30.0, 0.0, 0.0)),
        # Cohere and Cohere 2 multiply.
        (_Cfg(logit_scale = 0.0625), (0.0, 0.0625, 0.0)),
        # Granite and its MoE variants divide.
        (_Cfg(logits_scaling = 16.0), (0.0, 0.0, 16.0)),
        # Falcon-H1 multiplies, on the lm_head call line itself.
        (_Cfg(lm_head_multiplier = 4.0), (0.0, 4.0, 0.0)),
        # Llama and friends transform nothing.
        (_Cfg(), (0.0, 0.0, 0.0)),
    ],
)
def test_detects_the_transform_each_family_applies(config, expected):
    found = detect_logit_transforms(config)
    assert (
        found["logit_softcapping"],
        found["logit_scale_multiply"],
        found["logit_scale_divide"],
    ) == expected


def test_reads_a_nested_text_config():
    """Gemma 3 puts the soft cap on the text sub-config, not the top level."""
    config = _Cfg(text_config = _Cfg(final_logit_softcapping = 20.0))
    assert detect_logit_transforms(config)["logit_softcapping"] == 20.0


def test_reads_a_text_config_only_reachable_by_method():
    """Reading `.text_config` alone reports no soft cap, under-reserving the head's card by the tanh temporary plus the retained term."""
    config = _MethodOnlyTextConfig(_Cfg(final_logit_softcapping = 30.0))
    assert detect_logit_transforms(config)["logit_softcapping"] == 30.0


def test_a_contrastive_logit_scale_is_not_an_output_head_transform():
    """CLIP's `logit_scale` (spelled `logit_scale_init_value` in its config) scales image-text similarity, not an output head."""
    from transformers.models.clip import CLIPConfig

    found = detect_logit_transforms(CLIPConfig())
    assert found == {
        "logit_softcapping": 0.0,
        "logit_scale_multiply": 0.0,
        "logit_scale_divide": 0.0,
    }


def test_detection_never_raises():
    """An unreadable config reports no transforms rather than raising, leaving the caller on its pre-detection behaviour."""
    class _Hostile:
        def __getattr__(self, name):
            raise ValueError(name)

    for config in (None, object(), _Hostile(), _MethodOnlyTextConfig(None)):
        assert detect_logit_transforms(config) == {
            "logit_softcapping": 0.0,
            "logit_scale_multiply": 0.0,
            "logit_scale_divide": 0.0,
        }


def test_the_planner_sizes_a_detected_soft_cap_without_being_told():
    """The point of the detection: the same model plans the same either way."""
    model = _meta(vocab = 512)
    model.config = _Cfg(final_logit_softcapping = 30.0)
    auto = plan_device_map(model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB})
    told = plan_device_map(
        model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB}, softcapped = True,
    )
    assert auto is not None and told is not None
    assert auto.headroom_bytes == told.headroom_bytes


def test_the_planner_sizes_a_detected_logit_scale_without_being_told():
    model = _meta(vocab = 512)
    model.config = _Cfg(logits_scaling = 16.0)
    auto = plan_device_map(model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB})
    told = plan_device_map(
        model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB}, logit_scaled = True,
    )
    assert auto is not None and told is not None
    assert auto.headroom_bytes == told.headroom_bytes
    # And bigger than the un-scaled reserve, or the flag is inert.
    plain = plan_device_map(
        model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB}, logit_scaled = False,
    )
    assert auto.headroom_bytes > plain.headroom_bytes


@pytest.mark.parametrize("flag", [True, False])
def test_an_explicit_flag_still_wins_over_detection(flag):
    """Backwards compatible: only `None` asks for detection."""
    model = _meta(vocab = 512)
    model.config = _Cfg(final_logit_softcapping = 30.0, logits_scaling = 16.0)
    told = plan_device_map(
        model,
        max_memory = {0: 8 * _GiB, 1: 8 * _GiB},
        softcapped = flag,
        logit_scaled = flag,
    )
    expected = logit_headroom_bytes(
        512, 128, logit_dtype = model.lm_head.weight.dtype,
        softcapped = flag, logit_scaled = flag,
    )
    assert told.headroom_bytes == expected


@pytest.mark.parametrize(
    "error", [RuntimeError, KeyError, OSError, ImportError, RecursionError],
)
def test_a_config_that_raises_anything_does_not_abort_planning(error):
    """`getattr` with a default only swallows `AttributeError`, so a remote-code config raising any other error must not abort planning."""
    class _Attribute:
        def __getattr__(self, name):
            raise error(name)

    class _Property:
        @property
        def text_config(self):
            raise error("text_config")

    class _Method:
        text_config = None
        def get_text_config(self):
            raise error("get_text_config")

    for config in (_Attribute(), _Property(), _Method()):
        assert detect_logit_transforms(config) == {
            "logit_softcapping": 0.0,
            "logit_scale_multiply": 0.0,
            "logit_scale_divide": 0.0,
        }

    model = _meta(vocab = 512)
    model.config = _Property()
    plan = plan_device_map(model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB})
    assert plan is not None


def test_an_interrupt_is_not_swallowed():
    """Detection absorbs config errors, not the user's Ctrl-C."""
    class _Interrupting:
        def __getattr__(self, name):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        detect_logit_transforms(_Interrupting())


def test_one_unreadable_field_does_not_discard_the_readable_ones():
    """A non-numeric field is skipped alone; zeroing the whole result would drop a soft cap the old flag-free path did detect."""
    config = _Cfg(final_logit_softcapping = 30.0, logit_scale = ["not a number"])
    found = detect_logit_transforms(config)
    assert found["logit_softcapping"] == 30.0
    assert found["logit_scale_multiply"] == 0.0


def test_a_non_numeric_alias_falls_through_to_the_next_one():
    config = _Cfg(final_logit_softcapping = object(), logits_soft_cap = 20.0)
    assert detect_logit_transforms(config)["logit_softcapping"] == 20.0


@pytest.mark.parametrize(
    "junk",
    [
        ["not a number"],                    # TypeError
        "twenty",                            # ValueError
        10 ** 400,                           # OverflowError, not a ValueError
        torch.tensor([1.0, 2.0]),            # ValueError from torch
    ],
)
def test_no_conversion_error_discards_the_fields_already_read(junk):
    config = _Cfg(final_logit_softcapping = 30.0, logit_scale = junk)
    found = detect_logit_transforms(config)
    assert found["logit_softcapping"] == 30.0
    assert found["logit_scale_multiply"] == 0.0


@pytest.mark.parametrize("value", [0.0, 0, -0.0, False])
def test_a_zero_scale_reserves_nothing(value):
    """The chunked loss guards each transform with `!= 0.0` (rl_replacements.py), so a zero scale allocates no copy; the `bool` cast keeps planner and loss in step."""
    model = _meta(vocab = 512)
    model.config = _Cfg(logit_scale = value)
    auto = plan_device_map(model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB})
    off = plan_device_map(
        model, max_memory = {0: 8 * _GiB, 1: 8 * _GiB}, logit_scaled = False,
    )
    assert auto.headroom_bytes == off.headroom_bytes


def test_the_xlstm_spelling_of_the_soft_cap():
    """xLSTM's `output_logit_soft_cap` (default 30.0) is applied unguarded in modeling_xlstm.py, so NX-AI/xLSTM-7b was under-reserved; the name is xLSTM-only."""
    assert detect_logit_transforms(
        _Cfg(output_logit_soft_cap = 30.0))["logit_softcapping"] == 30.0


def test_reads_a_decoder_sub_config():
    """T5Gemma keeps the cap on `config.decoder`; on transformers 4.56.x get_text_config returns self, so only following `decoder` directly reaches it."""
    class _SelfReturning:
        def __init__(self, decoder):
            self.decoder = decoder
        def get_text_config(self, *args, **kwargs):
            return self

    config = _SelfReturning(_Cfg(final_logit_softcapping = 30.0))
    assert detect_logit_transforms(config)["logit_softcapping"] == 30.0


def test_a_decoder_that_transforms_nothing_stays_inert():
    """Following `decoder` must not invent a transform: an encoder-decoder whose decoder transforms nothing stays inert."""
    config = _Cfg(decoder = _Cfg(vocab_size = 32000), text_encoder = _Cfg())
    assert detect_logit_transforms(config) == {
        "logit_softcapping": 0.0,
        "logit_scale_multiply": 0.0,
        "logit_scale_divide": 0.0,
    }


def test_a_sub_config_left_as_a_dict_is_still_read():
    """A composite declaring no sub-config type keeps text_config as a raw dict, where getattr never sees the keys; the dict's keys must still be read."""
    config = _Cfg(text_config = {"final_logit_softcapping": 30.0})
    assert detect_logit_transforms(config)["logit_softcapping"] == 30.0


@pytest.mark.parametrize("model_type", ["aya_vision", "cohere2_vision"])
def test_a_wrapper_that_re_heads_its_text_tower_claims_no_transform(model_type):
    """Aya Vision and Cohere 2 Vision use their own unscaled nn.Linear lm_head and logit_scale is absent from their
    modeling files; the Cohere 2 config declares it, but crediting it reserves nothing real, so no transform is
    claimed."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config = CONFIG_MAPPING[model_type]()
    assert getattr(config.text_config, "logit_scale", None)   # the trap is real
    assert detect_logit_transforms(config)["logit_scale_multiply"] == 0.0


def test_a_wrapper_that_reuses_the_causal_lm_head_keeps_its_transform():
    """Granite Speech builds an AutoModelForCausalLM, so the logits_scaling divide happens inside the text model and the
    reserve is still owed; the name is absent from its file."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config = CONFIG_MAPPING["granite_speech"]()
    assert detect_logit_transforms(config)["logit_scale_divide"] == \
        config.text_config.logits_scaling


@pytest.mark.parametrize("wrapper", ["module", "_orig_mod"])
def test_a_wrapped_model_still_reports_its_transforms(wrapper):
    """`nn.Module.__getattr__` resolves submodules only, so DDP and torch.compile wrappers lack `.config`; the inner model would otherwise read as transform-free."""
    inner = nn.Module()
    inner.config = _Cfg(final_logit_softcapping = 30.0)
    outer = nn.Module()
    setattr(outer, wrapper, inner)
    assert detect_logit_transforms(outer)["logit_softcapping"] == 30.0


def test_the_muse_glimmer_multiplier():
    """Muse Glimmer pre-scales by `output_multiplier` before its soft cap, so it owes both buffers; the multiplier is
    applied in the composite's own forward and is named in no other config."""
    config = _Cfg(text_config = _Cfg(output_multiplier = 0.19611613513818404,
                                     final_logit_softcapping = 20.0))
    found = detect_logit_transforms(config)
    assert found["logit_scale_multiply"] == 0.19611613513818404
    assert found["logit_softcapping"] == 20.0


def test_logits_scaling_multiplies_for_hyperclovax_and_divides_for_granite():
    """logits_scaling multiplies for HyperCLOVA X but divides for Granite (per transformers' MuP comment); bucketing
    HyperCLOVA X as a divide reports the wrong magnitude."""
    granite = _Cfg(model_type = "granite", logits_scaling = 8.0)
    assert detect_logit_transforms(granite)["logit_scale_divide"] == 8.0
    assert detect_logit_transforms(granite)["logit_scale_multiply"] == 0.0

    clova = _Cfg(model_type = "hyperclovax", logits_scaling = 8.0)
    assert detect_logit_transforms(clova)["logit_scale_multiply"] == 8.0
    assert detect_logit_transforms(clova)["logit_scale_divide"] == 0.0


@pytest.mark.parametrize("tied", [True, False])
def test_falcon_h1_multiplies_whether_or_not_the_head_is_tied(tied):
    """FalconH1ForCausalLM multiplies the head unconditionally, with no tie_word_embeddings branch, so gating on tied
    drops the reserve; the mlx/utils.py tied gate is a different contract."""
    config = _Cfg(model_type = "falcon_h1", lm_head_multiplier = 4.0,
                  tie_word_embeddings = tied)
    assert detect_logit_transforms(config)["logit_scale_multiply"] == 4.0


def test_minicpm3_logits_scaling_is_not_a_logit_transform():
    """MiniCPM3's logits_scaling property divides the hidden states before the head, so no logits temporary exists and nothing is reserved."""
    config = _Cfg(model_type = "minicpm3", logits_scaling = 10.0)
    assert detect_logit_transforms(config) == {
        "logit_softcapping": 0.0,
        "logit_scale_multiply": 0.0,
        "logit_scale_divide": 0.0,
    }


def test_a_config_that_hands_back_itself_terminates():
    class _Circular:
        final_logit_softcapping = 20.0
        @property
        def text_config(self):
            return self

    assert detect_logit_transforms(_Circular())["logit_softcapping"] == 20.0


class _MTPLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias = False)
        self.mlp = nn.Linear(hidden, hidden, bias = False)


def test_a_sibling_of_a_declared_block_is_atomic_too():
    model = _meta(layers = 8)
    with torch.device("meta"):
        model.layers.append(_MTPLayer(64))
    assert resolve_no_split_classes(model) == ["_Block", "_MTPLayer"]
    plan = plan_device_map(model, max_memory = {0: "8GiB", 1: "8GiB"})
    assert plan is not None
    assert "layers.8" in plan.device_map
    assert not any(k.startswith("layers.8.") for k in plan.device_map)


def test_declared_classes_outside_a_module_list_are_kept_as_declared():
    model = _meta(layers = 4)
    model._no_split_modules = ["_Block", "LayerNorm"]
    assert resolve_no_split_classes(model) == ["LayerNorm", "_Block"]
    model._no_split_modules = ["LayerNorm"]
    assert resolve_no_split_classes(model) == ["LayerNorm"]


def _swap_sizes():
    model = _meta(layers = 8)
    sizes = _compute_module_sizes(model)
    return model, sizes[""], sizes["layers.0"]


def test_exclude_modules_leaves_layers_out_of_the_map():
    model, total, layer = _swap_sizes()
    plan = plan_device_map(
        model,
        max_memory = {0: total, 1: total},
        headroom_bytes = 0,
        activation_reserve_bytes = 0,
        exclude_modules = ["layers.6", "layers.7"],
    )
    assert not any(k.startswith(("layers.6", "layers.7")) for k in plan.device_map)
    assert plan.total_weight_bytes == total - 2 * layer
    with pytest.raises(ValueError, match = "larger placement unit"):
        plan_device_map(model, max_memory = {0: total, 1: total}, exclude_modules = ["layers.6.mlp"])


def test_block_swap_plan_one_device_is_the_fewest_layers_that_fit():
    from unsloth_zoo.device_map_planner import plan_block_swap
    model, total, layer = _swap_sizes()
    reserve = 3 * layer
    assert plan_block_swap(model = model, max_memory = {0: total + reserve}, reserve_bytes = reserve).layers == 0
    for short in (1, layer, 2 * layer + 1):
        plan = plan_block_swap(model = model, max_memory = {0: total + reserve - short}, reserve_bytes = reserve)
        n = plan.layers
        # Swapping n layers frees n of them but the depth + 1 slot pool keeps 3 on the card.
        assert (n - 3) * layer >= short and (n - 4) * layer < short, (short, n)


def test_block_swap_plan_multi_gpu_charges_the_pool_to_the_head_card():
    from unsloth_zoo.device_map_planner import plan_block_swap
    model, total, layer = _swap_sizes()
    embed = 512 * 64 * 4
    budget = embed + 600 + 5 * layer  # room for ten resident layers across both cards
    plan = plan_block_swap(model = model, max_memory = {0: budget, 1: budget}, reserve_bytes = 0,
                           headroom_bytes = 0)
    assert plan.layers == 0
    budget = embed + 600 + 3 * layer  # room for six layers, three of them taken by the pool
    plan = plan_block_swap(model = model, max_memory = {0: budget, 1: budget}, reserve_bytes = 0,
                           headroom_bytes = 0)
    resident = [k for k in plan.device_plan.device_map if k.startswith("layers.")]
    assert plan.layers == 5 and len(resident) == 3
    head = plan.device_plan.head_device
    assert plan.device_plan.weight_bytes[head] + 3 * layer <= budget


class _NarrowBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden // 2, bias = False)


def test_block_swap_plan_mixed_layers_finds_the_fewest_count():
    # Spread picks are not nested: these 7 alternating layers free a pool's worth at 4 swapped, nothing at 5.
    from unsloth_zoo.device_map_planner import plan_block_swap
    model = _meta(layers = 7)
    with torch.device("meta"):
        for i in range(1, 7, 2):
            model.layers[i] = _NarrowBlock(64)
    total = _compute_module_sizes(model)[""]
    plan = plan_block_swap(model = model, max_memory = {0: total + 100 - 1}, reserve_bytes = 100,
                           placement = "spread")
    assert plan.layers == 4


class _HalvingQuantizer:
    def adjust_max_memory(self, max_memory):
        return {k: v // 2 for k, v in max_memory.items()}


def test_block_swap_plan_applies_the_quantizer_haircut_once(monkeypatch):
    from unsloth_zoo import device_map_planner as dmp
    model, total, layer = _swap_sizes()
    sizes = dmp._compute_module_sizes(model)
    # Size the model the same with and without the quantizer: only the budget haircut differs.
    monkeypatch.setattr(dmp, "_compute_module_sizes", lambda model, hf_quantizer = None: sizes)
    budget = 512 * 64 * 4 + 600 + 3 * layer
    kw = dict(model = model, reserve_bytes = 0, headroom_bytes = 0)
    plain = dmp.plan_block_swap(max_memory = {0: budget, 1: budget}, **kw)
    halved = dmp.plan_block_swap(max_memory = {0: 2 * budget, 1: 2 * budget}, hf_quantizer = _HalvingQuantizer(), **kw)
    assert plain.layers > 0 and halved.layers == plain.layers


def test_block_swap_plan_counts_no_savings_for_a_block_listed_twice():
    # One block object at every index: swapping any of them frees nothing, so no count fits.
    from unsloth_zoo.device_map_planner import plan_block_swap
    with torch.device("meta"):
        model = _Tiny(layers = 1)
        block = model.layers[0]
        model.layers = nn.ModuleList([block] * 16)
    total = _compute_module_sizes(model)[""]
    with pytest.raises(DeviceMapInfeasible):
        plan_block_swap(model = model, max_memory = {0: total - 1}, reserve_bytes = 0)


def test_block_swap_plan_skips_a_card_too_small_for_the_pool():
    from unsloth_zoo.device_map_planner import plan_block_swap
    model, total, layer = _swap_sizes()
    # Three cards; the last (tried first for the pool) cannot hold three slots, so the next one takes it.
    half = (total - layer) // 2
    plan = plan_block_swap(model = model, max_memory = {0: half, 1: half, 2: 2 * layer - 1},
                           reserve_bytes = 0, headroom_bytes = 0)
    assert plan.layers == 4 and plan.device_plan.head_device == 1


def test_block_swap_plan_refuses_when_one_layer_cannot_stay():
    from unsloth_zoo.device_map_planner import plan_block_swap
    model, total, layer = _swap_sizes()
    with pytest.raises(DeviceMapInfeasible, match = "host RAM"):
        plan_block_swap(model = model, max_memory = {0: layer}, reserve_bytes = 0)


def test_block_swap_plan_moves_the_embedding_before_any_layer():
    from unsloth_zoo.device_map_planner import plan_block_swap
    model, total, layer = _swap_sizes()
    embed = 512 * 64 * 4
    reserve = embed
    # Short by less than the embedding: moving it is enough, no layer swaps.
    plan = plan_block_swap(model = model, max_memory = {0: total + reserve - embed // 2},
                           reserve_bytes = reserve, offload_embedding = True)
    assert plan.offload_embedding and plan.layers == 0 and plan.embedding_bytes == embed
    # Not allowed: layers have to go instead.
    plan = plan_block_swap(model = model, max_memory = {0: total + reserve - embed // 2},
                           reserve_bytes = reserve)
    assert not plan.offload_embedding and plan.layers > 0
    # Short by more: the embedding moves and only the rest is swapped.
    short = embed + 2 * layer
    plan = plan_block_swap(model = model, max_memory = {0: total + reserve - short},
                           reserve_bytes = reserve, offload_embedding = True)
    assert plan.offload_embedding and (plan.layers - 3) * layer >= 2 * layer > (plan.layers - 4) * layer


def test_block_swap_plan_fits_the_pool_before_the_embedding_moves():
    # Short sequences: the embedding outweighs the reserve, so attach (pool + embedding on the card) is the peak.
    from unsloth_zoo.device_map_planner import plan_block_swap
    model = _meta(layers = 16, vocab = 64)
    sizes = _compute_module_sizes(model)
    total, layer, embed = sizes[""], sizes["layers.0"], sizes["embed_tokens"]
    budget = total - embed - 2 * layer
    plan = plan_block_swap(model = model, max_memory = {0: budget}, reserve_bytes = 0, offload_embedding = True)
    assert plan.offload_embedding
    assert total - plan.layers * layer + 3 * layer <= budget


class _CachedBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden, bias = False)
        self.register_buffer("cache", torch.zeros(4 * hidden, hidden), persistent = False)


def test_block_swap_plan_leaves_layer_buffers_resident():
    from unsloth_zoo.device_map_planner import plan_block_swap
    model = _meta(layers = 16)
    with torch.device("meta"):
        for i in range(16):
            model.layers[i] = _CachedBlock(64)
    sizes = _compute_module_sizes(model)
    total, weight = sizes[""], sizes["layers.0.mlp"]
    budget = total - 2 * weight
    plan = plan_block_swap(model = model, max_memory = {0: budget}, reserve_bytes = 0)
    # Only weights leave the card: two freed beyond the three-slot pool.
    assert plan.layers == 5


def test_block_swap_plan_keeps_a_tied_embedding():
    from unsloth_zoo.device_map_planner import plan_block_swap
    model = _meta(layers = 8, tie = True)
    sizes = _compute_module_sizes(model)
    plan = plan_block_swap(model = model, max_memory = {0: sizes[""] + 100 - 1}, reserve_bytes = 100,
                           offload_embedding = True)
    assert not plan.offload_embedding and plan.layers > 0


def test_block_swap_plan_streams_extra_token_tables_of_a_tied_model(monkeypatch):
    # Gemma 3n / 4: tied input embedding, but the per-layer table streams to host, freeing its room for training.
    from unsloth_zoo import block_swap
    from unsloth_zoo.device_map_planner import plan_block_swap
    model = _meta(layers = 8, tie = True)
    model.per_layer = nn.Embedding(512, 64, device = "meta")
    extra = 512 * 64 * 4
    monkeypatch.setattr(block_swap, "EXTRA_EMBEDDING_MIN_BYTES", extra)
    total = _compute_module_sizes(model)[""]
    budget = total + extra - extra // 2
    plan = plan_block_swap(model = model, max_memory = {0: budget}, reserve_bytes = extra, offload_embedding = True)
    assert plan.offload_embedding and plan.layers == 0 and plan.embedding_bytes == extra
    plan = plan_block_swap(model = model, max_memory = {0: budget}, reserve_bytes = extra)
    assert not plan.offload_embedding and plan.layers > 0
    # The table lands on the card whole before it moves: the load still needs room for it.
    plan = plan_block_swap(model = model, max_memory = {0: total + 100 - extra // 2}, reserve_bytes = 100,
                           offload_embedding = True)
    assert plan.layers > 0
