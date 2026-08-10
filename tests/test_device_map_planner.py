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
import pytest
import torch
import torch.nn as nn

from unsloth_zoo.device_map_planner import (
    DeviceMapInfeasible,
    _auto_class_for,
    _compute_module_sizes,
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
        modules_to_not_convert = ["lm_head"]
        def adjust_target_dtype(self, dtype):
            return CustomDtype.INT4
        def get_special_dtypes_update(self, model, dtype):
            return {n: dtype for n, _ in model.named_parameters()
                    if any(m in n for m in self.modules_to_not_convert)}

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
