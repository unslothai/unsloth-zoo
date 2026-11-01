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

"""One compiled decoder-layer graph serves every layer and sequence length, capped at 2 Dynamo frames; per-layer
recompiles came from an id() guard on expert params, a lazy global, and router_logits length."""

import pytest
import torch
from torch._dynamo.testing import CompileCounter

LENGTHS = [5, 9, 12, 17, 23, 31, 40, 47, 58, 64]


class _Experts(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        # Square stacks: the orientation is read from the sibling's logical shape.
        self.gate_up_proj = torch.nn.Parameter(torch.randn(4, hidden, hidden))
        self.down_proj = torch.nn.Parameter(torch.randn(4, hidden, hidden))


def _count(fn, modules):
    torch._dynamo.reset()
    cc = CompileCounter()
    compiled = torch.compile(fn, backend = cc, dynamic = None)
    for module, n in zip(modules, LENGTHS):
        compiled(module, torch.randn(2, n, 8))
    return cc.frame_count


def test_logical_expert_shape_does_not_guard_on_parameter_identity():
    from unsloth_zoo.temporary_patches.moe_utils import _logical_expert_shape

    def layer(experts, x):
        shape = _logical_expert_shape(experts.down_proj)
        return x * shape[0]

    frames = _count(layer, [_Experts(8) for _ in LENGTHS])
    assert frames <= 2, f"{frames} graphs for 10 layers: guarding on a parameter's identity"


def test_logical_expert_shape_still_unwraps():
    from unsloth_zoo.temporary_patches.moe_utils import _logical_expert_shape

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.base_layer = inner

    p = torch.nn.Parameter(torch.zeros(3, 5, 7))
    lin = torch.nn.Linear(4, 6)
    assert _logical_expert_shape(p) == (3, 5, 7)
    assert _logical_expert_shape(Wrapper(Wrapper(lin))) == (6, 4)
    p._original_shape = (2, 2)
    assert _logical_expert_shape(p) == (2, 2)
    loop = Wrapper(lin)
    loop.base_layer = loop  # self reference ends the walk
    assert _logical_expert_shape(loop) is None or isinstance(_logical_expert_shape(loop), tuple)


def test_weighted_unpermute_is_resolved_at_import():
    from unsloth_zoo.temporary_patches import moe_utils
    # A None here is assigned on the first traced call: the guard on it recompiles every layer once.
    assert moe_utils._WEIGHTED_UNPERMUTE is not None


output_capturing = pytest.importorskip("transformers.utils.output_capturing")


def _collector():
    from unsloth_zoo.temporary_patches.misc import patch_output_collector_for_compiled_submodules
    patch_output_collector_for_compiled_submodules()
    return output_capturing._active_collector


class _Layer(torch.nn.Module):
    def __init__(self, var):
        super().__init__()
        self.router = torch.nn.Linear(8, 4)
        self.proj = torch.nn.Linear(8, 8)

        def hook(module, args, output):  # transformers' capture hook, minus the hidden_states branches
            collected = var.get()
            if collected is None or "router_logits" not in collected.keys():
                return
            collected["router_logits"].append(output)

        self.router.register_forward_hook(hook)

    def forward(self, x):
        return self.proj(x) + self.router(x).sum(-1, keepdim = True)


def _run_layers(scratch):
    import contextlib
    var = _collector()
    layers = [_Layer(var) for _ in LENGTHS]
    torch._dynamo.reset()
    cc = CompileCounter()
    compiled = torch.compile(_Layer.forward, backend = cc, dynamic = None)
    xs = [torch.randn(2, n, 8) for n in LENGTHS]
    with torch.no_grad():
        want = [layer.router(x) for layer, x in zip(layers, xs)]  # no collector active yet
    collected = {"router_logits": []}
    token = var.set(collected)
    try:
        for layer, x in zip(layers, xs):
            with (scratch() if scratch else contextlib.nullcontext()):
                compiled(layer, x)
    finally:
        var.reset(token)
    got = collected["router_logits"]
    return cc.frame_count, got, want


def test_router_logits_capture_compiles_once_for_all_layers():
    from unsloth_zoo.temporary_patches.misc import output_capture_scratch
    frames, got, want = _run_layers(output_capture_scratch)
    assert frames <= 2, f"{frames} graphs for 10 layers: guarding on the router_logits list length"
    assert len(got) == len(want)
    for g, w in zip(got, want):
        torch.testing.assert_close(g.detach(), w)


def test_router_logits_list_length_recompiles_without_scratch():
    # Control: the guard this fix removes is real on this torch.
    frames, got, want = _run_layers(None)
    # Dynamo stops recompiling at its limit (8 by default; importing unsloth_zoo raises it).
    config = torch._dynamo.config
    limit = getattr(config, "recompile_limit", getattr(config, "cache_size_limit", 8))
    assert frames >= min(len(LENGTHS) - 1, limit)
    assert len(got) == len(want)


def test_scratch_is_a_no_op_without_an_active_collector():
    from unsloth_zoo.temporary_patches.misc import output_capture_scratch
    var = _collector()
    before = var.__dict__.get("_unsloth_eager_value")
    with output_capture_scratch():
        assert var.__dict__.get("_unsloth_eager_value") is before


def test_experts_interface_lookups_are_filled_before_any_trace():
    pytest.importorskip("transformers.integrations.moe")
    from unsloth_zoo.temporary_patches import moe_experts_interface as mei
    mei.patch_experts_interface()
    # A key first filled inside a compiled MoE layer guards on its absence: one more compile per layer kind.
    assert "default_apply_gate" in mei._LAZY and "layout_is_standard" in mei._LAZY


def test_scratch_keeps_capture_order_after_a_nested_reset():
    import threading
    from unsloth_zoo.temporary_patches.misc import output_capture_scratch
    var = _collector()
    saved = {k: var.__dict__.get(k) for k in ("_unsloth_eager_value", "_unsloth_eager_single", "_unsloth_eager_tls")}
    value = {"router_logits": ["earlier"]}
    tls = threading.local()
    tls.depth = 1
    var._unsloth_eager_value, var._unsloth_eager_single, var._unsloth_eager_tls = value, True, tls
    try:
        with output_capture_scratch():
            var._unsloth_eager_value["router_logits"].append("before")  # a hook inside the call
            var._unsloth_eager_value = value  # a nested set / reset restores the original collector
            var._unsloth_eager_value["router_logits"].append("after")  # later hooks append to it
        assert value["router_logits"] == ["earlier", "before", "after"]
    finally:
        for k, v in saved.items():
            if v is None:
                var.__dict__.pop(k, None)
            else:
                setattr(var, k, v)
