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

# prepare_routed_experts is eager only: a compiled frame Inductor skips (SkipFrame, e.g. bfloat16 on a T4) runs
# eagerly, and its eager branch must not be traced as a nested frame. CPU only.
import os

import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
def test_table_check_runs_eagerly_when_the_compiled_frame_is_skipped():
    # Inductor raises SkipFrame for a graph it cannot compile (bfloat16 on a T4): the frame then runs eagerly, so
    # its eager-only branch calls prepare_routed_experts while Dynamo still intercepts nested frames. Tracing it
    # there fails (operator.itemgetter / data_ptr), so it must stay out of Dynamo.
    import torch._dynamo
    from torch._dynamo.exc import SkipFrame
    from unsloth_zoo.temporary_patches import gpt_oss_routed as routed

    class _Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_projs = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.down_projs = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    experts = _Experts()
    wrapper = {"base_layer": experts.gate_up_projs[0]}
    experts._unsloth_routed_nf4 = {"snapshot": {
        "mods": (tuple(experts.gate_up_projs._modules.values()), tuple(experts.down_projs._modules.values())),
        "wrappers": [wrapper], "wrapped": [object()],
    }}

    graphs = []

    def skip_first_backend(gm, example_inputs):
        # The outer frame's graph is declined, as Inductor declines a bfloat16 graph on a T4; nested ones compile.
        graphs.append(gm)
        if len(graphs) == 1:
            raise SkipFrame("graph not compilable here")
        return gm.forward

    def double(x):
        return x * 2

    def step(experts, x):
        if not torch.compiler.is_compiling():
            routed.prepare_routed_experts(experts)
        return double(x * 1)

    torch._dynamo.reset()
    try:
        out = torch.compile(step, backend = skip_first_backend, fullgraph = True)(experts, torch.ones(2))
    finally:
        torch._dynamo.reset()
    assert torch.equal(out, torch.full((2,), 2.0))
    assert len(graphs) >= 2  # the nested frame was compiled after the outer one was skipped



def test_routed_forwards_run_their_eager_branch_without_dynamo():
    # The routed forwards trace their compiled branch inline, but when Inductor skips that graph the frame runs
    # eagerly and its eager branch (pointer tables, direct Triton launches) must not be converted frame by frame.
    import sys
    import torch._dynamo
    from torch._dynamo.exc import SkipFrame
    from unsloth_zoo.temporary_patches import gpt_oss_routed as routed

    def eager_launch(x):
        assert sys._getframe() is not None  # Dynamo cannot trace sys._getframe
        return x + 1

    def forward(x):
        if torch.compiler.is_compiling():
            return x + 1  # the traceable (custom op) branch
        return eager_launch(x)

    def double(x):
        return x * 2

    def skip_add_backend(gm, example_inputs):
        # Declines every graph holding the forward's add, as Inductor declines a bfloat16 graph on a T4.
        if any(n.target is torch.ops.aten.add.Tensor or getattr(n.target, "__name__", "") == "add"
               for n in gm.graph.nodes):
            raise SkipFrame("graph not compilable here")
        return gm.forward

    for wrap, ok in ((routed._eager_out_of_dynamo, True), (lambda f: f, False)):
        fwd = wrap(forward)
        step = lambda x: double(fwd(x))  # noqa: E731
        torch._dynamo.reset()
        try:
            out = torch.compile(step, backend = skip_add_backend, fullgraph = True)(torch.ones(2))
            assert ok, "the unwrapped forward should fail here (its eager branch converted by Dynamo)"
            assert torch.equal(out, torch.full((2,), 4.0))
        except torch._dynamo.exc.Unsupported:
            assert not ok
        finally:
            torch._dynamo.reset()
    for fn in (routed.routed_experts_forward, routed.routed_bf16_forward, routed.routed_mlp_forward):
        assert fn.__wrapped__ is not None  # wrapped with _eager_out_of_dynamo


# Behavioural checks on the real forwards: each one's eager branch reaches an untraceable call inside a compiled
# frame the backend skips (every graph but the tail's is declined, as Inductor declines bfloat16 below sm80). The
# protected forward runs that branch with Dynamo off; without the protection Dynamo converts the nested frame.
def _skip_all_but_tail():
    from torch._dynamo.exc import SkipFrame

    def backend(gm, example_inputs):
        if "L_y_" in gm.code:
            return gm.forward
        raise SkipFrame("graph not compilable here")
    return backend


def _tail(y):
    return y * 3.25


def _untraceable(t):
    import sys

    sys._getframe()  # Dynamo cannot trace this: reached only on an eager branch
    return t + 1


def _run_skipped(step, *args):
    import torch._dynamo

    torch._dynamo.reset()
    try:
        with torch.no_grad():
            return torch.compile(step, backend = _skip_all_but_tail(), fullgraph = True)(*args)
    finally:
        torch._dynamo.reset()


class _EagerProbe:
    """An attribute read first on a forward's eager branch: a tensor op while traced (a non-empty graph, so the
    backend sees and skips the frame), an untraceable call when run eagerly."""

    def __init__(self, value):
        self.t, self.value, self.eager_reads = torch.zeros(1), value, 0

    def read(self):
        if torch.compiler.is_compiling():
            self.t.add(0)
        else:
            self.eager_reads += 1
            _untraceable(self.t)
        return self.value


def test_experts_forward_eager_branch_survives_a_skipped_frame():
    from unsloth_zoo.temporary_patches import gpt_oss_routed as routed

    probe = _EagerProbe(False)

    class Experts(torch.nn.Module):
        @property
        def training(self):
            return probe.read()

        @training.setter
        def training(self, value):
            pass

    experts, idx, w = Experts(), torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)

    def step(h):
        out = routed.routed_experts_forward(experts, h, idx, w)
        return _tail(h) if out is None else out

    out = _run_skipped(step, torch.ones(2, 4))
    assert probe.eager_reads == 1  # CPU inputs: the eager branch returns None after reading experts.training
    assert torch.equal(out, torch.full((2, 4), 3.25))


def test_bf16_forward_eager_branch_survives_a_skipped_frame(monkeypatch):
    import types
    from unsloth_zoo.temporary_patches import gpt_oss_routed as routed

    T, H, E = 2, 4, 3
    base = types.SimpleNamespace(hidden_size = H, gate_up_proj = torch.ones(E, H, 8), down_proj = torch.ones(E, 4, H),
                                 gate_up_proj_bias = None, down_proj_bias = None, alpha = 1.702, limit = 7.0)
    probe = _EagerProbe(None)

    def lora_terms(experts):
        probe.read()
        return base, {}

    monkeypatch.setattr(routed, "_lora_terms", lora_terms)
    monkeypatch.setattr(routed, "routed_bf16_moe", lambda x, *args, **kwargs: torch.zeros(x.shape[0], H))
    experts, idx, w = types.SimpleNamespace(), torch.zeros(T, 1, dtype = torch.long), torch.ones(T, 1)

    def step(h):
        return _tail(routed.routed_bf16_forward(experts, h, idx, w))

    out = _run_skipped(step, torch.ones(T, H))
    assert probe.eager_reads == 1
    assert torch.equal(out, torch.zeros(T, H))


def test_mlp_forward_eager_branch_survives_a_skipped_frame():
    from unsloth_zoo.temporary_patches import gpt_oss_routed as routed

    probe = _EagerProbe(False)

    class Hidden:
        # Stands in for hidden_states: is_cuda is the first attribute the eager branch reads.
        @property
        def is_cuda(self):
            return probe.read()

    mlp, hidden = torch.nn.Module(), Hidden()

    def step(h):
        out = routed.routed_mlp_forward(mlp, hidden)
        return _tail(h) if out is None else out

    out = _run_skipped(step, torch.ones(2, 4))
    assert probe.eager_reads == 1
    assert torch.equal(out, torch.full((2, 4), 3.25))
