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
