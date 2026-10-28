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

# The shared routed MoE path's eager branch (prepare_stacked_nf4, pointer tables, direct Triton launches) must stay
# out of Dynamo when a compiled frame is skipped (SkipFrame, e.g. bfloat16 on a T4: the frame then runs eagerly while
# Dynamo still converts its nested frames). CPU only.
import os
import types

import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(64, 1, dtype = torch.uint8), requires_grad = False)
        self.down_proj = torch.nn.Parameter(torch.zeros(32, 1, dtype = torch.uint8), requires_grad = False)
        for p in (self.gate_up_proj, self.down_proj):
            p.quant_state = types.SimpleNamespace(absmax = torch.ones(8), code = torch.ones(16), nested = False,
                                                  blocksize = 64)


def test_stacked_table_check_runs_eagerly_when_the_compiled_frame_is_skipped():
    import torch._dynamo
    from torch._dynamo.exc import SkipFrame
    from unsloth_zoo.temporary_patches import moe_routed as MR

    experts = _Experts()
    # A state built on an earlier eager call: revalidating it compares data pointers, which Dynamo cannot trace.
    experts.__dict__["_unsloth_routed_moe"] = {"key": MR._stacked_key(experts), "sem": MR._semantics(experts)}
    graphs = []

    def skip_first_backend(gm, example_inputs):
        graphs.append(gm)
        if len(graphs) == 1:
            raise SkipFrame("graph not compilable here")
        return gm.forward

    def double(x):
        return x * 2

    def step(x):
        if not torch.compiler.is_compiling():
            assert MR.prepare_stacked_nf4(experts) is experts.__dict__["_unsloth_routed_moe"]
        return double(x * 1)

    torch._dynamo.reset()
    try:
        out = torch.compile(step, backend = skip_first_backend, fullgraph = True)(torch.ones(2))
    finally:
        torch._dynamo.reset()
    assert torch.equal(out, torch.full((2,), 2.0))
    assert len(graphs) >= 2  # the nested frame was compiled after the outer one was skipped


def test_routed_entry_points_run_their_eager_branch_without_dynamo():
    from unsloth_zoo.temporary_patches import gpt_oss_routed as GR
    from unsloth_zoo.temporary_patches import moe_routed as MR

    assert GR._eager_out_of_dynamo is MR._eager_out_of_dynamo
    for fn in (MR.routed_moe_forward, MR.routed_bf16_moe):
        assert fn.__wrapped__ is not None  # wrapped with _eager_out_of_dynamo
    # Below the wrapper, a compiled trace still inlines the forward: CPU inputs take its early None.
    experts, idx, w = _Experts(), torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)
    out = torch.compile(lambda h: MR.routed_moe_forward(experts, h, idx, w), backend = "eager", fullgraph = True)(
        torch.ones(2, 4))
    assert out is None
