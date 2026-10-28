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

"""Gradient checkpoint recompute must retrace the first block-compiled bnb 4-bit experts graph; recording the backend
log key while tracing made Dynamo guard on it, so recompute missed the forward graph and raised CheckpointError on
the first backward."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

mb = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils_bnb4bit")


def test_backend_log_key_not_recorded_while_tracing():
    class Experts(torch.nn.Module):
        pass

    module = Experts()
    message = "Unsloth: test backend message."
    key = (id(Experts), message)

    def step(x):
        mb._log_moe_bnb4bit_backend_once(module, message)
        return x + 1

    torch._dynamo.reset()
    try:
        compiled = torch.compile(step, backend = "eager", fullgraph = True)
        compiled(torch.ones(2))
        assert key not in mb._LOGGED_BACKENDS
        # Eager calls still log once.
        step(torch.ones(2))
        assert key in mb._LOGGED_BACKENDS
    finally:
        mb._LOGGED_BACKENDS.discard(key)
        torch._dynamo.reset()


_COLD_CHECKPOINT_SCRIPT = textwrap.dedent(
    """
    import importlib.util, sys
    import torch
    from torch.utils.checkpoint import checkpoint
    from unsloth_zoo.temporary_patches import moe_utils as mu
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as mb
    from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE

    spec = importlib.util.spec_from_file_location("route_tests", sys.argv[1])
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)

    model = torch.nn.Module()
    model.experts = t._init(t._make_experts_class()(t._config()))
    t._quantize_like_the_loader(model, "cuda:0")
    experts = model.experts
    assert mb._moe_uses_bnb4bit_expert_weights(experts)
    mu.get_forward_moe_backend()

    class Layer(torch.nn.Module):
        def __init__(self, experts):
            super().__init__()
            self.experts = experts

        def forward(self, hidden_states, idx, weights):
            flat = hidden_states.reshape(-1, hidden_states.shape[-1]) * 2
            return (self.experts(flat, idx, weights) + 1).reshape_as(hidden_states)

    layer = Layer(experts).train()
    # As Unsloth's per-layer block compile: inductor, fullgraph off, the decode flag set while tracing.
    compiled = torch.compile(layer.forward, fullgraph = False, dynamic = None)

    def block(*args):
        previous = UNSLOTH_DECODE_COMPILE[0]
        UNSLOTH_DECODE_COMPILE[0] = True
        try:
            return compiled(*args)
        finally:
            UNSLOTH_DECODE_COMPILE[0] = previous

    for n in (16, 32):
        x, idx, w = t._routing(n, device = "cuda:0")
        x = x.to(torch.bfloat16).unsqueeze(0).requires_grad_()
        w = w.to(torch.bfloat16).requires_grad_()
        out = checkpoint(block, x, idx, w, use_reentrant = False)
        gx, gw = torch.autograd.grad(out.float().square().sum(), (x, w))
        assert torch.isfinite(gx).all() and torch.isfinite(gw).all()
    print("COLD_CHECKPOINT_OK")
    """
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "bnb 4-bit needs CUDA")
def test_cold_block_compiled_nf4_experts_checkpointed_backward(tmp_path):
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("transformers.integrations.moe")
    route_tests = os.path.join(os.path.dirname(__file__), "test_moe_bnb4bit_generic_experts_route.py")
    env = dict(os.environ)
    env.setdefault("UNSLOTH_IS_PRESENT", "1")
    env.pop("UNSLOTH_ENABLE_LOGGING", None)
    # Fresh process and caches: the failure only shows on the first, cold call.
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "inductor")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    env["UNSLOTH_COMPILE_LOCATION"] = str(tmp_path / "compiled")
    result = subprocess.run(
        [sys.executable, "-c", _COLD_CHECKPOINT_SCRIPT, route_tests],
        env = env, capture_output = True, text = True, timeout = 900,
    )
    assert result.returncode == 0 and "COLD_CHECKPOINT_OK" in result.stdout, (
        result.stdout[-4000:] + result.stderr[-4000:]
    )
