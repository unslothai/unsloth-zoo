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

# Guards the shared offloaded-embedding op: same rows as nn.Embedding, traceable without a graph
# break, and the one gpt-oss uses (a second copy would drift from the generic installer).
import pytest
import torch

from unsloth_zoo.offloaded_embedding import offloaded_embedding

needs_op = pytest.mark.skipif(offloaded_embedding is None, reason = "torch.library.custom_op unavailable")


def _table(V = 1000, H = 32, padding_idx = None):
    torch.manual_seed(0)
    # Frozen, as callers use it: the op has no autograd formula.
    return torch.nn.Embedding(V, H, padding_idx = padding_idx, dtype = torch.bfloat16).requires_grad_(False)


@needs_op
@pytest.mark.parametrize("padding_idx", [None, 3])
def test_rows_match_nn_embedding_on_cpu(padding_idx):
    emb = _table(padding_idx = padding_idx)
    ids = torch.randint(0, 1000, (4, 9))
    ids[0, 0] = 3
    out = offloaded_embedding(ids, emb.weight, padding_idx)
    assert torch.equal(out, emb(ids))
    assert out.device == ids.device and out.dtype == emb.weight.dtype


@needs_op
def test_scale_matches_scaled_word_embedding():
    # transformers' ScaledWordEmbedding: rows * embed_scale.to(weight.dtype), on the table's device.
    emb = _table(padding_idx = 0)
    scale = torch.tensor(32 ** 0.5)
    ids = torch.randint(0, 1000, (3, 6))
    out = offloaded_embedding(ids, emb.weight, 0, scale)
    assert torch.equal(out, emb(ids) * scale.to(emb.weight.dtype))


@needs_op
def test_fullgraph_compile_has_no_graph_break():
    from torch._dynamo.utils import counters
    emb = _table()
    scale = torch.tensor(2.0, dtype = torch.bfloat16)

    def f(ids):
        return offloaded_embedding(ids, emb.weight, None) * scale

    torch._dynamo.reset()
    counters.clear()
    ids = torch.randint(0, 1000, (2, 5))
    out = torch.compile(f, fullgraph = True, backend = "aot_eager")(ids)
    assert sum(counters["graph_break"].values()) == 0
    assert torch.equal(out, emb(ids) * scale)


def test_op_registers_wherever_custom_op_exists():
    # torch 2.4 to 2.7 have custom_op but no `tags` keyword; registration must not depend on it.
    if getattr(torch.library, "custom_op", None) is not None:
        assert offloaded_embedding is not None


@needs_op
def test_gpt_oss_uses_the_shared_op():
    try:
        from unsloth_zoo.temporary_patches import gpt_oss
    except Exception as exc:  # transformers without gpt-oss
        pytest.skip(f"gpt_oss patches unavailable: {exc}")
    assert gpt_oss._offloaded_embedding is offloaded_embedding


@needs_op
@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_cpu_table_cuda_ids_under_cuda_graphs():
    emb = _table(V = 50021, H = 64)
    proj = torch.randn(64, 8, device = "cuda", dtype = torch.bfloat16)

    def step(ids):
        return offloaded_embedding(ids, emb.weight, None) @ proj

    compiled = torch.compile(step, fullgraph = True, mode = "reduce-overhead")
    with torch.no_grad():
        for i in range(6):
            ids = torch.randint(0, 50021, (4, 1), device = "cuda", generator = None)
            torch.compiler.cudagraph_mark_step_begin()
            out = compiled(ids).clone()
            torch.testing.assert_close(out, emb(ids.cpu()).cuda() @ proj)
