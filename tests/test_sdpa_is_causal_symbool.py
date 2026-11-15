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

"""transformers 4.x SDPA attention (OLMoE, StableLM, Nemotron, DBRX, PhiMoE ...) sets
`is_causal = causal_mask is None and q_len > 1`, a SymBool once the sequence dim is dynamic (Unsloth's
per-layer block compile marks it). SDPA takes only a bool, so every attention call broke the graph.
The compiler rewrites it to the conditional form Llama uses."""
import textwrap

import torch

from unsloth_zoo.compiler import sdpa_is_causal_as_bool

ATTN = textwrap.dedent('''
    def attn(q, k, v, causal_mask = None):
        q_len = q.shape[2]
        is_causal = causal_mask is None and q_len > 1
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = causal_mask, is_causal = is_causal)
''')


def test_bool_form_is_rewritten_too():
    src = ATTN.replace("is_causal = causal_mask is None and q_len > 1", "is_causal = bool(causal_mask is None and q_len > 1)")
    assert _breaks(src) > 0
    out = sdpa_is_causal_as_bool(src)
    assert "is_causal = True if causal_mask is None and q_len > 1 else False" in out
    assert _breaks(out) == 0


def test_rewrite_is_exact():
    out = sdpa_is_causal_as_bool(ATTN)
    assert "is_causal = True if causal_mask is None and q_len > 1 else False" in out
    assert out.replace("True if causal_mask is None and q_len > 1 else False", "causal_mask is None and q_len > 1") == ATTN
    assert sdpa_is_causal_as_bool("is_causal = mask is None and q_len > 1 and x\n") == "is_causal = mask is None and q_len > 1 and x\n"


def _breaks(source):
    from torch._dynamo.utils import counters

    ns = {"torch": torch}
    exec(source, ns)
    fn = torch.compile(ns["attn"], backend = "eager", dynamic = True)
    torch._dynamo.reset()
    counters.clear()
    try:
        outs = []
        for L in (8, 12):
            q = torch.randn(1, 2, L, 16)
            outs.append((fn(q, q, q), torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal = True)))
    finally:
        torch._dynamo.reset()
    for got, want in outs:
        torch.testing.assert_close(got, want)
    return sum(counters["graph_break"].values())


def test_rewritten_attention_compiles_without_a_break():
    assert _breaks(ATTN) > 0  # the stock form breaks under dynamic shapes (else this test proves nothing)
    assert _breaks(sdpa_is_causal_as_bool(ATTN)) == 0
