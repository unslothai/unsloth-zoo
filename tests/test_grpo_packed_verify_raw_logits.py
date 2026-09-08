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

"""The packed path's first-use verifier needs the same width dispatch.

The packed call site dispatches on width, but the self-verify below it takes a
second per-row forward and always sends it through the lm_head matmul helper.
A forward that ignores UNSLOTH_RETURN_HIDDEN_STATES returns real vocab logits
from both, so the verifier raises, the outer handler sets
`_unsloth_seq_packing_grad_ok = False`, and packing is off for the rest of the
run. Since the very first packed batch is always verified, the raw-logits
branch would never be reachable.

The block is exec'd from its own source against a model that returns real
logits, so the test exercises the dispatch rather than reading it.
"""

import contextlib
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

import unsloth_zoo.rl_replacements as rr


VOCAB, HIDDEN = 17, 8
PAD_ID, L, KEEP = 0, 8, 4


class _Model(torch.nn.Module):
    """`hidden_states = False` ignores UNSLOTH_RETURN_HIDDEN_STATES and returns
    real [.., vocab] logits; True is the Unsloth generated forward.

    Position-local, so the packed block-diagonal forward and the per-row
    forward agree exactly and the verifier's own tolerance is not what is
    under test here.
    """
    def __init__(self, hidden_states = False):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.hidden_states = hidden_states

    def forward(self, input_ids = None, position_ids = None,
                packed_seq_lengths = None, use_cache = None, **kwargs):
        h = torch.tanh(self.emb(input_ids))
        return SimpleNamespace(logits = h if self.hidden_states else self.head(h))


def _packed_block_source():
    src = inspect.getsource(rr.grpo_accumulated_loss)
    start = src.index("    new_logprobs = None")
    end = src.index("    # ---- PrefixGrouper resolution")
    return textwrap.dedent(src[start:end])


def _run_packed_block(hidden_states = False):
    """Exec the real packed + verify block and hand back its locals."""
    model = _Model(hidden_states = hidden_states)
    lm_head = model.head.weight                       # [vocab, hidden]
    # left-padded rows, as the caller has already left-packed them
    input_ids = torch.tensor([
        [PAD_ID, PAD_ID, 3, 5, 7, 9, 11, 13],
        [PAD_ID, 2,      4, 6, 8, 10, 12, 14],
    ])
    left_pad = rr.calculate_pad_tokens_in_prompt(input_ids, KEEP, PAD_ID)
    max_left_pad = int(left_pad.max())
    completion_mask = rr.create_completion_attention_mask(
        input_ids[:, -(KEEP + max_left_pad):], left_pad, max_left_pad, PAD_ID,
    )
    ns = {
        "os": rr.os, "torch": torch,
        "chunked_hidden_states_selective_log_softmax":
            rr.chunked_hidden_states_selective_log_softmax,
        "chunked_selective_log_softmax": rr.chunked_selective_log_softmax,
        "create_completion_attention_mask": rr.create_completion_attention_mask,
        "device_synchronize": lambda *a, **k: None,
        "UNSLOTH_ENABLE_LOGGING": False,
        "trainer": SimpleNamespace(processing_class = SimpleNamespace(pad_token_id = PAD_ID)),
        "unwrapped_model": model,
        "lm_head": lm_head,
        "input_ids": input_ids,
        "left_pad_tokens_per_prompt": left_pad,
        "max_left_pad": max_left_pad,
        "completion_mask": completion_mask,
        "logits_to_keep": KEEP,
        "total_rows": input_ids.shape[0],
        "multiplier": 1,
        "autocaster": contextlib.nullcontext(),
        "pixel_values": None, "token_type_ids": None, "mm_token_type_ids": None,
        "_pg_skip_pack": False,
        "logit_scale_multiply": 0.0, "logit_scale_divide": 0.0,
        "logit_softcapping": 0.0, "temperature": 1.0,
    }
    exec(_packed_block_source(), ns)
    return ns, model, input_ids, max_left_pad


def _reference_logprobs(model, input_ids, max_left_pad):
    """Per-row logprobs straight from the model, no packing involved.

    Pads are dropped first, so the row's leading token has no predecessor and
    stays 0, exactly as both the packed scatter and the padded loop leave it.
    """
    width = KEEP + max_left_pad
    out = torch.zeros(input_ids.shape[0], L)
    for row in range(input_ids.shape[0]):
        cols = (input_ids[row] != PAD_ID).nonzero(as_tuple = False).squeeze(1)
        real = input_ids[row][cols].unsqueeze(0)
        with torch.no_grad():
            out_ = model(input_ids = real).logits.float()
            if model.hidden_states: out_ = out_ @ model.head.weight.t().float()
            logps = torch.log_softmax(out_, dim = -1)[0]
        for j in range(1, real.shape[1]):
            out[row, cols[j]] = logps[j - 1, real[0, j]]
    return out[:, -width:]


def test_verifier_survives_a_forward_that_returns_real_logits():
    ns, model, input_ids, max_left_pad = _run_packed_block()
    assert getattr(model, "_unsloth_seq_packing_grad_ok", None) is not False, (
        "the first-use verifier sent raw logits into the lm_head matmul, so "
        "packing was disabled for the whole run and the packed raw-logits "
        "branch is unreachable"
    )
    assert ns["_pack_use"] is True
    assert ns["_pack_result"] is not None


@pytest.mark.parametrize("hidden_states", [False, True])
def test_verified_packed_result_matches_the_per_row_logprobs(hidden_states):
    ns, model, input_ids, max_left_pad = _run_packed_block(hidden_states = hidden_states)
    if not ns["_pack_use"]:
        pytest.fail("packed path rejected; nothing to compare")
    mask = rr.create_completion_attention_mask(
        input_ids[:, -(KEEP + max_left_pad):],
        rr.calculate_pad_tokens_in_prompt(input_ids, KEEP, PAD_ID),
        max_left_pad, PAD_ID,
    ).float()
    ref = _reference_logprobs(model, input_ids, max_left_pad)
    got = ns["_pack_result"].detach().float()
    assert torch.allclose(got * mask, ref * mask, atol = 1e-5), (got * mask, ref * mask)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
