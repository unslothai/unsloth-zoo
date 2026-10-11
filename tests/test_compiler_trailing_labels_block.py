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

"""Labels-block statements after the loss call (Bamba's z-loss) survive the fused branches."""

import inspect
import os
import textwrap
import types

import pytest
import torch

from unsloth_zoo import compiler

# Mirrors BambaForCausalLM.forward's head + labels block (transformers 4.55 .. 5.18).
FORWARD = """    def forward(self, hidden_states, labels=None, logits_to_keep=0, **kwargs):
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            if self.z_loss_coefficient > 0:
                z_loss = logits.logsumexp(dim=-1).to(dtype=loss.dtype).pow(2).mean()
                loss = loss + self.z_loss_coefficient * z_loss

        return loss, logits
"""


def ForCausalLMLoss(logits, labels, vocab_size = None, **kwargs):
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index = -100,
    )


def _run(source, z, return_logits):
    calls = []

    def fused(hidden_states, lm_head_weight, labels, **kwargs):
        calls.append(1)
        return ForCausalLMLoss(hidden_states @ lm_head_weight.t(), labels)

    ns = dict(
        torch = torch, os = os, EMPTY_LOGITS = torch.empty(0), UNSLOTH_ENABLE_CCE = False,
        HAS_CUT_CROSS_ENTROPY = False, UNSLOTH_COMPILE_DISABLE = True, unsloth_fused_ce_loss = fused,
    )
    exec(textwrap.dedent(source), ns)
    torch.manual_seed(0)
    lm_head = torch.nn.Linear(8, 16, bias = False)
    model = types.SimpleNamespace(
        lm_head = lm_head, loss_function = ForCausalLMLoss, z_loss_coefficient = z,
        config = types.SimpleNamespace(vocab_size = 16),
    )
    hidden = torch.randn(2, 5, 8)
    labels = torch.randint(0, 16, (2, 5))
    old = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1" if return_logits else "0"
    try:
        loss = ns["forward"](model, hidden, labels = labels)[0]
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = old
    return loss, len(calls)


@pytest.mark.parametrize("return_logits", [False, True])
@pytest.mark.parametrize("z", [0.0, 0.1])
def test_trailing_if_block_keeps_z_loss(z, return_logits):
    new, fused = compiler.apply_fused_lm_head(FORWARD, "BambaForCausalLM")
    assert fused
    expected, _ = _run(FORWARD, z, return_logits)
    got, n_fused = _run(new, z, return_logits)
    torch.testing.assert_close(got, expected)
    # The fused kernel runs only when the trailing block would not read the logits.
    assert n_fused == (1 if (z == 0.0 and not return_logits) else 0)


def test_other_trailing_statements_are_not_fused():
    source = FORWARD.replace(
        "            if self.z_loss_coefficient > 0:\n",
        "            loss = loss + logits.mean()\n            if self.z_loss_coefficient > 0:\n",
    )
    new, fused = compiler.apply_fused_lm_head(source, "BambaForCausalLM")
    assert not fused
    assert "EMPTY_LOGITS" not in new


def test_real_bamba_forward_is_guarded():
    bamba = pytest.importorskip("transformers.models.bamba.modeling_bamba")
    source = compiler.fixup_fused_lm_head(inspect.getsource(bamba.BambaForCausalLM.forward))
    if "z_loss" not in source:
        pytest.skip("this transformers BambaForCausalLM has no z-loss")
    new, fused = compiler.apply_fused_lm_head(source, "BambaForCausalLM")
    assert fused
    assert new.count("and not (self.z_loss_coefficient > 0):") == 2
    assert "and labels is not None and not NOT_RETURN_LOGITS:" in new
    assert new.count("z_loss = logits.logsumexp(") == 2
