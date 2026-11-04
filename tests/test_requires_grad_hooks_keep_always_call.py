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

"""requires_grad_for_gradient_checkpointing must keep always_call on the forward hooks it re-registers."""

import pytest
import torch

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Linear(4, 4)
        self.fail = False

    def forward(self, hidden_states):
        if self.fail:
            raise RuntimeError("boom")
        return self.lora(hidden_states)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()

    def forward(self, hidden_states):
        return self.block(hidden_states)


def _prepared_model():
    model = _Model()
    for p in model.parameters():
        p.requires_grad_(True)
    calls = {"always": 0, "plain": 0}

    def always(module, args, output):
        calls["always"] += 1

    def plain(module, args, output):
        calls["plain"] += 1

    model.block.register_forward_hook(always, always_call = True)
    model.block.register_forward_hook(plain)
    requires_grad_for_gradient_checkpointing(model)
    return model, calls


def test_always_call_hook_still_runs_when_forward_raises():
    model, calls = _prepared_model()
    model.block.fail = True
    with pytest.raises(RuntimeError, match = "boom"):
        model(torch.randn(2, 4))
    assert calls == {"always": 1, "plain": 0}, calls


def test_hooks_still_run_after_a_good_forward():
    model, calls = _prepared_model()
    model(torch.randn(2, 4))
    assert calls == {"always": 1, "plain": 1}, calls
    assert set(model.block._forward_hooks_always_called) <= set(model.block._forward_hooks)
