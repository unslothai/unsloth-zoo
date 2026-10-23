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

"""requires_grad_for_gradient_checkpointing rebuilds a module's hook dict to drop older requires-grad
hooks; the hooks it keeps must keep their with_kwargs registration. A pre-hook registered with
with_kwargs = True (Unsloth's inputs_embeds hook on every PreTrainedModel) re-registered without it is
called with two arguments on the next forward: TypeError (Qwen3-VL / Qwen3.5 vision towers)."""

import torch

from unsloth_zoo.peft_utils import requires_grad_for_gradient_checkpointing


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.lora(hidden_states)


class _Tower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])

    def get_input_embeddings(self):
        raise NotImplementedError

    def forward(self, hidden_states = None):
        for blk in self.blocks:
            hidden_states = blk(hidden_states)
        return hidden_states


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Tower()

    def forward(self, hidden_states = None):
        return self.visual(hidden_states = hidden_states)


def test_kept_kwargs_pre_hook_still_gets_kwargs():
    model = _Model()
    for p in model.parameters():
        p.requires_grad_(True)
    seen = []

    def make_inputs_embeds_require_grads(module, args, kwargs):
        seen.append(sorted(kwargs))
        return None

    model.visual.register_forward_pre_hook(make_inputs_embeds_require_grads, with_kwargs = True)
    requires_grad_for_gradient_checkpointing(model)
    # The tower got the fallback pre-hook, and the existing kwargs hook was re-registered.
    names = [getattr(h, "__name__", "") for h in model.visual._forward_pre_hooks.values()]
    assert "requires_grad_pre_hook" in names, names
    model(hidden_states = torch.randn(2, 4))
    assert seen == [["hidden_states"]], seen
