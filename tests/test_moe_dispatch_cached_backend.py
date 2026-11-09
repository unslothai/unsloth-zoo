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

"""The experts dispatch runs once per MoE layer per forward: it must reuse the resolved backend, since
get_forward_moe_backend() re-reads and compares the cached moe_utils.py on every call."""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")


class _Experts(nn.Module):
    has_gate = True


@pytest.mark.parametrize("resolved", [True, False])
def test_dispatch_reuses_the_resolved_backend(monkeypatch, resolved):
    from unsloth_zoo.temporary_patches import moe_experts_interface as mei
    from unsloth_zoo.temporary_patches import moe_utils

    seen = []

    def backend(module, hidden_states, top_k_index, top_k_weights):
        seen.append(module)
        return hidden_states

    lookups = []

    def get_forward_moe_backend():
        lookups.append(1)
        monkeypatch.setattr(moe_utils, "_CACHED_FORWARD_MOE_BACKEND", backend)
        return backend

    monkeypatch.setattr(moe_utils, "_CACHED_FORWARD_MOE_BACKEND", backend if resolved else None)
    monkeypatch.setattr(moe_utils, "get_forward_moe_backend", get_forward_moe_backend)
    monkeypatch.setattr(mei, "_dense_standard_with_expert_lora", lambda module: False)
    monkeypatch.setattr(mei, "_has_custom_gate", lambda module: False)

    experts = _Experts()
    hidden = torch.zeros(2, 4)
    for _ in range(3):
        out = mei._unsloth_experts_dispatch(experts, hidden, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1))
        assert out is hidden
    assert seen == [experts] * 3
    # Resolved once (at most), not once per forward.
    assert len(lookups) == (0 if resolved else 1)


@pytest.mark.parametrize("resolved", [True, False])
def test_bnb4bit_generic_route_reuses_the_resolved_backend(monkeypatch, resolved):
    from unsloth_zoo.temporary_patches import moe_utils
    from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as mb

    seen = []

    def backend(module, hidden_states, top_k_index, top_k_weights):
        seen.append(module)
        return hidden_states

    lookups = []

    def resolve():
        lookups.append(1)
        monkeypatch.setattr(moe_utils, "_CACHED_FORWARD_MOE_BACKEND", backend)
        return backend

    monkeypatch.setattr(moe_utils, "_CACHED_FORWARD_MOE_BACKEND", backend if resolved else None)
    experts = _Experts()
    hidden = torch.zeros(2, 4)
    for _ in range(3):
        out = mb._forward_generic_experts_eagerly(
            resolve, experts, hidden, torch.zeros(2, 1, dtype = torch.long), torch.ones(2, 1)
        )
        assert out is hidden
    assert seen == [experts] * 3
    assert len(lookups) == (0 if resolved else 1)
