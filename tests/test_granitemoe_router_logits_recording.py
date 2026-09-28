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

"""Granite MoE CausalLMs return a real router aux loss under output_router_logits=True (transformers 5.x)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.utils.output_capturing")
import transformers  # noqa: E402

from unsloth_zoo.temporary_patches.misc import patch_granitemoe_router_logits_recording  # noqa: E402

_BASE = dict(vocab_size = 64, hidden_size = 32, intermediate_size = 32, num_hidden_layers = 2,
             num_attention_heads = 4, num_key_value_heads = 2, num_local_experts = 4, num_experts_per_tok = 2)
_CASES = {
    "GraniteMoeConfig": {},
    "GraniteMoeSharedConfig": dict(shared_intermediate_size = 32),
    "GraniteMoeSWAConfig": {},
    "GraniteMoeHybridConfig": dict(shared_intermediate_size = 32, layer_types = ["mamba", "attention"],
                                   mamba_n_heads = 4, mamba_d_head = 16),
}


def _build(config_name):
    config_cls = getattr(transformers, config_name, None)
    if config_cls is None:
        pytest.skip(f"{config_name} not in this transformers")  # reason: older 5.x ship fewer Granite MoE classes and the patch only touches classes that exist
    kwargs = {**_BASE, **_CASES[config_name]}
    try:
        config = config_cls(**kwargs)
    except Exception:  # layer_types spelling differs across 5.x
        kwargs["layer_types"] = ["linear_attention", "full_attention"]
        config = config_cls(**kwargs)
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config).eval()


@pytest.mark.parametrize("config_name", list(_CASES))
def test_granite_moe_aux_loss_is_a_tensor_added_to_the_loss(config_name):
    patch_granitemoe_router_logits_recording()
    model = _build(config_name)
    # The CausalLM forward does not pass its output_router_logits kwarg to the inner model, which reads the config;
    # TRL >= 1.7 sets the config flag, so mirror that.
    model.config.output_router_logits = True
    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        with_aux = model(input_ids = ids, labels = ids, output_router_logits = True)
        without = model(input_ids = ids, labels = ids, output_router_logits = False)
    assert isinstance(with_aux.aux_loss, torch.Tensor) and torch.isfinite(with_aux.aux_loss)
    assert len(with_aux.router_logits) == 2 and with_aux.router_logits[0].shape[-1] == 4
    expected = without.loss + model.config.router_aux_loss_coef * with_aux.aux_loss
    torch.testing.assert_close(with_aux.loss, expected)


def test_patch_is_idempotent():
    patch_granitemoe_router_logits_recording()
    patch_granitemoe_router_logits_recording()
    from transformers.models.granitemoehybrid import modeling_granitemoehybrid as m

    assert list(m.GraniteMoeHybridPreTrainedModel._can_record_outputs).count("router_logits") == 1
