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

import torch

def test_fused_ce_loss(
    compute_fused_ce_loss,
    unsloth_fused_ce_loss,
    bszs = [1, 2, 1, 2,],
    qlens = [123, 323, 123, 323,],
    hds = [513, 1023, 877, 1111,],
    vocab_sizes = [16 * 1024, 32 * 1024, 32 * 1024, 64 * 1024,],
    dtypes = [torch.float32, torch.float32, torch.float32, torch.float32,],
    devices = ["cuda", "cuda", "cuda", "cuda",],
    scalings = [1.0, 16.0, 2.0, 4.0,],
    logit_scale_multiplys = [0.0, 1.0, 0.0, 0.0,],
    logit_scale_divides = [0.0, 0.0, 2.0, 0.0,],
    logit_softcappings = [10.0, 0.0, 0.0, 50.0,],
    seeds = [3407, 3408, 3409, 3410,],
    mask_ratios = [0.0, 0.1, 0.2, 0.3,],
    lm_head_requires_grads = [False, False, True, True,],
    lm_bias_requires_grads = [False, True, False, True,],
):
    for (
        bsz, qlen, hd, vocab_size, dtype, device, scaling,
        logit_scale_multiply, logit_scale_divide, logit_softcapping,
        seed, mask_ratio,
        lm_head_requires_grad, lm_bias_requires_grad,
    ) in zip(
        bszs, qlens, hds, vocab_sizes, dtypes, devices, scalings,
        logit_scale_multiplys, logit_scale_divides, logit_softcappings,
        seeds, mask_ratios,
        lm_head_requires_grads, lm_bias_requires_grads,
    ):
        kwargs = {}
        kwargs["logit_scale_multiply"] = logit_scale_multiply
        kwargs["logit_scale_divide"] = logit_scale_divide
        kwargs["logit_softcapping"] = logit_softcapping
        torch.cuda.manual_seed(seed)
        lm_head_weight = torch.randn((vocab_size, hd), dtype = dtype, device = device)
        lm_head_bias = torch.randn(vocab_size, dtype = dtype, device = device)
        input_ids = torch.randint(low = 0, high = vocab_size, size = (bsz, qlen), device = device)
        hidden_states = torch.randn((bsz, qlen, hd), dtype = dtype, device = device)
        hidden_states.requires_grad_(True)
        if lm_head_requires_grad: lm_head_weight.requires_grad_(True)
        if lm_bias_requires_grad: lm_head_bias.requires_grad_(True)

        # Mask out total ratio
        mask = torch.zeros(input_ids.numel(), dtype = torch.bool, device = device)
        mask[:int(mask.numel()*mask_ratio)] = 1
        permutation = torch.randperm(mask.numel())
        mask = mask[permutation]
        input_ids.ravel()[mask] = -100

        n_items = (input_ids != -100).sum()
        old_loss = compute_fused_ce_loss(
            hidden_states = hidden_states,
            lm_head_weight = lm_head_weight,
            lm_head_bias = lm_head_bias,
            labels = input_ids,
            n_items = n_items,
            scaling = scaling,
            **kwargs,
        )
        old_loss[0].backward()
        old_loss = old_loss[1][0]
        old_hidden_states_grad = hidden_states.grad.detach().clone()
        del hidden_states.grad
        if lm_head_requires_grad:
            old_lm_head_grad = lm_head_weight.grad.detach().clone()
            del lm_head_weight.grad
        if lm_bias_requires_grad:
            old_lm_bias_grad = lm_head_bias.grad.detach().clone()
            del lm_head_bias.grad

        new_loss = unsloth_fused_ce_loss(
            None,
            hidden_states = hidden_states,
            lm_head_weight = lm_head_weight,
            lm_head_bias = lm_head_bias,
            labels = input_ids,
            n_items = n_items,
            scaling = scaling,
            target_gb = 0.001, # Force chunking to per row
            torch_compile = True,
            **kwargs,
        )
        new_loss.backward()
        torch.testing.assert_close(old_loss, new_loss, msg = "Loss diff too high")
        new_hidden_states_grad = hidden_states.grad.detach().clone()
        del hidden_states.grad
        torch.testing.assert_close(new_hidden_states_grad, old_hidden_states_grad, msg = "Hidden States grad diff too high")
        if lm_head_requires_grad:
            new_lm_head_grad = lm_head_weight.grad.detach().clone()
            del lm_head_weight.grad
            torch.testing.assert_close(new_lm_head_grad, old_lm_head_grad, msg = "LM Head grad diff too high")
        if lm_bias_requires_grad:
            new_lm_bias_grad = lm_head_bias.grad.detach().clone()
            del lm_head_bias.grad
            torch.testing.assert_close(new_lm_bias_grad, old_lm_bias_grad, msg = "LM Bias grad diff too high")
    pass
pass

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