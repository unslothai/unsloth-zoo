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

"""CPU forwards with an int logits_to_keep (generate() prefill, GRPO log-probs) run the bf16 head as one GEMM."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        if str(func.overloadpacket) == "aten.bmm":
            self.ops.append(tuple(tuple(a.shape) for a in args[:2]))
        return func(*args, **(kwargs or {}))


def _tiny(dtype, patched = True):
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth_zoo.cpu.ops import patch_chunked_causal_lm_loss
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size = 96, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                         num_attention_heads = 4, num_key_value_heads = 2, tie_word_embeddings = False)
    model = LlamaForCausalLM(config).to(dtype).eval()
    model.lm_head.weight.requires_grad_(False)  # LoRA: a trainable head makes torch fold into one GEMM anyway
    if patched:
        assert patch_chunked_causal_lm_loss(model)
    return model


@pytest.mark.parametrize("keep", [1, 5])
@pytest.mark.parametrize("grad", [False, True])
def test_bf16_head_one_gemm_and_identical_logits(keep, grad):
    model, stock = _tiny(torch.bfloat16), _tiny(torch.bfloat16, patched = False)
    ids = torch.randint(0, 96, (3, 11))
    with torch.set_grad_enabled(grad):
        with _Ops() as seen:
            logits = model(ids, logits_to_keep = keep).logits
        reference = stock(ids, logits_to_keep = keep).logits
    assert logits.shape == (3, keep, 96)
    assert torch.equal(logits, reference)
    # torch < 2.14 runs a strided hidden[:, -k:] slice through bmm against the head expanded per row ([B, H, V]).
    assert not [shapes for shapes in seen.ops if shapes[1][-1] == 96], seen.ops
    if grad:
        logits.float().sum().backward()
        reference.float().sum().backward()
        for (name, p), (_, q) in zip(model.named_parameters(), stock.named_parameters()):
            if q.grad is None:  # the frozen head
                assert p.grad is None, name
                continue
            # bf16: mm and bmm round the hidden-state gradient differently (one ulp), which then propagates.
            scale = float(q.grad.float().abs().max()) or 1.0
            assert float((p.grad.float() - q.grad.float()).abs().max()) / scale < 3e-2, name


def test_generate_tokens_unchanged():
    model, stock = _tiny(torch.bfloat16), _tiny(torch.bfloat16, patched = False)
    ids = torch.randint(0, 96, (2, 9))
    kwargs = dict(max_new_tokens = 6, do_sample = False, pad_token_id = 0)
    assert torch.equal(model.generate(ids, **kwargs), stock.generate(ids, **kwargs))


def test_int_kept_where_index_not_needed_or_unsupported(monkeypatch):
    from unsloth_zoo.cpu import ops
    bf16, fp32 = _tiny(torch.bfloat16), _tiny(torch.float32)
    ids = torch.zeros(2, 7, dtype = torch.long)
    convert = ops._logits_to_keep_as_index
    index = convert(bf16, ids, None, 3)
    assert torch.equal(index, torch.arange(4, 7))
    embeds = torch.zeros(2, 7, 32, dtype = torch.bfloat16)
    assert torch.equal(convert(bf16, None, embeds, 1), torch.tensor([6]))
    assert convert(fp32, ids, None, 3) == 3  # fp32 heads already fold
    assert convert(bf16, ids, None, 0) == 0  # 0 = every position
    assert convert(bf16, ids, None, 7) == 7 and convert(bf16, ids, None, 9) == 9  # nothing to slice
    tensor = torch.tensor([1, 2])
    assert convert(bf16, ids, None, tensor) is tensor
    assert convert(bf16, None, None, 1) == 1
    # A forward that does not slice with transformers' int-or-tensor pattern keeps its int.
    monkeypatch.setattr(ops, "_forward_takes_index_logits_to_keep", lambda cls: False)
    assert convert(bf16, ids, None, 3) == 3
