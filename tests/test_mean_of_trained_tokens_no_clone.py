# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""mean_of_trained_tokens reads the embeddings in place: no vocab x hidden copies, weights untouched."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from unsloth_zoo.tokenizer_utils import mean_of_trained_tokens


class _CountCopies(TorchDispatchMode):
    def __init__(self, numel):
        super().__init__()
        self.numel, self.big_copies = numel, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor) and out.numel() >= self.numel:
            self.big_copies += 1
        return out


class _Model(torch.nn.Module):
    def __init__(self, tied, dtype):
        super().__init__()
        self.emb = torch.nn.Embedding(512, 32, dtype = dtype)
        self.head = self.emb if tied else torch.nn.Linear(32, 512, bias = False, dtype = dtype)
        torch.manual_seed(0)
        with torch.no_grad():
            self.emb.weight.normal_()
            self.emb.weight[-5:] = 0
            if not tied:
                self.head.weight.normal_()

    def get_input_embeddings(self):  return self.emb
    def get_output_embeddings(self): return self.head


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_of_trained_tokens_reads_in_place(tied, dtype):
    model = _Model(tied, dtype)
    emb0 = model.emb.weight.detach().clone()
    head0 = model.head.weight.detach().clone()
    with _CountCopies(emb0.numel()) as counter:
        mean_emb, mean_head = mean_of_trained_tokens(model)
    assert counter.big_copies == 0

    assert torch.equal(model.emb.weight, emb0) and torch.equal(model.head.weight, head0)
    trained = emb0.float().abs().amax(1) > 1e-16
    torch.testing.assert_close(mean_emb, emb0.float()[trained].mean(0), rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(mean_head, head0.float()[trained].mean(0), rtol = 1e-5, atol = 1e-6)
    assert mean_emb.data_ptr() != model.emb.weight.data_ptr()
