# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""align_logprobs_with_mask: bit for bit the boolean-selection scatter it replaces (left padding, holes, all-zero
rows, narrower and wider masks, int / bool tool masks, NaN / -inf / -0.0 values), with fixed shapes only, so it
compiles as one graph with no data-dependent op."""

from __future__ import annotations

import pytest
import torch

from unsloth_zoo.rl_replacements import align_completion_tool_mask, align_logprobs_with_mask


def _boolean_select_reference(logprob_tensor, attention_mask, pad_value = 0.0):
    # The previous implementation, verbatim.
    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    mask_seq_len = attention_mask.shape[1]
    padded_logprobs = torch.full(attention_mask.shape, fill_value = pad_value, dtype = logprob_tensor.dtype, device = device)
    left_pad_counts = torch.argmax(attention_mask, dim = 1)
    cols = torch.arange(logprob_seq_len, device = device)
    dest_indices = left_pad_counts.unsqueeze(1) + cols
    row_indices = torch.arange(batch_size, device = device).unsqueeze(1).expand_as(dest_indices)
    valid_mask = dest_indices < mask_seq_len
    padded_logprobs[row_indices[valid_mask], dest_indices[valid_mask]] = logprob_tensor[valid_mask]
    return padded_logprobs


def _left_padded_mask(lengths, width):
    mask = torch.zeros(len(lengths), width, dtype = torch.long)
    for i, n in enumerate(lengths):
        if n:
            mask[i, width - n:] = 1
    return mask


def _cases():
    g = torch.Generator().manual_seed(0)
    yield "left_pad", torch.randn(4, 6, generator = g), _left_padded_mask([6, 3, 1, 5], 8), 0.0
    yield "all_zero_row", torch.randn(3, 5, generator = g), _left_padded_mask([5, 0, 2], 7), 0.0
    yield "all_zero_mask", torch.randn(2, 4, generator = g), torch.zeros(2, 4, dtype = torch.long), 0.0
    yield "narrower_mask", torch.randn(3, 9, generator = g), _left_padded_mask([4, 1, 3], 4), 0.0
    yield "equal_width", torch.randn(2, 5, generator = g), _left_padded_mask([5, 2], 5), 0.0
    holes = _left_padded_mask([6, 6], 8)
    holes[0, 4] = 0
    holes[1, -1] = 0
    yield "holes", torch.randn(2, 6, generator = g), holes, 0.0
    special = torch.tensor([[float("nan"), -float("inf"), -0.0, 1e-38], [3.0, float("inf"), -1.0, -0.0]])
    yield "special_values", special, _left_padded_mask([2, 4], 6), 0.0
    yield "pad_nan", torch.randn(3, 4, generator = g), _left_padded_mask([1, 4, 2], 6), float("nan")
    yield "pad_neg", torch.randn(2, 3, generator = g), _left_padded_mask([3, 1], 5), -100.0
    yield "bf16", torch.randn(3, 5, generator = g).bfloat16(), _left_padded_mask([5, 2, 4], 7), 0.0
    yield "fp64", torch.randn(2, 3, generator = g).double(), _left_padded_mask([1, 3], 4), 0.0
    yield "int_tool_mask", torch.randint(0, 2, (3, 4), generator = g), _left_padded_mask([4, 2, 1], 6), 0
    yield "bool_mask_values", torch.randint(0, 2, (2, 5), generator = g).bool(), _left_padded_mask([5, 3], 7), 0
    yield "bool_attention_mask", torch.randn(2, 3, generator = g), _left_padded_mask([3, 2], 5).float(), 0.0
    yield "empty_logprobs", torch.randn(3, 0, generator = g), _left_padded_mask([2, 0, 1], 4), 0.0
    yield "empty_batch", torch.randn(0, 4, generator = g), torch.zeros(0, 6, dtype = torch.long), 0.0
    yield "one_token", torch.randn(1, 1, generator = g), _left_padded_mask([1], 1), 0.0


@pytest.mark.parametrize("name, logps, mask, pad", list(_cases()), ids = [c[0] for c in _cases()])
def test_align_matches_boolean_selection_bit_for_bit(name, logps, mask, pad):
    expected = _boolean_select_reference(logps, mask, pad)
    got = align_logprobs_with_mask(logps, mask, pad)
    assert got.dtype == expected.dtype and got.shape == expected.shape and got.device == expected.device
    if got.is_floating_point():
        # Bit patterns: NaN payloads and -0.0 included.
        view = {torch.float32: torch.int32, torch.float64: torch.int64, torch.bfloat16: torch.int16}[got.dtype]
        assert torch.equal(got.view(view), expected.view(view))
    else:
        assert torch.equal(got, expected)


def test_align_zero_width_mask_raises_like_before():
    logps, mask = torch.randn(2, 3), torch.zeros(2, 0, dtype = torch.long)
    with pytest.raises(IndexError):
        _boolean_select_reference(logps, mask)
    with pytest.raises(IndexError):
        align_logprobs_with_mask(logps, mask)


def test_align_completion_tool_mask_unchanged():
    tool = torch.tensor([[1, 0, 1], [0, 1, 1]])
    completion = _left_padded_mask([3, 2], 5)
    out = align_completion_tool_mask(tool, completion)
    expected = completion * _boolean_select_reference(tool, completion, 0).to(completion.dtype)
    assert out.dtype == completion.dtype and torch.equal(out, expected)


def test_align_compiles_as_one_graph_with_no_data_dependent_sizes():
    # A boolean selection has a data-dependent size: Dynamo either cannot capture it or captures an unbacked size
    # (on CUDA, a read of that size back to the host). The gather has none.
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    compiled = torch.compile(align_logprobs_with_mask, backend = backend, fullgraph = True, dynamic = True)
    for B, L, W in [(4, 6, 8), (3, 5, 9), (5, 7, 7)]:
        logps = torch.randn(B, L)
        mask = _left_padded_mask([max(1, (i * 3) % (L + 1)) for i in range(B)], W)
        expected = _boolean_select_reference(logps, mask)
        assert torch.equal(compiled(logps, mask).view(torch.int32), expected.view(torch.int32))
    assert len(graphs) == 1
    unbacked = [
        n.name for n in graphs[0].graph.nodes
        if isinstance(n.meta.get("example_value"), torch.Tensor) and free_unbacked_symbols(n.meta["example_value"])
    ]
    assert not unbacked, unbacked
    torch._dynamo.reset()
