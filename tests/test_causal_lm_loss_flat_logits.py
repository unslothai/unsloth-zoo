# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""The patched causal LM loss accepts logits already flattened to (tokens, vocab).

transformers' stock ForCausalLMLoss does, and remote code relies on it: the
multi-token-prediction head of inclusionAI/Ling-2.6-flash passes
`mtp_logits.view(-1, vocab)` with 1-D labels. The patched loss handed those
straight to a kernel that unpacks (batch, seq, vocab) and failed with
"not enough values to unpack (expected 3, got 2)" on the first training step.
"""

import subprocess
import sys
import textwrap

_RUNNER = textwrap.dedent(
    """
    import torch
    import torch.nn.functional as F
    import transformers.loss.loss_utils as LU
    from unsloth_zoo.loss_utils import patch_loss_functions

    stock = LU.ForCausalLMLoss

    def fast_ce(logits, labels, n_items = None):
        batch, seq_len, d = logits.shape  # what the Unsloth kernel does
        loss = F.cross_entropy(
            logits.reshape(-1, d).float(), labels.reshape(-1), ignore_index = -100, reduction = "sum"
        )
        n = n_items if n_items is not None else (labels != -100).sum()
        return loss / n

    patch_loss_functions(fast_ce, torch_compile = False)
    patched = LU.LOSS_MAPPING["ForCausalLM"]
    assert patched is not stock

    torch.manual_seed(0)
    V = 32
    logits = torch.randn(2, 7, V)
    labels = torch.randint(0, V, (2, 7))
    labels[0, 2] = -100
    for n_items in (None, torch.tensor(9)):
        # 3-D: unchanged behaviour
        want = stock(logits, labels, V, num_items_in_batch = n_items)
        got = patched(logits, labels, V, num_items_in_batch = n_items)
        torch.testing.assert_close(got, want)
        # flattened: the form remote MTP heads use
        flat_logits, flat_labels = logits.view(-1, V), labels.view(-1)
        want = stock(flat_logits, flat_labels, V, num_items_in_batch = n_items)
        got = patched(flat_logits, flat_labels, V, num_items_in_batch = n_items)
        torch.testing.assert_close(got, want)
        # A non-default ignore_index takes the F.cross_entropy fallback, flat and 3-D.
        other = labels.masked_fill(labels == -100, -1)
        for lg, lb in ((logits, other), (logits.view(-1, V), other.view(-1))):
            want = stock(lg, lb, V, num_items_in_batch = n_items, ignore_index = -1)
            got = patched(lg, lb, V, num_items_in_batch = n_items, ignore_index = -1)
            torch.testing.assert_close(got, want)
    print("OK")
    """
)


def test_flattened_logits_match_the_stock_loss():
    proc = subprocess.run([sys.executable, "-c", _RUNNER], capture_output = True, text = True, timeout = 600)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-4000:]
