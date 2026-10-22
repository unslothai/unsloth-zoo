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

"""The fused and UNSLOTH_RETURN_LOGITS branches of a rewritten forward compute the same loss."""
import ast
import os
import types

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.fused_losses.ast_rewriter import (  # noqa: E402
    rewrite_forward_source,
    rewrite_forward_source_spliced,
)
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_count_aware_cross_entropy  # noqa: E402
from test_fused_num_items_fallback import MAMBA_5_16_1, T5_ENC_DEC, XGLM_KWARGLESS  # noqa: E402

VOCAB, HIDDEN = 11, 8


def _fused_stub(hidden_states, lm_head, labels, vocab_size = None, **kwargs):
    n_items = kwargs.pop("num_items_in_batch", None)
    if n_items is None:
        n_items = kwargs.pop("n_items", None)
    else:
        kwargs.pop("n_items", None)
    shift = kwargs.pop("shift_labels", None)
    logits = lm_head(hidden_states).float()
    if shift is not None and not isinstance(shift, bool):
        logits, target = logits, shift
    elif shift is False:
        target = labels
    else:
        logits, target = logits[..., :-1, :], labels[..., 1:]
    summed = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction = "sum")
    return summed / (n_items if n_items is not None else (target != -100).sum())


class _Inner(torch.nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, *args, **kwargs):
        return (self.hidden_states,)


def _head(hidden_states):
    torch.manual_seed(0)
    head = torch.nn.Module()
    head.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
    head.backbone = head.decoder = head.model = _Inner(hidden_states)
    head.config = types.SimpleNamespace(vocab_size = VOCAB, pad_token_id = 0)
    head._shift_right = lambda labels: labels
    head.loss_function = None
    return head


def _run(source, return_logits, **kwargs):
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 6, HIDDEN)
    labels = torch.randint(0, VOCAB, (2, 6))
    labels[0, :3] = -100
    ns = {
        "torch": torch, "os": os, "CrossEntropyLoss": torch.nn.CrossEntropyLoss, "EMPTY_LOGITS": None,
        "unsloth_fused_lm_head_loss": _fused_stub,
        "unsloth_count_aware_cross_entropy": unsloth_count_aware_cross_entropy,
    }
    exec(source, ns)
    old = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1" if return_logits else "0"
    try:
        loss = ns["forward"](_head(hidden_states), input_ids = torch.zeros(2, 6, dtype = torch.long),
                             labels = labels, **kwargs)[0]
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS")
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = old
    return loss


@pytest.mark.parametrize("source", [MAMBA_5_16_1, T5_ENC_DEC], ids = ["mamba_shifted", "t5_aligned"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"num_items_in_batch": torch.tensor(17)},
        {"n_items": torch.tensor(17)},
        {"num_items_in_batch": torch.tensor(17), "shift_labels": torch.randint(0, VOCAB, (2, 6))},
    ],
    ids = ["no_count", "count", "n_items_alias", "caller_shift_labels"],
)
def test_legacy_branches_compute_the_same_loss(source, kwargs):
    new = rewrite_forward_source_spliced(source)[0]
    assert new is not None
    fused = _run(new, False, **dict(kwargs))
    unfused = _run(new, True, **dict(kwargs))
    torch.testing.assert_close(fused, unfused)


def test_injected_count_only_reaches_a_loss_function_when_present():
    new = rewrite_forward_source(XGLM_KWARGLESS)[0]
    assert new is not None
    tree = ast.parse(new)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "self.loss_function"]
    assert len(calls) == 1
    call = calls[0]
    assert not any(kw.arg == "num_items_in_batch" for kw in call.keywords), ast.unparse(call)
    starred = [ast.unparse(kw.value) for kw in call.keywords if kw.arg is None]
    assert starred == [
        "unsloth_loss_count_kwargs(self.loss_function, kwargs.get('num_items_in_batch', None) "
        "if kwargs.get('num_items_in_batch', None) is not None else kwargs.get('n_items', None))"
    ], ast.unparse(call)
    fused = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "unsloth_fused_lm_head_loss"]
    assert fused and any(kw.arg == "num_items_in_batch" for kw in fused[0].keywords)

    seen = {}

    def strict(logits, labels, vocab_size, pad_token_id = None):
        seen["strict"] = True
        return logits.sum() * 0

    def counting(logits, labels, vocab_size, pad_token_id = None, num_items_in_batch = None):
        seen["count"] = num_items_in_batch
        return logits.sum() * 0

    from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_loss_count_kwargs
    ns = {"torch": torch, "os": os, "EMPTY_LOGITS": None, "unsloth_fused_lm_head_loss": _fused_stub,
          "unsloth_loss_count_kwargs": unsloth_loss_count_kwargs}
    exec(new, ns)
    old = os.environ.get("UNSLOTH_RETURN_LOGITS")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    try:
        head = _head(torch.randn(2, 6, HIDDEN))
        head.loss_function = strict
        ns["forward"](head, input_ids = torch.zeros(2, 6, dtype = torch.long), labels = torch.zeros(2, 6, dtype = torch.long))
        ns["forward"](head, input_ids = torch.zeros(2, 6, dtype = torch.long), labels = torch.zeros(2, 6, dtype = torch.long),
                      num_items_in_batch = 9)
        head.loss_function = counting
        ns["forward"](head, input_ids = torch.zeros(2, 6, dtype = torch.long), labels = torch.zeros(2, 6, dtype = torch.long),
                      num_items_in_batch = 9)
    finally:
        if old is None:
            os.environ.pop("UNSLOTH_RETURN_LOGITS")
        else:
            os.environ["UNSLOTH_RETURN_LOGITS"] = old
    assert seen.get("strict") and seen.get("count") == 9


def test_sampler_unwraps_training_wrappers():
    from unsloth_zoo.loss_utils import _unwrap_training_wrappers

    class DistributedDataParallel(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    class OptimizedModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

    class HasModuleChild(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = torch.nn.Linear(1, 1)

    inner = torch.nn.Linear(1, 1)
    assert _unwrap_training_wrappers(DistributedDataParallel(OptimizedModule(inner))) is inner
    plain = HasModuleChild()
    assert _unwrap_training_wrappers(plain) is plain
