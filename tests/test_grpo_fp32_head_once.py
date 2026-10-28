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

"""With UNSLOTH_VLLM_FP32_LM_HEAD and the recompute scorer, the mini-batches grpo_accumulated_loss keeps until the loss share one fp32 head copy, not one each."""
import functools
import types

import pytest
import torch
from torch.overrides import TorchFunctionMode

import unsloth_zoo.rl_replacements as _rl
from unsloth_zoo.temporary_patches.common import UNSLOTH_COMPILE_DISABLE

HIDDEN, VOCAB, ROWS = 16, 64, 4


class _Model(torch.nn.Module):
    def __init__(self, raw_logits = False):
        super().__init__()
        self.raw_logits = raw_logits
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB + 1, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False).to(torch.bfloat16).requires_grad_(False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        if self.raw_logits:
            # A forward that ignores UNSLOTH_RETURN_HIDDEN_STATES: no projection through the head follows.
            return types.SimpleNamespace(logits = hidden.to(torch.bfloat16) @ self.head.weight.t())
        return types.SimpleNamespace(logits = hidden)


@pytest.mark.parametrize("fp32_head, raw_logits, expected", [(True, False, 1), (False, False, 0), (True, True, 0)])
def test_mini_batches_share_one_fp32_head(monkeypatch, fp32_head, raw_logits, expected):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "1")
    for name in ("chunked_hidden_states_selective_log_softmax", "chunked_hidden_states_selective_log_softmax_recompute"):
        monkeypatch.setattr(_rl, name, functools.partial(getattr(_rl, name), fp32_lm_head = fp32_head))
    seen = []

    class _Loss:
        @staticmethod
        def apply(new_logps, *args):
            seen.append(new_logps.shape[0])
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = ROWS, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = _Model(raw_logits),
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False,
        beta = 0.0,
        _autocast_dtype = torch.bfloat16,
    )
    input_ids = torch.randint(1, VOCAB, (ROWS, 6), generator = torch.Generator().manual_seed(0))
    heads, casts = {}, []

    def pack(t):
        if t.numel() == VOCAB * HIDDEN:
            heads[t.untyped_storage().data_ptr()] = t.dtype
        return t

    class _CountHeadCasts(TorchFunctionMode):
        def __torch_function__(self, func, types, args = (), kwargs = None):
            out = func(*args, **(kwargs or {}))
            if func is torch.Tensor.to and args[0].shape == (VOCAB, HIDDEN) and args[0].dtype != torch.float32 \
                    and isinstance(out, torch.Tensor) and out.dtype == torch.float32:
                casts.append(out)
            return out

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), _CountHeadCasts():
        _rl.grpo_accumulated_loss(
            trainer, input_ids, torch.ones_like(input_ids), 3, torch.ones(ROWS, 3, dtype = torch.long),
            torch.zeros(ROWS), None, None, loss_type = "dapo", num_items_in_batch = 3 * ROWS,
        )
    assert seen == [ROWS]
    fp32_copies = [d for d in heads.values() if d == torch.float32]
    assert len(fp32_copies) == expected, heads
    assert len(casts) == expected, len(casts)


def _inductor_builds():
    try:
        torch.compile(lambda x: x + 1, backend = "inductor")(torch.ones(2))
        return True
    except Exception:
        return False


@pytest.mark.parametrize("train_head", [False, True])
def test_default_scorer_keeps_sharing_the_head_parameter(monkeypatch, train_head):
    # A pre-cast fp32 head made the compiled scorer save a bf16 head copy per mini-batch.
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0")
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "1")
    for name in ("chunked_hidden_states_selective_log_softmax", "chunked_hidden_states_selective_log_softmax_recompute"):
        monkeypatch.setattr(_rl, name, functools.partial(getattr(_rl, name), fp32_lm_head = True))

    class _Loss:
        @staticmethod
        def apply(new_logps, *args):
            return (new_logps.sum(),) + (torch.zeros(()),) * 5

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    model = _Model()
    model.head.requires_grad_(train_head)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = ROWS, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False,
        beta = 0.0,
        _autocast_dtype = torch.bfloat16,
    )
    input_ids = torch.randint(1, VOCAB, (ROWS, 6), generator = torch.Generator().manual_seed(0))
    heads = {}

    def pack(t):
        if t.numel() == VOCAB * HIDDEN:
            heads[t.untyped_storage().data_ptr()] = t.dtype
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = _rl.grpo_accumulated_loss(
            trainer, input_ids, torch.ones_like(input_ids), 3, torch.ones(ROWS, 3, dtype = torch.long),
            torch.zeros(ROWS), None, None, loss_type = "dapo", num_items_in_batch = 3 * ROWS,
        )[0]
    loss.backward()
    if not UNSLOTH_COMPILE_DISABLE and not _inductor_builds():
        pytest.skip("Inductor cannot build here, so the compiled scorer ran its eager fallback")
    if UNSLOTH_COMPILE_DISABLE:
        assert list(heads.values()) == [torch.float32], heads
    else:
        assert heads == {model.head.weight.untyped_storage().data_ptr(): torch.bfloat16}, heads


@pytest.mark.parametrize("autocast_off", ["dtype_none", "disabled"])
def test_compiled_scorer_without_autocast_shares_one_fp32_head(monkeypatch, autocast_off):
    # Without autocast the compiled scorer's backward keeps its own fp32 head per call unless the head is pre-cast.
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_LOGPROB_RECOMPUTE", "0")
    monkeypatch.setenv("UNSLOTH_VLLM_FP32_LM_HEAD", "1")
    for name in ("chunked_hidden_states_selective_log_softmax", "chunked_hidden_states_selective_log_softmax_recompute"):
        monkeypatch.setattr(_rl, name, functools.partial(getattr(_rl, name), fp32_lm_head = True))

    class _Loss:
        @staticmethod
        def apply(new_logps, *args):
            return (new_logps.sum(),) + (torch.zeros(()),) * 5

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    model = _Model()
    model.head.requires_grad_(True)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        use_vllm = False,
        beta = 0.0,
        _autocast_dtype = None if autocast_off == "dtype_none" else torch.bfloat16,
    )
    if autocast_off == "disabled":
        trainer._autocast_enabled = False
    input_ids = torch.randint(1, VOCAB, (ROWS, 6), generator = torch.Generator().manual_seed(0))
    heads = {}

    def pack(t):
        if t.numel() == VOCAB * HIDDEN:
            heads[t.untyped_storage().data_ptr()] = t.dtype
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        _rl.grpo_accumulated_loss(
            trainer, input_ids, torch.ones_like(input_ids), 3, torch.ones(ROWS, 3, dtype = torch.long),
            torch.zeros(ROWS), None, None, loss_type = "dapo", num_items_in_batch = 3 * ROWS,
        )
    if not UNSLOTH_COMPILE_DISABLE and not _inductor_builds():
        pytest.skip("Inductor cannot build here, so the compiled scorer ran its eager fallback")
    fp32_heads = [p for p, d in heads.items() if d == torch.float32]
    assert len(fp32_heads) == 1, heads
