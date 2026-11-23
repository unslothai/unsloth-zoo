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

"""mtp_label_source="teacher": one teacher projection per batch (not one per depth), same loss and grads."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_leaves  # noqa: E402

from unsloth_zoo.speculative import losses  # noqa: E402
from unsloth_zoo.speculative.capture import effective_lm_head_weight, resolve_text_parts, run_frozen_target  # noqa: E402
from unsloth_zoo.speculative.losses import chunked_lm_loss, fastmtp_step_weights  # noqa: E402
from unsloth_zoo.speculative.mtp import build_mtp_head  # noqa: E402
from unsloth_zoo.speculative.trainer import MTPTrainingArguments, _shift_window, mtp_step_losses  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB = 97
MATMULS = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.matmul.default}


def _target():
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size = VOCAB, hidden_size = 64, intermediate_size = 128, num_attention_heads = 4,
        num_key_value_heads = 2, head_dim = 16, num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"], mtp_num_hidden_layers = 1,
    )
    return Qwen3_5ForCausalLM(config).to(DEVICE).eval()


def _per_depth_reference(head, parts, final_hidden, input_ids, attention_mask, data_labels, args):
    """The previous mtp_step_losses: the teacher argmax projected once per depth on its shifted window."""
    steps = head(final_hidden, input_ids, parts["embed_tokens"], attention_mask = attention_mask,
                 num_steps = args.mtp_num_steps)
    weights = fastmtp_step_weights(args.mtp_step_beta, len(steps))
    target_w = effective_lm_head_weight(parts["lm_head"])
    supervised = data_labels != -100
    total = final_hidden.new_zeros((), dtype = torch.float32)
    stats = []
    for k, (step, weight) in enumerate(zip(steps, weights)):
        hidden, labels = step["hidden"], step["labels"]
        w_draft = step.get("lm_head_weight")
        w_draft = target_w if w_draft is None else w_draft
        length = labels.shape[1]
        keep = _shift_window(supervised, k + 1, length, False)
        labels = torch.where(keep, labels, torch.full_like(labels, -100))
        teacher_hidden = _shift_window(final_hidden, k, length, 0).reshape(-1, final_hidden.shape[-1])
        per_row = target_w.shape[0] * target_w.element_size()
        rows = max(1, min(args.mtp_chunk_size, losses.LOGITS_BUDGET_BYTES // per_row))
        with torch.no_grad():
            ids = torch.cat([(c.to(target_w.dtype) @ target_w.T).argmax(-1) for c in teacher_hidden.split(rows)])
        labels = torch.where(labels != -100, ids.view_as(labels), labels)
        loss_sum, n_tokens, n_correct = chunked_lm_loss(
            hidden.reshape(-1, hidden.shape[-1]), w_draft, labels.reshape(-1), loss_type = args.mtp_loss_type,
            teacher_hidden = teacher_hidden,
            teacher_lm_head_weight = None if w_draft is target_w else target_w, chunk_size = args.mtp_chunk_size,
        )
        n = max(int(n_tokens), 1)
        total = total + weight * loss_sum / n
        stats.append((float(loss_sum.detach()) / n, int(n_tokens), int(n_correct)))
    return total, stats


class _OwnDraftHead(torch.nn.Module):
    """The real head drafting with its own lm_head copy, so the target's matrix is used by the teacher only."""

    def __init__(self, head, weight):
        super().__init__()
        self.head = head
        self.weight = weight

    def forward(self, *args, **kwargs):
        return [dict(step, lm_head_weight = self.weight) for step in self.head(*args, **kwargs)]


class _CountTargetMatmuls(TorchDispatchMode):
    def __init__(self, weight):
        super().__init__()
        self.ptr, self.count = weight.untyped_storage().data_ptr(), 0

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        if func in MATMULS and any(
            isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() == self.ptr for t in tree_leaves(args)
        ):
            self.count += 1
        return func(*args, **(kwargs or {}))


def _setup(tmp_path, loss_type, steps = 3, seq = 12):
    target = _target()
    parts = resolve_text_parts(target)
    torch.manual_seed(1)
    head = build_mtp_head(parts["config"], device = DEVICE)
    for p in head.parameters():
        torch.nn.init.normal_(p, std = 0.2)
    input_ids = torch.randint(1, VOCAB, (2, seq), device = DEVICE)
    labels = input_ids.clone()
    labels[:, :3] = -100
    mask = torch.ones_like(input_ids)
    mask[1, -2:] = 0
    labels[1, -2:] = -100
    final = run_frozen_target(target, input_ids, attention_mask = mask)["final"]
    args = MTPTrainingArguments(output_dir = str(tmp_path), mtp_num_steps = steps, mtp_label_source = "teacher",
                                mtp_loss_type = loss_type, report_to = "none", use_cpu = DEVICE == "cpu")
    return head, parts, final, input_ids, mask, labels, args


def _loss_and_grads(fn, head, *rest):
    head.zero_grad(set_to_none = True)
    loss, stats = fn(head, *rest)
    loss.backward()
    return loss.detach(), stats, [p.grad.clone() for p in head.parameters() if p.grad is not None]


@pytest.mark.parametrize("loss_type", ["ce", "kl_div", "lk_hybrid"])
@pytest.mark.parametrize("budget_rows", [None, 5])
def test_teacher_ids_once_match_per_depth_bitwise(tmp_path, monkeypatch, loss_type, budget_rows):
    head, parts, final, ids, mask, labels, args = _setup(tmp_path, loss_type)
    if budget_rows is not None:  # projection split into several row chunks
        element = parts["lm_head"].weight.element_size()
        monkeypatch.setattr(losses, "LOGITS_BUDGET_BYTES", budget_rows * VOCAB * element)
    rest = (parts, final, ids, mask, labels, args)
    ref_loss, ref_stats, ref_grads = _loss_and_grads(_per_depth_reference, head, *rest)
    loss, stats, grads = _loss_and_grads(mtp_step_losses, head, *rest)
    assert torch.equal(loss, ref_loss), (loss, ref_loss)
    assert stats == ref_stats
    assert len(grads) == len(ref_grads) > 0
    assert all(torch.equal(a, b) for a, b in zip(grads, ref_grads))


@pytest.mark.parametrize("steps", [1, 3, 5])
def test_one_teacher_projection_per_batch(tmp_path, steps):
    head, parts, final, ids, mask, labels, args = _setup(tmp_path, "ce", steps = steps)
    target_w = parts["lm_head"].weight
    draft = _OwnDraftHead(head, target_w.detach().clone())
    with _CountTargetMatmuls(target_w) as counter:
        mtp_step_losses(draft, parts, final, ids, mask, labels, args)
    assert counter.count == 1, counter.count  # was one per depth
