# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Numerics for the GKD divergence: does the student actually move to the teacher?

Behaviour over steps, not construction. Every training assertion also reports a
divergence-to-teacher number, because a loss can fall while the student drifts
away from the target (reverse KL does exactly that to forward KL).

Real MLX required, so this skips on Linux; the guard/config/preflight half runs
under the torch shim in test_mlx_gkd_guards.py. Synthetic tensors only -- no
weights, no network, no Dataset.map.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is only available on Apple Silicon")
nn = pytest.importorskip("mlx.nn", reason="MLX is only available on Apple Silicon")
optim = pytest.importorskip("mlx.optimizers", reason="MLX is only available on Apple Silicon")

from unsloth_zoo.mlx.distill import (
    DEFAULT_CHUNK_SIZE,
    generalized_jsd_loss,
)


VOCAB, BATCH, SEQ, HIDDEN = 512, 4, 24, 64
STEPS = 8


def _fixture():
    mx.random.seed(0)
    teacher_logits = mx.random.normal((BATCH, SEQ, VOCAB)) * 2.0
    labels = mx.concatenate(
        [mx.full((BATCH, SEQ // 2), -100), mx.zeros((BATCH, SEQ - SEQ // 2))], axis=1,
    ).astype(mx.int32)
    ids = mx.random.randint(0, VOCAB, (BATCH, SEQ))
    return teacher_logits, labels, ids


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB)

    def __call__(self, ids):
        return self.head(self.emb(ids))


def _new_student():
    mx.random.seed(0)
    student = _Student()
    mx.eval(student.parameters())
    return student


def _masked_divergence(student_logits, teacher_logits, labels, direction):
    """KL to the teacher on supervised positions only."""
    student_lp = nn.log_softmax(student_logits, axis=-1)
    teacher_lp = nn.log_softmax(teacher_logits, axis=-1)
    if direction == "forward":       # KL(teacher || student)
        per = mx.exp(teacher_lp) * (teacher_lp - student_lp)
    else:                            # KL(student || teacher)
        per = mx.exp(student_lp) * (student_lp - teacher_lp)
    per = per.sum(axis=-1)
    mask = (labels != -100)
    return float((per * mask).sum() / mx.maximum(mask.sum(), 1))


def _train(student, teacher_logits, ids, labels, beta, steps=STEPS, chunk_size=DEFAULT_CHUNK_SIZE):
    grad_fn = nn.value_and_grad(
        student,
        lambda m, i, t, l: generalized_jsd_loss(m(i), t, l, beta=beta, chunk_size=chunk_size),
    )
    opt = optim.Adam(learning_rate=5e-3)
    losses = []
    for _ in range(steps):
        loss, grads = grad_fn(student, ids, teacher_logits, labels)
        opt.update(student, grads)
        mx.eval(student.parameters(), opt.state)
        losses.append(float(loss))
    return losses


@pytest.mark.parametrize("beta,direction", [(0.0, "forward"), (0.5, "forward"), (1.0, "reverse")])
def test_student_moves_toward_teacher(beta, direction):
    """Loss falls AND the corresponding divergence to the teacher falls."""
    teacher_logits, labels, ids = _fixture()
    student = _new_student()
    before = _masked_divergence(student(ids), teacher_logits, labels, direction)

    losses = _train(student, teacher_logits, ids, labels, beta)

    after = _masked_divergence(student(ids), teacher_logits, labels, direction)
    assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)), (
        f"beta={beta} loss not strictly decreasing: {losses}"
    )
    assert after < before, (
        f"beta={beta}: loss fell but {direction} KL to teacher rose "
        f"({before:.5f} -> {after:.5f})"
    )


def test_chunked_matches_unchunked_numerically():
    """Chunking is the default, so it must be exactly the naive computation."""
    teacher_logits, labels, ids = _fixture()
    student = _new_student()
    logits = student(ids)

    naive = generalized_jsd_loss(logits, teacher_logits, labels, beta=0.5, chunk_size=0)
    for chunk in (1, 4, 8, SEQ - 1, SEQ, SEQ + 5):
        chunked = generalized_jsd_loss(logits, teacher_logits, labels, beta=0.5, chunk_size=chunk)
        assert abs(float(naive) - float(chunked)) < 1e-5, (
            f"chunk_size={chunk} disagreed with the unchunked loss: "
            f"{float(chunked):.8f} vs {float(naive):.8f}"
        )


def test_chunked_and_unchunked_train_identically():
    """Equal values are not enough -- the gradients must agree too."""
    teacher_logits, labels, ids = _fixture()
    naive = _train(_new_student(), teacher_logits, ids, labels, 0.5, chunk_size=0)
    chunked = _train(_new_student(), teacher_logits, ids, labels, 0.5, chunk_size=8)
    worst = max(abs(a - b) for a, b in zip(naive, chunked))
    assert worst < 1e-5, f"chunked training diverged from naive by {worst:.3e}"


def test_labels_mask_is_honoured():
    teacher_logits, labels, ids = _fixture()
    student = _new_student()
    logits = student(ids)

    all_masked = mx.full((BATCH, SEQ), -100).astype(mx.int32)
    masked_loss = generalized_jsd_loss(logits, teacher_logits, all_masked, beta=0.5)
    assert float(masked_loss) == 0.0
    assert bool(mx.isfinite(masked_loss).item()), "denominator guard missing: 0/0"

    unmasked = generalized_jsd_loss(logits, teacher_logits, None, beta=0.5)
    half = generalized_jsd_loss(logits, teacher_logits, labels, beta=0.5)
    assert abs(float(half) - float(unmasked)) > 1e-6, (
        "masking half the positions changed nothing; the mask is being ignored"
    )


def test_identical_distributions_give_zero_loss():
    teacher_logits, labels, _ = _fixture()
    for beta in (0.0, 0.5, 1.0):
        loss = generalized_jsd_loss(teacher_logits, teacher_logits, labels, beta=beta)
        assert abs(float(loss)) < 1e-5, f"beta={beta}: self-distillation loss {float(loss)}"


def test_temperature_is_wired():
    teacher_logits, labels, ids = _fixture()
    logits = _new_student()(ids)
    at_one = float(generalized_jsd_loss(logits, teacher_logits, labels, beta=0.5, temperature=1.0))
    at_two = float(generalized_jsd_loss(logits, teacher_logits, labels, beta=0.5, temperature=2.0))
    assert at_one != at_two
    assert at_two < at_one, "softening both distributions should reduce the divergence"


def test_beta_endpoints_are_asymmetric():
    """beta=0 and beta=1 are different divergences, not the same number."""
    teacher_logits, labels, ids = _fixture()
    logits = _new_student()(ids)
    forward = float(generalized_jsd_loss(logits, teacher_logits, labels, beta=0.0))
    reverse = float(generalized_jsd_loss(logits, teacher_logits, labels, beta=1.0))
    assert abs(forward - reverse) > 1e-4


def test_loss_survives_mx_compile():
    """MLXTrainer compiles the step with inputs=state, outputs=state."""
    teacher_logits, labels, ids = _fixture()

    def run(compiled):
        student = _new_student()
        opt = optim.Adam(learning_rate=5e-3)
        grad_fn = nn.value_and_grad(
            student, lambda m, i, t, l: generalized_jsd_loss(m(i), t, l, beta=0.5),
        )
        state = [student.state, opt.state, mx.random.state]

        def step(i, t, l):
            loss, grads = grad_fn(student, i, t, l)
            opt.update(student, grads)
            return loss

        if compiled:
            step = mx.compile(step, inputs=state, outputs=state)
        out = []
        for _ in range(STEPS):
            loss = step(ids, teacher_logits, labels)
            mx.eval(state)
            out.append(float(loss))
        return out

    plain, compiled = run(False), run(True)
    worst = max(abs(a - b) for a, b in zip(plain, compiled))
    assert worst == 0.0, f"mx.compile changed the trajectory by {worst:.3e}"
