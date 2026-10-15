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

"""GKD divergence numerics on real MLX (the mlx-cpu lane runs them on Linux)."""

import pytest

mx = pytest.importorskip("mlx.core", reason="needs real MLX")
nn = pytest.importorskip("mlx.nn", reason="needs real MLX")
optim = pytest.importorskip("mlx.optimizers", reason="needs real MLX")

# The torch shim installs process-wide while another module is being collected, so the
# imports above can succeed against it; these numerics need real MLX (the mlx-cpu lane).
from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip("needs real MLX, the torch shim is installed", allow_module_level = True)

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
    """Loss falls AND the divergence to the teacher falls: reverse KL can lower
    the loss while forward KL rises."""
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


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_half_precision_logits_are_reduced_in_float32(beta):
    """bf16 log-probs made the beta=0.5 mixture loss ~9x off near convergence."""
    teacher_logits, labels, _ = _fixture()
    student_logits = teacher_logits + 0.05 * mx.random.normal(teacher_logits.shape, key=mx.random.key(1))
    s16, t16 = student_logits.astype(mx.bfloat16), teacher_logits.astype(mx.bfloat16)
    reference = generalized_jsd_loss(s16.astype(mx.float32), t16.astype(mx.float32), labels, beta=beta)
    got = generalized_jsd_loss(s16, t16, labels, beta=beta)
    assert got.dtype == mx.float32
    assert abs(float(got) - float(reference)) <= 1e-3 * abs(float(reference))


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
    teacher_logits, labels, ids = _fixture()
    logits = _new_student()(ids)
    forward = float(generalized_jsd_loss(logits, teacher_logits, labels, beta=0.0))
    reverse = float(generalized_jsd_loss(logits, teacher_logits, labels, beta=1.0))
    assert abs(forward - reverse) > 1e-4


def test_loss_survives_mx_compile():
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
    # Bit-identical on CPU; Metal fuses the compiled graph differently (8.9e-8 seen on macOS 26).
    assert worst <= 1e-6 * max(abs(v) for v in plain), f"mx.compile changed the trajectory by {worst:.3e}"


@pytest.mark.parametrize("with_labels", [False, True])
def test_trains_the_same_positions_as_cross_entropy(with_labels):
    """Labels beyond a row's length are padding, not targets."""
    from types import SimpleNamespace
    from unsloth_zoo.mlx.distill import build_gkd_loss_fn
    from unsloth_zoo.mlx.utils import make_baseline_loss_fn

    _, _, ids = _fixture()
    lengths = mx.array([[0, SEQ], [2, SEQ - 5], [4, SEQ // 2], [0, SEQ - 1]])
    labels = mx.concatenate([mx.full((BATCH, 3), -100), ids[:, 3:]], axis=1) if with_labels else None
    student, teacher = _new_student(), _new_student()
    args = SimpleNamespace(gkd_chunk_size=5, max_seq_length=SEQ)
    gkd = build_gkd_loss_fn(teacher, args, VOCAB, BATCH, 0, 1 << 40)
    _, gkd_ntoks = gkd(student, ids, lengths, labels)
    _, ce_ntoks = make_baseline_loss_fn()(student, ids, lengths, labels)
    assert int(gkd_ntoks) == int(ce_ntoks)


@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("chunk", [0, 5])
def test_analytic_gradient_matches_autodiff(beta, temperature, chunk):
    """The loss carries a hand-written VJP; check it against autodiff of the math."""
    from unsloth_zoo.mlx.distill import _jsd_per_position

    teacher_logits, labels, _ = _fixture()
    student_logits = mx.random.normal(teacher_logits.shape, key=mx.random.key(3)) * 2.0
    mask = (labels != -100).astype(mx.float32)

    def reference(s):
        per = _jsd_per_position(s, teacher_logits, beta, temperature)
        return (per * mask).sum() / mx.maximum(mask.sum(), 1)

    def ours(s):
        return generalized_jsd_loss(s, teacher_logits, labels, beta=beta,
                                    temperature=temperature, chunk_size=chunk)

    ref_value, ref_grad = mx.value_and_grad(reference)(student_logits)
    value, grad = mx.value_and_grad(ours)(student_logits)
    assert abs(float(value) - float(ref_value)) <= 1e-5 * max(1.0, abs(float(ref_value)))
    assert float(mx.abs(grad - ref_grad).max()) <= 1e-6
