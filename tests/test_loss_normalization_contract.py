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

"""The grad-accumulation loss-normalisation contract in _unsloth_get_batch_samples.

num_items_in_batch is set from the forward signature, but training_step divides by
grad-accum off self.model_accepts_loss_kwargs. A count returned while that flag is
False normalises twice, since TRL's chunked_nll and our own fused CE each divide by
it, so loss and gradients end up scaled by 1/GA. The guard mirrors stock
Trainer._get_num_items_in_batch: count only when the flag is True or a
compute_loss_func exists. Those losses fall back to a mean when it is None, leaving
training_step's /GA as the single correct normalisation, with no per-model or
per-trainer knowledge needed.

CPU only, no model downloads.
"""

import inspect
import re

import pytest


def _loss_utils():
    return pytest.importorskip("unsloth_zoo.loss_utils")


def _source():
    mod = _loss_utils()
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    if fn is None: pytest.skip("_unsloth_get_batch_samples not present")
    return inspect.getsource(fn)


def test_guard_is_present_and_mirrors_the_stock_predicate():
    src = _source()
    assert "model_accepts_loss_kwargs" in src, (
        "the grad-accum guard is gone; a count returned while the flag is False "
        "gets normalised twice and silently scales gradients by 1/GA"
    )
    assert "compute_loss_func" in src, (
        "the guard must exempt compute_loss_func, which suppresses the division "
        "upstream exactly as stock transformers does"
    )


def test_guard_only_suppresses_never_flips_the_flag():
    """Flipping the flag True would break the callers that disable it on purpose.

    TRL trainers set model_accepts_loss_kwargs = False in __init__ to enable the
    grad-accum scaling, and the membership is version dependent: seven on 0.22.2
    and 0.24.0, thirteen on 1.9.2, and RewardTrainer does not assign it before 1.x,
    so no fixed trainer list is safe. transformers 5.5.0 adds another: its own
    Trainer.__init__ sets the flag False for the DeepSpeed sequence-parallel
    backend. Suppressing the count is right for all of them; assigning the flag is
    not, and the check is module wide so a helper cannot smuggle it back in.
    """
    src = inspect.getsource(_loss_utils())
    assert not re.search(r"\.model_accepts_loss_kwargs\s*=[^=]", src), (
        "loss_utils must not assign model_accepts_loss_kwargs. Suppressing the "
        "count is enough, and assigning it would override trainers that set it "
        "False deliberately, plus the DeepSpeed sequence-parallel opt-out that "
        "transformers 5.5.0 sets in Trainer.__init__"
    )


def test_stock_trainer_gates_its_count_on_the_same_predicate():
    """If upstream changes this, the guard is mirroring the wrong condition."""
    from transformers import Trainer

    fn = getattr(Trainer, "_get_num_items_in_batch", None)
    if fn is None: pytest.skip("no _get_num_items_in_batch on this transformers")
    try: src = inspect.getsource(fn)
    except (OSError, TypeError): pytest.skip("source unavailable")
    assert "model_accepts_loss_kwargs" in src and "compute_loss_func" in src, (
        "stock transformers no longer gates its token count on these, so the "
        "guard in _unsloth_get_batch_samples is mirroring a stale predicate"
    )


def test_training_step_still_divides_on_the_same_flag():
    from transformers import Trainer

    try: src = inspect.getsource(Trainer.training_step)
    except (OSError, TypeError): pytest.skip("source unavailable")
    assert "model_accepts_loss_kwargs" in src
    assert "num_items_in_batch" in src


def test_losses_fall_back_to_a_mean_when_the_count_is_none():
    """The guard is only safe because every such loss handles None as a mean."""
    ce = pytest.importorskip("unsloth_zoo.fused_losses.cross_entropy_loss")
    fn = getattr(ce, "unsloth_fused_ce_loss", None)
    if fn is None: pytest.skip("unsloth_fused_ce_loss not present")
    doc = (fn.__doc__ or "") + inspect.getsource(fn)
    assert "n_items" in doc, (
        "unsloth_fused_ce_loss no longer takes n_items; re-check the guard"
    )

    trl_sft = pytest.importorskip("trl.trainer.sft_trainer")
    chunked = getattr(trl_sft, "_chunked_cross_entropy_loss", None)
    if chunked is None: pytest.skip("this TRL has no chunked CE")
    src = inspect.getsource(chunked)
    assert "num_items_in_batch is None" in src, (
        "TRL's chunked CE no longer branches on a None count. The guard relies on "
        "it falling back to a mean; re-measure GA-invariance before trusting it."
    )


def test_get_batch_samples_still_returns_the_documented_pair():
    src = _source().strip()
    assert src.endswith("return batch_samples, num_items_in_batch"), (
        "unsloth/models/_utils.py asserts on this exact shape at patch time"
    )


# Numerical: the guard is what makes accumulated gradients match a single batch.

def _tiny_model():
    """A token normalising loss: sum / count when a count is given, mean otherwise.

    That is what unsloth_fused_ce_loss and TRL's _chunked_cross_entropy_loss do. The
    "CausalLM" class name and **kwargs forward make _unsloth_get_batch_samples count.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyForCausalLM(nn.Module):
        accepts_loss_kwargs = False

        def __init__(self, vocab = 11, hidden = 6):
            super().__init__()
            torch.manual_seed(0)
            self.embed = nn.Embedding(vocab, hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias = False)

        def forward(self, input_ids, labels = None, num_items_in_batch = None, **kwargs):
            logits = self.lm_head(self.embed(input_ids))
            flat_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))
            flat_labels = labels[..., 1:].reshape(-1)
            total = F.cross_entropy(flat_logits, flat_labels, ignore_index = -100, reduction = "sum")
            n = (flat_labels != -100).sum() if num_items_in_batch is None else num_items_in_batch
            return total / n

    return TinyForCausalLM()


def _fake_trainer(model, accepts, compute_loss_func = None):
    from transformers.training_args import ParallelMode

    class Args:
        average_tokens_across_devices = False
        n_gpu = 1
        world_size = 1
        parallel_mode = ParallelMode.NOT_DISTRIBUTED

    class Accelerator:
        parallelism_config = None

    class Trainer:
        pass

    t = Trainer()
    t.model = model
    t.args = Args()
    t.accelerator = Accelerator()
    t.model_accepts_loss_kwargs = accepts
    t.compute_loss_func = compute_loss_func
    # transformers 5.x reads this INSIDE the try that counts labels, and the except
    # swallows AttributeError. A stand-in without it therefore does not measure stock's
    # decision at all: stock raises on the first list comprehension, returns None, and a
    # differential test reads that as "stock declined to count". Every real Trainer sets
    # it in __init__. True is the value for this fixture's model, which shifts labels
    # internally like any causal LM, which is exactly the case the flag selects.
    t._loss_shifts_labels = True
    return t


def _microbatches(n, tokens_per_row):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(7)
    out = []
    for _ in range(n):
        ids = torch.randint(0, 11, (2, tokens_per_row + 1), generator = g)
        out.append({"input_ids": ids, "labels": ids.clone()})
    return out


def _accumulate(model, microbatches, count, grad_accum, accepts_loss_kwargs):
    """Sum gradients the way Trainer.training_step does (4.57.6 and 5.5.0 are byte
    identical here):

        if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) \
                and self.compute_loss_func is None:
            loss = loss / self.current_gradient_accumulation_steps
    test_training_step_still_divides_on_the_same_flag guards that drift.
    """
    model.zero_grad(set_to_none = True)
    for batch in microbatches:
        loss = model(**batch, num_items_in_batch = count)
        if not accepts_loss_kwargs or count is None:
            loss = loss / grad_accum
        loss.backward()
    return model.lm_head.weight.grad.clone()


def test_accumulated_gradient_matches_the_single_batch_gradient():
    """GA invariance, the property the whole guard exists to restore.

    Four microbatches of equal token count versus one batch holding all of them.
    Returning the count while the flag is False divides once in the loss and again
    in training_step, so the gradient lands at exactly 1/GA. CPU only.
    """
    mod = _loss_utils()
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    if fn is None: pytest.skip("_unsloth_get_batch_samples not present")
    torch = pytest.importorskip("torch")

    grad_accum = 4
    model = _tiny_model()
    microbatches = _microbatches(grad_accum, 5)
    single = {
        "input_ids": torch.cat([b["input_ids"] for b in microbatches]),
        "labels":    torch.cat([b["labels"]    for b in microbatches]),
    }
    reference = _accumulate(model, [single], None, 1, False)

    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, guarded = fn(_fake_trainer(model, False), iter(microbatches), grad_accum)
    assert guarded is None, "the guard must suppress the count when the flag is False"
    after = _accumulate(model, microbatches, guarded, grad_accum, False)
    assert torch.allclose(after, reference, atol = 1e-6), (
        "accumulated gradients no longer match the single batch gradient"
    )

    # What the unguarded code returned, run through the same training_step rule.
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, unguarded = fn(_fake_trainer(model, True), iter(microbatches), grad_accum)
    assert unguarded is not None
    before = _accumulate(model, microbatches, unguarded, grad_accum, False)
    ratio = (before.norm() / reference.norm()).item()
    assert abs(ratio - 1.0 / grad_accum) < 1e-4, (
        f"expected the double normalised gradient at 1/GA, measured x{ratio}"
    )



def _stock_counts_shifted_labels(stock):
    """Does the installed transformers count over `labels[..., 1:]` rather than `labels`?

    Read off the source rather than the version, so a backport or a fork lands on the
    right branch and this does not need a table of version numbers.
    """
    import inspect
    try:
        return "_loss_shifts_labels" in inspect.getsource(stock)
    except (OSError, TypeError):
        return False

def test_guard_returns_a_count_exactly_when_stock_transformers_would():
    """Same decision as Trainer._get_num_items_in_batch, for every flag pairing, and
    the answer to "why not keep the count and normalise per token across
    microbatches": stock never emits a count while the flag is False, because
    training_step would then divide by GA on top of it.
    """
    from transformers import Trainer

    mod = _loss_utils()
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    if fn is None: pytest.skip("_unsloth_get_batch_samples not present")
    stock = getattr(Trainer, "_get_num_items_in_batch", None)
    if stock is None: pytest.skip("no _get_num_items_in_batch on this transformers")
    torch = pytest.importorskip("torch")

    model = _tiny_model()
    microbatches = _microbatches(2, 5)
    device = torch.device("cpu")
    for accepts in (True, False):
        for loss_func in (None, lambda *a, **k: None):
            trainer = _fake_trainer(model, accepts, loss_func)
            mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
            _, ours = fn(trainer, iter(microbatches), 2, device)
            theirs = stock(trainer, microbatches, device)
            assert (ours is None) == (theirs is None), (
                f"accepts={accepts} compute_loss_func={loss_func is not None}: "
                f"we returned {ours!r} where stock returned {theirs!r}"
            )
            # Where stock counts over labels[..., 1:] the COUNT has to agree too, not
            # just whether one was produced. transformers adopted that rule in 5.x
            # (position 0 of a row is never a prediction target for a causal LM); zoo
            # has always counted that way, so before 5.x the two legitimately differ by
            # one token per row and only the nullity above is a shared contract.
            if _stock_counts_shifted_labels(stock) and ours is not None:
                assert int(ours) == int(theirs), (
                    f"accepts={accepts} compute_loss_func={loss_func is not None}: "
                    f"counted {int(ours)} where stock counted {int(theirs)}"
                )
