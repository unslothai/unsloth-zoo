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


def _fake_trainer(model, accepts, compute_loss_func = None, shifts = True):
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
    # transformers 5.x reads this inside the try that counts labels, and the except
    # swallows the AttributeError, so a stand-in without it makes stock look like it
    # declined to count. Every real Trainer sets it; True matches this fixture's model,
    # which shifts labels internally like any causal LM. shifts=None omits it, which is
    # the 4.x shape, where the loss_type lookup decides instead.
    if shifts is not None: t._loss_shifts_labels = shifts
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


def test_custom_loss_counts_tokens_without_model_forward_kwargs():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class ExplicitForCausalLM(type(_tiny_model())):
        def forward(self, input_ids):
            return self.lm_head(self.embed(input_ids))

    def compute_loss(logits, labels, num_items_in_batch = None):
        total = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)),
            labels[..., 1:].reshape(-1),
            reduction = "sum",
        )
        count = (labels[..., 1:] != -100).sum() if num_items_in_batch is None else num_items_in_batch
        return total / count

    model = ExplicitForCausalLM()
    batches = _microbatches(2, 5)
    batches[1]["labels"][:, 1:4] = -100
    full_input = torch.cat([batch["input_ids"] for batch in batches])
    full_labels = torch.cat([batch["labels"] for batch in batches])
    compute_loss(model(full_input), full_labels).backward()
    reference_grad = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none = True)

    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(model, False, compute_loss), iter(batches), len(batches),
    )
    for batch in batches:
        compute_loss(model(batch["input_ids"]), batch["labels"], count).backward()

    torch.testing.assert_close(model.lm_head.weight.grad, reference_grad)
    assert count == 14


@pytest.mark.parametrize("name", ["ExplicitForCausalLM", "MysteryHead"])
def test_a_module_carrying_no_shifted_label_signal_is_not_counted(name):
    """Scope: the widened gate needs a POSITIVE shifted-label signal, and a hand
    written nn.Module has none (no loss_type, and no Trainer flag before 5.x). It
    keeps today's behaviour rather than being guessed at from its class name.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    cls = type(name, (nn.Module,), {"forward": lambda self, input_ids, labels = None: None})
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(cls(), False, lambda *a, **k: None, shifts = None),
        iter(_microbatches(2, 5)), 2,
    )
    assert count is None


def test_a_real_loss_type_outranks_the_detector_fallback():
    """A masked LM whose forward the detector would match anyway still declines,
    because its own loss_type says the labels are not shifted."""
    pytest.importorskip("torch")

    class ExplicitForCausalLM(type(_tiny_model())):
        def forward(self, input_ids, labels = None): return None

    model = ExplicitForCausalLM()
    model.loss_type = "ForMaskedLM"
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(model, False, lambda *a, **k: None, shifts = None),
        iter(_microbatches(2, 5)), 2,
    )
    assert count is None


# Widening the gate to compute_loss_func widens WHO is eligible, not what the
# counting body means. That body counts shifted causal targets, and the helper is
# installed as a class attribute on transformers.trainer.Trainer
# (unsloth/models/_utils.py patch_gradient_accumulation_fix), so every Trainer
# subclass in the process inherits it, including ones holding a classification
# head. A *ForSequenceClassification carries one label per row: labels[..., 1:]
# would drop an entire example, return an empty count on a batch of one, and then
# fail to broadcast against attention_mask[..., 1:], which the enclosing
# `except Exception: raise RuntimeError(...)` turns into a dead training run.


def _tiny_sequence_classifier():
    """LlamaForSequenceClassification as _unsloth_get_batch_samples really sees it.

    unsloth sets LlamaModel.forward = LlamaModel_fast_forward class-wide, and the
    walk stops at the first "_fast_forward" qualname, so a classification model
    descends into its own backbone and is detected as causal. A fixture without that
    inner model would pass for the wrong reason.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    def LlamaModel_fast_forward(self, input_ids, **kwargs):
        raise AssertionError("the counter must never call the model")

    class Backbone(nn.Module):
        forward = LlamaModel_fast_forward

    class LlamaForSequenceClassification(nn.Module):
        def __init__(self, vocab = 11, hidden = 6, num_labels = 3):
            super().__init__()
            torch.manual_seed(0)
            self.model = Backbone()
            self.score = nn.Linear(hidden, num_labels, bias = False)

        def forward(self, input_ids, **kwargs):
            raise AssertionError("the counter must never call the model")

    return LlamaForSequenceClassification()


def test_the_classifier_fixture_really_is_detected_as_causal():
    """Guards the fixture itself: if the walk stops matching the patched backbone,
    the classification tests below would pass without exercising anything."""
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    mod._unsloth_get_batch_samples(
        _fake_trainer(_tiny_sequence_classifier(), True), iter([]), 0,
    )
    has_kwargs, *_ = mod.ALLOWED_NUM_ITEMS_IN_BATCH["LlamaForSequenceClassification"]
    assert has_kwargs, (
        "the fixture no longer reproduces the _fast_forward descent, so the "
        "classification tests are no longer testing the real situation"
    )


def _classification_batch(batch_size, seq_len = 5, with_attention_mask = True):
    """One label per row, which is what a sequence classifier is trained on."""
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(13)
    batch = {
        "input_ids": torch.randint(0, 11, (batch_size, seq_len), generator = g),
        "labels": torch.randint(0, 3, (batch_size,), generator = g),
    }
    if with_attention_mask:
        batch["attention_mask"] = torch.ones(batch_size, seq_len, dtype = torch.long)
    return batch


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("with_attention_mask", [True, False])
@pytest.mark.parametrize("custom_loss", [True, False])
def test_classification_labels_are_never_run_through_the_causal_counter(
    batch_size, with_attention_mask, custom_loss,
):
    """batch_size 1 is the empty slice, the no-mask case is the silent miscount.

    With a mask the shapes do not broadcast and the run dies; without one the count
    comes back quietly short by one example. custom_loss False is the pre-existing
    route, reached through has_kwargs alone once the walk has matched the patched
    backbone, so this fails on main too and not only on the widened gate.
    """
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    batch = _classification_batch(batch_size, with_attention_mask = with_attention_mask)
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(_tiny_sequence_classifier(), False,
                      (lambda *a, **k: None) if custom_loss else None),
        iter([batch]), 1,
    )
    assert count is None, (
        f"batch_size={batch_size} attention_mask={with_attention_mask} "
        f"custom_loss={custom_loss}: the causal token counter ran on "
        f"one-label-per-row targets and returned {count}"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_a_classifier_the_detector_never_matches_is_not_widened_into(batch_size):
    """The case the widened gate adds on its own: an unpatched backbone, so the walk
    matches nothing and has_kwargs stays False. Before #1217 it was never counted.
    Distinct from the fixture above, which main already admits via has_kwargs.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    class DebertaV2ForSequenceClassification(nn.Module):
        def forward(self, input_ids, **kwargs): return None

    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(DebertaV2ForSequenceClassification(), False, lambda *a, **k: None),
        iter([_classification_batch(batch_size)]), 1,
    )
    assert count is None, f"batch_size={batch_size}: counted {count} on row labels"


@pytest.mark.parametrize("labels", [
    [[1, 2, 3], [4, 5, 6]],          # a plain nested list
    "not labels at all",             # a string, which has no .ndim either
])
def test_labels_without_ndim_are_not_counted_instead_of_killing_the_run(labels):
    """The counting body is tensor arithmetic, and the enclosing
    `except Exception: raise RuntimeError(...)` turns a batch we merely cannot
    measure into a dead run. Stock swallows this and keeps training. Widening to
    compute_loss_func is what newly routes explicit-forward models here.
    """
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    class ExplicitForCausalLM(type(_tiny_model())):
        def forward(self, input_ids): return None

    batch = {"input_ids": torch.randint(0, 11, (2, 6)), "labels": labels}
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(ExplicitForCausalLM(), False, lambda *a, **k: None), iter([batch]), 1,
    )
    assert count is None, f"expected no count for uncountable labels, got {count}"


def test_numpy_labels_still_count():
    """The guard above keys on .ndim rather than torch.is_tensor, so a collator that
    hands back numpy keeps the behaviour it has today instead of quietly losing it."""
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    batch = {"labels": numpy.arange(12, dtype = "int64").reshape(2, 6)}
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(_tiny_model(), True), iter([batch]), 1,
    )
    assert count is not None and int(count) == 10, f"numpy labels stopped counting: {count}"


@pytest.mark.parametrize("via", ["flag", "loss_type"])
@pytest.mark.parametrize("head, loss_type, labels_shape, true_count", [
    ("ViTForImageClassification", "ForImageClassification", (4,), 4),
    ("SegformerForSemanticSegmentation", "ForSemanticSegmentation", (2, 8, 8), 128),
    # 2D, token aligned, column 0 supervised, and NOT shifted. Rank alone cannot tell
    # this from a causal batch, which is why eligibility asks what the loss does.
    ("BertForMaskedLM", "ForMaskedLM", (2, 6), 12),
])
def test_only_a_shifted_label_loss_may_enable_the_new_count(
    via, head, loss_type, labels_shape, true_count,
):
    """The new route needs a positive shifted-label signal, not a suffix list.

    Vision batches carry pixel_values and no attention_mask, so no shape check fires
    and labels[..., 1:] drops one entry per row: 3 of 4 images, 112 of 128 pixels.
    Masked LM is worse, since it looks exactly like a causal batch by rank and every
    column-0 target is dropped. Both branches of the signal are exercised: `flag` is
    transformers 5.x's Trainer._loss_shifts_labels, `loss_type` is the 4.x lookup.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    cls = type(head, (nn.Module,), {"forward": lambda self, *a, **kw: None})
    model = cls()
    model.loss_type = loss_type
    batch = {"labels": torch.randint(0, 3, labels_shape)}
    if head == "BertForMaskedLM":
        batch["input_ids"] = torch.randint(0, 11, labels_shape)
    else:
        batch["pixel_values"] = torch.randn(labels_shape[0], 3, 8, 8)

    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(model, False, lambda *a, **k: None,
                      shifts = False if via == "flag" else None),
        iter([batch]), 1,
    )
    assert count is None, (
        f"{head} via {via}: counted {count} where the true label count is {true_count}"
    )


def test_the_shifted_label_signal_survives_patch_loss_functions():
    """patch_loss_functions rewrites every causal entry in the very LOSS_MAPPING this
    signal reads, so an identity test against the stock function reads False for every
    causal model the moment unsloth loads, silently switching the count back off on any
    transformers without Trainer._loss_shifts_labels. The keys are snapshotted at
    import instead, so run the patch and check the answer does not move.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    mod = _loss_utils()
    if not hasattr(mod, "_loss_shifts_labels"): pytest.skip("signal helper not present")

    causal = type("LlamaForCausalLM", (nn.Module,), {"forward": lambda self, x: None})()
    causal.loss_type = "ForCausalLM"
    masked = type("BertForMaskedLM", (nn.Module,), {"forward": lambda self, x: None})()
    masked.loss_type = "ForMaskedLM"
    trainer = object()  # no _loss_shifts_labels, so the loss_type branch answers

    assert mod._loss_shifts_labels(trainer, causal, False) is True
    # Restore the mapping: it is process global, and leaving it rewritten changes what
    # every later test in the session sees.
    lu = pytest.importorskip("transformers.loss.loss_utils")
    saved = dict(lu.LOSS_MAPPING)
    try:
        mod.patch_loss_functions(lambda *a, **k: None, torch_compile = False)
        assert mod._loss_shifts_labels(trainer, causal, False) is True, (
            "patching LOSS_MAPPING switched the causal signal off, which silently "
            "disables the count for every custom loss on this transformers"
        )
        assert mod._loss_shifts_labels(trainer, masked, False) is False
    finally:
        lu.LOSS_MAPPING.clear()
        lu.LOSS_MAPPING.update(saved)


def _two_rank_trainer(model, peer_count, peer_flags):
    """This rank plus one simulated peer. The count path makes ONE collective, over
    [count, degenerate, all_short], so the stub answers with the peer's triple."""
    torch = pytest.importorskip("torch")
    t = _fake_trainer(model, True)
    t.args.average_tokens_across_devices = True
    t.args.world_size = 2
    calls = []

    def gather(x):
        calls.append(tuple(x.shape))
        peer = torch.tensor([peer_count, *peer_flags], dtype = x.dtype)
        return torch.cat([x.reshape(-1), peer])
    t.accelerator.gather = gather
    t.gather_calls = calls
    return t


def test_the_distributed_count_path_makes_exactly_one_collective():
    """A second gather would be a second sync point per accumulation group, and every
    rank must reach the same number of collectives or they deadlock."""
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    ids = torch.randint(0, 11, (2, 6))
    trainer = _two_rank_trainer(_tiny_model(), peer_count = 10, peer_flags = [0, 0])
    mod._unsloth_get_batch_samples(trainer, iter([{"input_ids": ids, "labels": ids.clone()}]), 1)
    assert trainer.gather_calls == [(3,)], (
        f"expected one gather of [count, degenerate, all_short], got {trainer.gather_calls}"
    )


def test_a_locally_short_rank_keeps_the_gathered_count():
    """all_short is rank-local. Once the count has been gathered it is global, so a
    rank holding only [B, 1] batches must keep it rather than drop a divisor its peers
    still use, or the two ranks normalise differently."""
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    short = {"input_ids": torch.randint(0, 11, (2, 1)), "labels": torch.randint(0, 11, (2, 1))}
    # Peer is healthy: 10 targets, neither flag set.
    trainer = _two_rank_trainer(_tiny_model(), peer_count = 10, peer_flags = [0, 0])
    _, count = mod._unsloth_get_batch_samples(trainer, iter([short]), 1)
    assert count is not None and int(count) == 10, (
        f"the short rank dropped the gathered count and got {count}"
    )


def test_an_all_short_world_still_declines_the_zero_count():
    """The other side of the same reduction: if EVERY rank is short the gathered total
    really is 0, and handing that back is a division by zero in the custom loss."""
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    short = {"input_ids": torch.randint(0, 11, (2, 1)), "labels": torch.randint(0, 11, (2, 1))}
    trainer = _two_rank_trainer(_tiny_model(), peer_count = 0, peer_flags = [0, 1])
    _, count = mod._unsloth_get_batch_samples(trainer, iter([short]), 1)
    assert count is None, f"an all-short world handed back {count}"


def test_one_degenerate_rank_makes_every_rank_decline():
    """A layout this cannot count on one rank cannot be counted on any: the peers must
    not keep a divisor that rank has thrown away, or they scale gradients differently."""
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    ids = torch.randint(0, 11, (2, 6))
    healthy = {"input_ids": ids, "labels": ids.clone()}
    # This rank is fine; the peer reports degenerate.
    trainer = _two_rank_trainer(_tiny_model(), peer_count = 10, peer_flags = [1, 0])
    _, count = mod._unsloth_get_batch_samples(trainer, iter([healthy]), 1)
    assert count is None, (
        f"a healthy rank kept {count} while its peer dropped the count"
    )


def test_a_short_microbatch_does_not_void_the_whole_accumulation_group():
    """A [B, 1] member contributes 0 targets; the group total is still positive.

    Voiding the group would hand None to a custom loss that then averages
    per-microbatch means, losing the GA invariance this whole path exists for, and
    would leave the short member taking a mean over zero targets.
    """
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    short = {"input_ids": torch.randint(0, 11, (2, 1)), "labels": torch.randint(0, 11, (2, 1))}
    ids = torch.randint(0, 11, (2, 6))
    normal = {"input_ids": ids, "labels": ids.clone()}
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(_tiny_model(), True), iter([short, normal]), 2,
    )
    assert count is not None and int(count) == 10, (
        f"expected the group total of 10 from the normal member, got {count}"
    )

    # Every member short means the total really is 0, and that must stay a no count.
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(_tiny_model(), True), iter([short, short]), 2,
    )
    assert count is None, f"an all-short group must not hand back 0, got {count}"


def test_token_classification_labels_are_not_silently_miscounted():
    """The quiet one. Token-classification labels are (B, T), unshifted, one per
    token, so nothing raises: they clear every shape check and the count comes back
    short by one per row, inflating loss and grads instead of failing.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    def LlamaModel_fast_forward(self, input_ids, **kwargs): return None

    class Backbone(nn.Module):
        forward = LlamaModel_fast_forward

    class LlamaForTokenClassification(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

        def forward(self, input_ids, **kwargs): return None

    batch = {
        "input_ids": torch.randint(0, 11, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype = torch.long),
        "labels": torch.randint(0, 3, (2, 5)),
    }
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(LlamaForTokenClassification(), True), iter([batch]), 1,
    )
    assert count is None, (
        f"token-classification targets were counted as shifted causal ones and "
        f"returned {count}, where the true unshifted target count is 10"
    )


def test_causal_model_with_a_single_label_column_degrades_to_no_count():
    """labels[..., 1:] on a one column tensor is empty, so the count would be 0 and
    every sum/count loss would divide by it."""
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    batch = {
        "input_ids": torch.randint(0, 11, (2, 1)),
        "labels": torch.randint(0, 11, (2, 1)),
    }
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(_tiny_model(), True), iter([batch]), 1,
    )
    assert count is None, f"expected no count for a single label column, got {count}"


@pytest.mark.parametrize("forward_has_kwargs", [True, False])
@pytest.mark.parametrize("custom_loss", [True, False])
@pytest.mark.parametrize("accepts", [True, False])
def test_eligibility_truth_table_for_a_causal_model(forward_has_kwargs, custom_loss, accepts):
    """A count exactly when the model is causal and something consumes it.

    accepts only drives the final guard, which exempts compute_loss_func, so the
    only combination it suppresses is a **kwargs forward with no custom loss.
    """
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    if forward_has_kwargs:
        model = _tiny_model()
    else:
        class ExplicitForCausalLM(type(_tiny_model())):
            def forward(self, input_ids):
                return self.lm_head(self.embed(input_ids))
        model = ExplicitForCausalLM()

    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(model, accepts, (lambda *a, **k: None) if custom_loss else None),
        iter(_microbatches(2, 5)), 2,
    )
    expected = (forward_has_kwargs or custom_loss) and (accepts or custom_loss)
    if expected:
        assert count is not None and int(count) == 20, f"expected 20 targets, got {count}"
    else:
        assert count is None, f"expected no count, got {count}"


def test_a_vlm_with_an_explicit_forward_and_a_custom_loss_still_counts():
    """The VLM branch sets the causal flag too, so widening reaches it as well."""
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    class TinyForConditionalGeneration(type(_tiny_model())):
        def forward(self, input_ids):
            return self.lm_head(self.embed(input_ids))

    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(TinyForConditionalGeneration(), False, lambda *a, **k: None),
        iter(_microbatches(2, 5)), 2,
    )
    assert count is not None and int(count) == 20, f"expected 20 targets, got {count}"


@pytest.mark.parametrize("forward_has_kwargs", [True, False])
@pytest.mark.parametrize("custom_loss", [True, False])
def test_seq2seq_is_never_run_through_the_causal_counter(forward_has_kwargs, custom_loss):
    """T5, Bart, Marian, LED, Whisper, M2M100: decoder targets the model shifts
    itself, against an ENCODER mask of a different length, so the AND raises.

    Both signatures are real, explicit on 4.57.6 and **kwargs on 5.17.0. The
    **kwargs one is reached through has_kwargs, so it fails on main too.
    """
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    if forward_has_kwargs:
        class Seq2SeqForConditionalGeneration(type(_tiny_model())):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"is_encoder_decoder": True})()
            def forward(self, input_ids, **kwargs):
                return self.lm_head(self.embed(input_ids))
    else:
        class Seq2SeqForConditionalGeneration(type(_tiny_model())):
            def __init__(self):
                super().__init__()
                self.config = type("Config", (), {"is_encoder_decoder": True})()
            def forward(self, input_ids):
                return self.lm_head(self.embed(input_ids))

    batch = {
        # Encoder inputs and mask are longer than the decoder labels, which is the
        # shape mismatch that kills the run.
        "input_ids": torch.randint(0, 11, (2, 9)),
        "attention_mask": torch.ones(2, 9, dtype = torch.long),
        "labels": torch.randint(0, 11, (2, 4)),
    }
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(Seq2SeqForConditionalGeneration(), True,
                      (lambda *a, **k: None) if custom_loss else None),
        iter([batch]), 1,
    )
    # Either way: no count. With **kwargs this batch is a dead run on main today
    # (the encoder mask cannot broadcast against the decoder labels), so there is no
    # working count to preserve here, and falling back to the mean is strictly
    # better than RuntimeError. Without **kwargs only the widened gate could have
    # admitted it at all.
    assert count is None, f"seq2seq returned {count}"


def test_a_seq2seq_batch_carrying_no_encoder_mask_counts_exactly_as_before():
    """The status-quo half, and the reason is_encoder_decoder does not gate the
    has_kwargs route. A Whisper style batch carries input_features and labels and no
    attention_mask, so nothing mismatches and nothing raises: it is counted today,
    off by one row-start, and it must keep being counted identically. Rescaling live
    seq2seq runs is a separate change from this PR.
    """
    torch = pytest.importorskip("torch")
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    class WhisperForConditionalGeneration(type(_tiny_model())):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"is_encoder_decoder": True})()
        def forward(self, input_ids, **kwargs):
            return self.lm_head(self.embed(input_ids))

    batch = {"labels": torch.randint(0, 11, (2, 4))}
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(WhisperForConditionalGeneration(), True), iter([batch]), 1,
    )
    assert count is not None and int(count) == 6, (
        f"the pre-existing seq2seq count changed from 6 to {count}; that rescales "
        f"every live Whisper and Florence2 run"
    )


@pytest.mark.parametrize("shifts, expected", [(True, 10), (None, None)])
def test_an_lm_head_model_follows_the_shifted_label_signal(shifts, expected):
    """GPT2LMHeadModel is the edge of the signal, so pin which way it falls.

    It matches none of CausalLM / ForConditionalGeneration / VisionText2Text and
    exposes .transformer rather than .model, so the walk never sees it, and
    `loss_type` comes back None because the name carries no LOSS_MAPPING key. It does
    shift labels in its own forward, but neither we nor stock transformers can prove
    that, so without the flag it takes the mean and training_step's /GA, which is
    always correct. When the Trainer does say the loss shifts, it counts.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()

    class GPT2LMHeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()

        def forward(self, input_ids): return None

    ids = torch.randint(0, 11, (2, 6))
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(GPT2LMHeadModel(), False, lambda *a, **k: None, shifts = shifts),
        iter([{"input_ids": ids, "labels": ids.clone()}]), 1,
    )
    assert (None if count is None else int(count)) == expected, (
        f"shifts={shifts}: expected {expected}, got {count}"
    )


def test_unequal_microbatch_counts_stay_grad_accum_invariant():
    """Equal token counts cannot tell sum/N apart from a mean of per-batch means.

    Counts of 3 and 17 can. This is the property the widened gate exists to give a
    custom loss whose model forward takes no **kwargs.
    """
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class ExplicitForCausalLM(type(_tiny_model())):
        def forward(self, input_ids):
            return self.lm_head(self.embed(input_ids))

    def compute_loss(logits, labels, num_items_in_batch = None):
        total = F.cross_entropy(
            logits[..., :-1, :].reshape(-1, logits.size(-1)),
            labels[..., 1:].reshape(-1),
            ignore_index = -100,
            reduction = "sum",
        )
        count = (labels[..., 1:] != -100).sum() if num_items_in_batch is None else num_items_in_batch
        return total / count

    model = ExplicitForCausalLM()
    batches = _microbatches(2, 10)
    # 2 rows x 10 shifted targets each. Leave 3 live in the first, all 17 in the
    # second minus the 3 it keeps masked, for 3 + 17 = 20 -> deliberately lopsided.
    batches[0]["labels"][0, 1:] = -100
    batches[0]["labels"][1, 4:] = -100
    batches[1]["labels"][0, 1:4] = -100

    full_input = torch.cat([batch["input_ids"] for batch in batches])
    full_labels = torch.cat([batch["labels"] for batch in batches])
    compute_loss(model(full_input), full_labels).backward()
    reference_grad = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none = True)

    mod = _loss_utils()
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, count = mod._unsloth_get_batch_samples(
        _fake_trainer(model, False, compute_loss), iter(batches), len(batches),
    )
    counts = [int((b["labels"][..., 1:] != -100).sum()) for b in batches]
    assert counts[0] != counts[1], "the microbatches must be lopsided or this proves nothing"
    assert int(count) == sum(counts)

    for batch in batches:
        compute_loss(model(batch["input_ids"]), batch["labels"], count).backward()
    torch.testing.assert_close(model.lm_head.weight.grad, reference_grad)


# The N-1 internal boundaries of a packed row are not training positions, but
# subtracting N-1 double counts every boundary a collator already masked with
# -100 (TRL >= 0.23.1 labels[position_ids == 0] = -100, completion_only_loss /
# assistant_masks on every version), under-counting num_items_in_batch and
# inflating loss and grads. So the arithmetic must be idempotent, not version
# detected. unsloth attaches packed_seq_lengths even when packing is off, so
# this reaches ordinary LoRA runs with per_device_train_batch_size > 1.


def _padding_free_batch(lengths, premask_starts = False, completion_only = 0):
    """One flattened padding-free row holding len(lengths) documents.

    premask_starts reproduces TRL >= 0.23.1, which writes -100 at every
    position_ids == 0 slot. completion_only masks the first N tokens of each
    document, which is what completion_only_loss and assistant_masks do on every
    TRL version, and which also lands -100 on the document starts. Synthesising
    both regimes here is what lets one installed TRL stand in for all of them.
    """
    torch = pytest.importorskip("torch")
    total = int(sum(int(x) for x in lengths))
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(1, 11, (1, total), generator = g)
    labels = ids.clone()
    offset = 0
    for length in lengths:
        length = int(length)
        if length <= 0: continue
        if premask_starts:
            labels[0, offset] = -100
        for i in range(min(completion_only, length)):
            labels[0, offset + i] = -100
        offset += length
    # Padding-free batches carry no attention_mask; unsloth's collator pops it.
    return {
        "input_ids": ids,
        "labels": labels,
        "packed_seq_lengths": torch.tensor([int(x) for x in lengths], dtype = torch.int32),
    }


def _true_target_count(labels, lengths):
    """Truth by an independent method, so a bug cannot hide in the oracle.

    Walks the batch in Python lists rather than reusing any of the tensor
    arithmetic under test. A shifted slot is a target when its label is not -100
    and the token it predicts is not the first token of a document (which
    includes the first token of a row, dropped by the labels[..., 1:] slice).
    """
    rows = labels.tolist()
    if not isinstance(rows[0], list): rows = [rows]
    width = len(rows[0])
    starts = set()
    offset = 0
    for length in lengths:
        length = int(length)
        if length <= 0: continue
        starts.add(offset)
        offset += length
    count = 0
    for r, row in enumerate(rows):
        for col in range(1, width):
            if (r * width + col) in starts: continue
            if row[col] == -100: continue
            count += 1
    return count


def _counted(batch, model = None):
    """num_items_in_batch as _unsloth_get_batch_samples reports it for one batch.

    ALLOWED_NUM_ITEMS_IN_BATCH is keyed on the model class name and a stale entry
    makes the counting branch not run at all, so every test clears it first or
    passes vacuously.
    """
    mod = _loss_utils()
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    if fn is None: pytest.skip("_unsloth_get_batch_samples not present")
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    _, count = fn(_fake_trainer(model if model is not None else _tiny_model(), True), iter([batch]), 1)
    return None if count is None else int(count)


def _normalizer():
    mod = _loss_utils()
    fn = getattr(mod, "_normalize_packed_seq_lengths", None)
    if fn is None: pytest.fail("_normalize_packed_seq_lengths is gone; the counting path relies on it")
    return fn


def test_normalize_accepts_every_shape_a_collator_can_emit():
    torch = pytest.importorskip("torch")
    normalize = _normalizer()
    expected = [4, 3, 3]
    candidates = [
        torch.tensor(expected, dtype = torch.int32),
        torch.tensor(expected, dtype = torch.int64),
        list(expected),
        tuple(expected),
    ]
    numpy = pytest.importorskip("numpy")
    candidates.append(numpy.array(expected, dtype = "int32"))
    for candidate in candidates:
        out = normalize(candidate)
        assert out is not None, f"{type(candidate).__name__} was rejected"
        assert out.dtype == torch.int64 and out.device.type == "cpu"
        assert out.tolist() == expected


def test_normalize_treats_unusable_metadata_as_absent():
    """It must degrade, never raise: the caller counts inside a try that re-raises
    as RuntimeError, which turns a bad batch into a dead training run."""
    normalize = _normalizer()
    for junk in (None, "not lengths", object(), [[1, 2], [3]], {"a": 1}):
        assert normalize(junk) is None, f"{junk!r} should have been treated as absent"


def test_normalize_drops_nonpositive_and_short_circuits_one_document():
    torch = pytest.importorskip("torch")
    normalize = _normalizer()
    assert normalize(torch.tensor([4, 0, 3, 0, 3])).tolist() == [4, 3, 3]
    # One document has no internal boundary, so there is nothing to drop.
    assert normalize(torch.tensor([7])) is None
    assert normalize(torch.tensor([7, 0, 0])) is None
    assert normalize(torch.tensor([], dtype = torch.int64)) is None
    assert normalize(torch.tensor(7)) is None


def test_count_is_identical_whether_or_not_the_collator_premasked():
    """The headline property. TRL >= 0.23.1 masks the document starts and earlier
    versions do not, and completion-only masking does it on every version. The
    count must not depend on which of those produced the batch."""
    lengths = [4, 3, 3]
    plain = _counted(_padding_free_batch(lengths))
    masked = _counted(_padding_free_batch(lengths, premask_starts = True))
    assert plain == masked == 7, (
        f"pre-masked batch counted {masked} against {plain} unmasked; the boundary "
        "slots are being removed twice on whichever collator already masked them"
    )


def test_count_equals_total_tokens_minus_document_count():
    for lengths in ([4, 3, 3], [8] * 4, [256, 256], [5, 1, 9, 2]):
        total = sum(lengths)
        for premask in (False, True):
            batch = _padding_free_batch(lengths, premask_starts = premask)
            counted = _counted(batch)
            assert counted == _true_target_count(batch["labels"], lengths)
            assert counted == total - len(lengths), (
                f"lengths={lengths} premask={premask}: counted {counted}, "
                f"expected {total - len(lengths)}"
            )


def test_completion_only_masking_is_not_double_subtracted():
    """completion_only_loss puts -100 on the document starts on every TRL version,
    which is why a version gate would leave the commonest SFT config broken."""
    lengths = [5, 5, 5, 5]
    batch = _padding_free_batch(lengths, completion_only = 3)
    counted = _counted(batch)
    assert counted == _true_target_count(batch["labels"], lengths)
    # 4 documents x 5 tokens, first 3 of each masked; the boundary targets are
    # already inside those masked prefixes.
    assert counted == 8, f"counted {counted}, expected 8"


def test_single_document_batch_is_untouched():
    """N = 1 has no internal boundary. Provable no-op, not an approximate one."""
    lengths = [12]
    batch = _padding_free_batch(lengths)
    counted = _counted(batch)
    assert counted == _true_target_count(batch["labels"], lengths) == 11


def test_zero_length_entries_are_ignored():
    torch = pytest.importorskip("torch")
    lengths = [4, 0, 3, 0, 3]
    batch = _padding_free_batch(lengths)
    assert batch["input_ids"].shape[-1] == 10
    counted = _counted(batch)
    assert counted == _true_target_count(batch["labels"], lengths) == 7


def test_lengths_overrunning_the_batch_do_not_kill_live_targets():
    """Metadata is trusted to describe the row, so guard the case where it does
    not. Boundaries past the end of the batch must be dropped, not wrapped onto a
    row that is really there."""
    torch = pytest.importorskip("torch")
    # 5 puts a boundary at flat 15: past a 10 token batch and not on a row start,
    # so only the rows < n_rows filter catches it. Without it, IndexError inside
    # the try that re-raises as RuntimeError.
    for overrun in ([4, 3, 3, 5, 50], [4, 3, 3, 50, 50], [4, 3, 3, 1]):
        batch = _padding_free_batch([4, 3, 3])
        batch["packed_seq_lengths"] = torch.tensor(overrun, dtype = torch.int32)
        counted = _counted(batch)
        assert counted == 7, f"lengths={overrun}: counted {counted}, expected 7"


def test_metadata_shorter_than_the_batch_never_targets_the_trailing_boundary():
    """cumsum(...)[:-1] rather than the full cumsum: the trailing boundary of a
    truncated description is a live target, and truncated metadata must not eat
    it."""
    torch = pytest.importorskip("torch")
    batch = _padding_free_batch([4, 3, 3])
    batch["packed_seq_lengths"] = torch.tensor([4, 3], dtype = torch.int32)
    counted = _counted(batch)
    assert counted == 8, f"counted {counted}, expected 8 (10 tokens, 1 row start, 1 boundary)"


def test_a_document_starting_at_a_row_boundary_is_not_dropped_twice():
    """A document that begins exactly at a row start was already dropped by the
    labels[..., 1:] slice, so it must not be zeroed again."""
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(1, 11, (2, 6), generator = g)
    lengths = [6, 6]
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "packed_seq_lengths": torch.tensor(lengths, dtype = torch.int32),
    }
    counted = _counted(batch)
    assert counted == _true_target_count(batch["labels"], lengths) == 10, (
        f"counted {counted}, expected 10: both rows keep all 5 shifted targets"
    )


def test_multi_row_boundaries_map_row_major():
    """Documents tile the flattened batch row-major, so a boundary at flat index s
    lands at row s // width, column s % width."""
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(1, 11, (3, 4), generator = g)
    lengths = [3, 5, 4]
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "packed_seq_lengths": torch.tensor(lengths, dtype = torch.int32),
    }
    counted = _counted(batch)
    # 12 tokens, 3 row starts dropped by the slice, boundaries at flat 3 and 8.
    # Flat 8 is a row start and already gone, so only flat 3 is live.
    assert counted == _true_target_count(batch["labels"], lengths) == 8, (
        f"counted {counted}, expected 8"
    )


def test_applying_the_boundary_removal_twice_is_idempotent():
    """The property that makes a version check unnecessary, asserted directly on
    the counting path rather than inferred from the two-collator comparison."""
    torch = pytest.importorskip("torch")
    lengths = [4, 3, 3]
    # Pre-masked, so the boundary slots are already gone before the counting path
    # touches them: removing them a second time is exactly the bug.
    batch = _padding_free_batch(lengths, premask_starts = True)
    first = _counted(batch)
    second = _counted(batch)
    assert first == second == 7, f"first {first}, second {second}"
    assert int((batch["labels"] != -100).sum()) == 7, (
        "the counting path mutated the caller's labels"
    )


def test_batch_without_packed_seq_lengths_is_unchanged():
    """The no-metadata path must be bit identical to before the fix."""
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(1, 11, (4, 7), generator = g)
    labels = ids.clone()
    labels[0, 3] = -100
    counted = _counted({"input_ids": ids, "labels": labels})
    assert counted == int((labels[..., 1:] != -100).sum()) == 23


def test_a_plain_list_of_lengths_does_not_kill_the_run():
    """`seq_lengths > 0` raises on a list, and the enclosing except re-raises as
    RuntimeError, so this used to be a dead training run rather than a bad count."""
    lengths = [4, 3, 3]
    batch = _padding_free_batch(lengths)
    batch["packed_seq_lengths"] = list(lengths)
    assert _counted(batch) == 7


def test_int32_metadata_counts_like_int64():
    torch = pytest.importorskip("torch")
    lengths = [4, 3, 3]
    counted = []
    for dtype in (torch.int32, torch.int64):
        batch = _padding_free_batch(lengths)
        batch["packed_seq_lengths"] = torch.tensor(lengths, dtype = dtype)
        counted.append(_counted(batch))
    assert counted == [7, 7], f"int32 and int64 metadata disagreed: {counted}"


def test_unusable_metadata_degrades_instead_of_raising():
    batch = _padding_free_batch([4, 3, 3])
    batch["packed_seq_lengths"] = "garbage"
    # Falls back to the plain shifted count rather than raising RuntimeError.
    assert _counted(batch) == 9


def test_normalize_does_not_raise_when_the_length_filter_cannot_run():
    """Dropping the non-positive lengths is a boolean mask, so it lowers to
    aten.nonzero, whose output shape is data dependent. Under FakeTensorMode that
    raises DynamicOutputShapeException. The whole body therefore has to sit inside
    the try: the caller counts inside `except Exception: raise RuntimeError(...)`,
    so an escape here is a dead training run, not the missing correction this
    helper exists to degrade to."""
    torch = pytest.importorskip("torch")
    fake = pytest.importorskip("torch._subclasses.fake_tensor")
    normalize = _normalizer()
    with fake.FakeTensorMode():
        # Positive lengths only, so this is metadata the helper would normally
        # accept: the raise comes from the filter, not from rejecting the input.
        assert normalize(torch.tensor([4, 3, 3])) is None


def test_the_counting_path_does_not_raise_on_a_traced_tensor():
    """The same escape, observed where it actually hurts: through the public
    counting entry point, whose except clause converts anything that gets out into
    a RuntimeError that ends the run."""
    torch = pytest.importorskip("torch")
    fake = pytest.importorskip("torch._subclasses.fake_tensor")
    mod = _loss_utils()
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    if fn is None: pytest.skip("_unsloth_get_batch_samples not present")
    with fake.FakeTensorMode():
        ids = torch.randint(1, 11, (1, 10))
        batch = {
            "input_ids": ids,
            "labels": ids.clone(),
            "packed_seq_lengths": torch.tensor([4, 3, 3], dtype = torch.int32),
        }
        mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
        # Called directly, not through _counted: its int(count) is a .item() that
        # raises on a fake tensor by itself. No number exists under a fake tensor
        # either, so the only contract here is that it does not end the run.
        fn(_fake_trainer(_tiny_model(), True), iter([batch]), 1)


def test_the_count_can_never_go_negative():
    """Subtracting N-1 had no lower bound. On a batch whose live targets number
    fewer than N, the old arithmetic returned zero or a negative num_items_in_batch,
    so the loss divided by zero or changed sign and the gradients ascended. Zeroing
    slots that may already be False is bounded below by construction, and this
    pins that.

    Every case here has real trainable targets, so declining to count is not an
    acceptable answer either.
    """
    torch = pytest.importorskip("torch")
    # (lengths, completion_only, premask, true target count)
    cases = [
        ([4, 3, 3],       3, False, 1),
        ([1, 1, 2],       1, True,  1),
        ([5, 3],          4, False, 1),
        ([2, 5, 2],       4, False, 1),
        ([1, 5, 6, 3, 3], 4, True,  3),
        ([5, 3, 4, 2],    3, False, 3),
    ]
    for lengths, completion_only, premask, expected in cases:
        batch = _padding_free_batch(
            lengths, premask_starts = premask, completion_only = completion_only,
        )
        counted = _counted(batch)
        # The bound first: after the equality below it would be dead, since every
        # expected value here is >= 1 and no negative count could ever reach it.
        assert counted is not None and counted > 0, (
            f"lengths={lengths} completion_only={completion_only}: num_items_in_batch "
            f"came back {counted}. A non-positive count makes the loss divide by zero "
            "or flip sign, which is what subtracting an unbounded N-1 allowed"
        )
        assert counted == _true_target_count(batch["labels"], lengths)
        assert counted == expected, (
            f"lengths={lengths} completion_only={completion_only}: "
            f"counted {counted}, expected {expected}"
        )


def test_the_real_installed_trl_collator_is_counted_correctly():
    """One end-to-end pass through whatever TRL is installed, so the synthesised
    regimes above cannot drift away from a real collator's output."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    # Moved between trl.trainer.utils and trl.trainer.sft_trainer across releases.
    cls = None
    for module in ("trl.trainer.sft_trainer", "trl.trainer.utils"):
        try: candidate = __import__(module, fromlist = ["DataCollatorForLanguageModeling"])
        except Exception: continue
        cls = getattr(candidate, "DataCollatorForLanguageModeling", None)
        if cls is not None: break
    if cls is None: pytest.skip("this TRL has no DataCollatorForLanguageModeling")
    # return_position_ids only exists on some releases; padding_free implies it elsewhere.
    collator = None
    for kwargs in ({"return_position_ids": True}, {}):
        try:
            collator = cls(pad_token_id = 0, padding_free = True, **kwargs)
            break
        except TypeError:
            continue
    if collator is None: pytest.skip("this TRL's collator does not take padding_free")

    lengths = [4, 3, 3]
    examples = [{"input_ids": list(range(1, n + 1))} for n in lengths]
    batch = collator(examples)
    if "position_ids" not in batch: pytest.skip("collator returned no position_ids")
    if batch["labels"].shape[-1] != sum(lengths): pytest.skip("collator did not flatten the batch")
    batch = dict(batch)
    batch.pop("attention_mask", None)
    batch["packed_seq_lengths"] = torch.tensor(lengths, dtype = torch.int32)

    counted = _counted(batch)
    assert counted == _true_target_count(batch["labels"], lengths), (
        f"counted {counted} against an independently computed "
        f"{_true_target_count(batch['labels'], lengths)} on the real collator"
    )
    assert counted == 7, (
        f"counted {counted}, expected 10 tokens minus 3 document starts"
    )


def test_the_counting_path_never_branches_on_the_trl_version():
    """Locks in the architectural decision. Feature detection answers the wrong
    question (what matters is whether these exact slots are already -100, which
    completion-only masking also decides), inspects a class the batch may never
    have come from, and inspect.getsource raises on frozen, zipped, -OO or wrapped
    installs, inside the try that turns it into a hard crash."""
    import ast

    mod = _loss_utils()
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "trl", "loss_utils must not import trl"
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").split(".")[0] == "trl", "loss_utils must not import trl"

    counting = inspect.getsource(mod._unsloth_get_batch_samples)
    counting += inspect.getsource(mod._normalize_packed_seq_lengths)
    # Strip comments: the rationale above is allowed to name versions, the code is not.
    code = "\n".join(line.split("#", 1)[0] for line in counting.splitlines())
    for banned in ("__version__", "Version(", "getsource", "trl"):
        assert banned not in code, (
            f"the packed-boundary count branches on {banned!r}; it must be "
            "idempotent instead, because completion-only masking regresses the "
            "same slots on every TRL version"
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
            # Where stock counts over labels[..., 1:] the count must agree too, not
            # just whether one was produced. transformers adopted that rule in 5.x;
            # zoo always counted that way, so before 5.x the two legitimately differ
            # by one token per row and only the nullity above is shared.
            if _stock_counts_shifted_labels(stock) and ours is not None:
                assert int(ours) == int(theirs), (
                    f"accepts={accepts} compute_loss_func={loss_func is not None}: "
                    f"counted {int(ours)} where stock counted {int(theirs)}"
                )
