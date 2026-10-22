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

"""_unsloth_get_batch_samples against the batch shape TRL's GRPO trainer produces.

GRPOTrainer passes `data_collator=identity`, so a batch is whatever the sampler
handed the collator: a LIST of dataset rows, never a mapping. The pre-#1217 gate
was `"labels" in batch_samples[0]`, and `in` on a list of dicts is simply False,
so the counting body was skipped and GRPO trained. #1217 rewrote that membership
test as `batch_samples[0].get("labels")`, which raises

    AttributeError: 'list' object has no attribute 'get'

on every GRPO step, on TRL 0.22.2 and on TRL 1.13.0 alike.

What #1217 bought with `.get` must survive: `in` cannot tell a tensor `labels`
from a non-tensor one, so a mapping carrying list labels used to enter the try
and die as RuntimeError. The `.get` plus the `.ndim` probe declines to count it
instead. Both halves are pinned below.

CPU only, no model downloads.
"""

import inspect

import pytest


def _loss_utils():
    return pytest.importorskip("unsloth_zoo.loss_utils")


def _fake_trainer(model, accepts = True, compute_loss_func = None, shifts = True):
    """The minimum of a Trainer that _unsloth_get_batch_samples reads."""
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
    if shifts is not None: t._loss_shifts_labels = shifts
    return t


def _causal_model():
    """A **kwargs CausalLM, so has_kwargs is True and the counting gate is live."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class TinyForCausalLM(nn.Module):
        accepts_loss_kwargs = False

        def __init__(self, vocab = 11, hidden = 6):
            super().__init__()
            torch.manual_seed(0)
            self.embed = nn.Embedding(vocab, hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias = False)

        def forward(self, input_ids, labels = None, num_items_in_batch = None, **kwargs):
            return self.lm_head(self.embed(input_ids))

    return TinyForCausalLM()


def _grpo_rows(n_prompts = 4):
    """Rows as a GRPO dataset holds them: prompt text, no tensors, no labels."""
    return [
        {"prompt": f"What is {i} + {i}? Answer with the number only.", "answer": str(2 * i)}
        for i in range(n_prompts)
    ]


def _call(trainer, batches):
    mod = _loss_utils()
    fn = getattr(mod, "_unsloth_get_batch_samples", None)
    if fn is None: pytest.skip("_unsloth_get_batch_samples not present")
    mod.ALLOWED_NUM_ITEMS_IN_BATCH.clear()
    return fn(trainer, iter(batches), len(batches))


# ---------------------------------------------------------------------------
# The regression: a batch that is a list, not a mapping.
# ---------------------------------------------------------------------------

def test_grpo_identity_collator_batch_is_a_list_of_dicts():
    """Pin the shape the rest of this file asserts against, off TRL itself.

    Same on 0.22.x and on 1.x: GRPOTrainer builds with `data_collator=identity`,
    so the collator returns the sampler's list untouched.
    """
    trl = pytest.importorskip("trl")
    torch = pytest.importorskip("torch")
    from trl.trainer.grpo_trainer import GRPOTrainer, identity

    assert "data_collator=identity" in inspect.getsource(GRPOTrainer.__init__), (
        f"TRL {trl.__version__} GRPOTrainer no longer collates with identity; "
        "re-derive the batch shape this file pins"
    )
    loader = torch.utils.data.DataLoader(
        _grpo_rows(), batch_size = 2, collate_fn = identity,
    )
    batch = next(iter(loader))
    assert isinstance(batch, list) and isinstance(batch[0], dict), (
        f"expected a list of dicts from the identity collator, got {type(batch)}"
    )


def test_a_list_batch_is_not_countable_and_does_not_raise():
    """The regression itself.

    On main this raises AttributeError: 'list' object has no attribute 'get',
    which is what every GRPO user sees. Pre-#1217, `"labels" in batch_samples[0]`
    was False on a list and the run trained.
    """
    pytest.importorskip("torch")
    batches = [_grpo_rows(), _grpo_rows()]
    samples, count = _call(_fake_trainer(_causal_model()), batches)
    assert count is None, "a list batch carries no labels and must not be counted"
    assert samples == batches, "the batches must be handed back untouched"


def test_a_list_batch_is_not_countable_on_the_widened_route_either():
    """#1217 widened WHO is eligible to compute_loss_func without **kwargs. Both
    routes read batch_samples[0] through the same expression, so both crashed."""
    pytest.importorskip("torch")
    trainer = _fake_trainer(
        _causal_model(), accepts = False, compute_loss_func = lambda *a, **k: None,
    )
    _, count = _call(trainer, [_grpo_rows()])
    assert count is None


@pytest.mark.parametrize("batch", [
    [],                                   # empty rows, still a list
    [{"prompt": "hi"}],                   # rows with no labels at all
    [{"labels": [0, 1, 2]}],              # a row that does carry the key
])
def test_no_list_batch_shape_reaches_the_counting_body(batch):
    """`x["labels"]` in the loop would fail on a list exactly as `.get` does, so
    the guard has to hold for every list, including one whose rows carry labels."""
    pytest.importorskip("torch")
    _, count = _call(_fake_trainer(_causal_model()), [batch])
    assert count is None


# ---------------------------------------------------------------------------
# Negative controls: the mapping case must still take #1217's path.
# ---------------------------------------------------------------------------

def test_a_mapping_batch_still_counts_exactly_what_it_counted_before():
    """Control. A normal causal batch is a mapping of tensors and must still be
    counted, to the same number: 2 rows x 5 shifted targets, minus 3 masked."""
    torch = pytest.importorskip("torch")
    ids = torch.randint(0, 11, (2, 6), generator = torch.Generator().manual_seed(7))
    labels = ids.clone()
    labels[1, 1:4] = -100
    _, count = _call(_fake_trainer(_causal_model()), [{"input_ids": ids, "labels": labels}])
    assert int(count) == 7, f"expected 7 shifted targets, got {count}"


def test_a_mapping_with_non_tensor_labels_declines_instead_of_raising():
    """Control for what #1217 bought.

    `"labels" in x` is True here, so the pre-#1217 spelling entered the try,
    `labels[..., 1:]` blew up on a list and the except re-raised it as
    RuntimeError. The `.get` plus `.ndim` probe declines to count instead.

    Spelling the guard as a membership test again, `"labels" in batch_samples[0]`,
    turns this test red with RuntimeError while every other test here stays green.
    """
    pytest.importorskip("torch")
    _, count = _call(_fake_trainer(_causal_model()), [{"labels": [[1, 2, 3]]}])
    assert count is None, "labels with no .ndim must be declined, not counted"


def test_a_mapping_that_is_not_a_dict_subclass_still_counts():
    """Control for the shape of the guard itself.

    transformers hands collated batches back as BatchEncoding, a UserDict, so
    `isinstance(x, dict)` is False for it. The guard must ask for a Mapping, or
    every DataCollatorForLanguageModeling batch silently stops being counted.
    """
    torch = pytest.importorskip("torch")
    from transformers.tokenization_utils_base import BatchEncoding

    ids = torch.randint(0, 11, (2, 6), generator = torch.Generator().manual_seed(7))
    batch = BatchEncoding({"input_ids": ids, "labels": ids.clone()})
    assert not isinstance(batch, dict), "BatchEncoding stopped being a UserDict"
    _, count = _call(_fake_trainer(_causal_model()), [batch])
    assert int(count) == 10, f"a non-dict Mapping must still be counted, got {count}"
