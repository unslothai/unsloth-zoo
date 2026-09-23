"""`mask_truncated_completions` must reach the loss on the text path.

TRL drops a truncated completion by zeroing its whole row in `completion_mask`, and from TRL 1.9
it also leaves that row out of `num_items_in_batch`. The text branch of `grpo_accumulated_loss`
rebuilds the mask from token ids, which used to bring the row back: the loss trained on a
completion the user asked to drop, divided by a count that did not include it, and a batch of
nothing but truncated completions divided a non-zero sum by 0.

These tests run the real function up to the loss call on CPU and read the mask it hands over.
"""
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as _rl

HIDDEN, VOCAB = 4, 8


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


# Prompt of 2 tokens, completion of 3, pad id 0. Row 1 is a truncated completion TRL zeroed.
INPUT_IDS = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 3], [0, 2, 5, 5, 0]])
TRL_MASK = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 1, 0]])


def _loss_mask(monkeypatch, completion_mask, **trainer_attrs):
    """The completion_mask grpo_accumulated_loss passes to the loss."""
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    seen = {}

    class _Loss:
        @staticmethod
        def apply(new, old, ref, sampling, lm_head, ids, mask, advantages, beta, scaler, n, kwargs):
            seen["mask"] = mask.clone()
            zero = torch.zeros(())
            return zero, zero, zero, zero, zero, zero

    monkeypatch.setattr(_rl, "UnslothEfficientGRPO", _Loss)
    model = _Model()
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None,
        ),
        use_vllm = False,
        _autocast_dtype = None,
        beta = 0.0,
        **trainer_attrs,
    )
    _rl.grpo_accumulated_loss(
        trainer, INPUT_IDS, (INPUT_IDS != 0).long(), 3, completion_mask,
        torch.zeros(INPUT_IDS.shape[0]), None, None,
        loss_type = "dapo", num_items_in_batch = int(completion_mask.sum()),
    )
    return seen["mask"]


def test_a_truncated_row_stays_out_of_the_loss(monkeypatch):
    kept = _loss_mask(monkeypatch, TRL_MASK, mask_truncated_completions = True)
    rebuilt = _loss_mask(monkeypatch, TRL_MASK, mask_truncated_completions = False)
    assert rebuilt[1].sum() > 0, "control: the rebuild alone must bring the row back"
    assert kept[1].sum() == 0
    # Only the dropped row moves; the kept rows keep the rebuilt, left-pad aligned mask.
    torch.testing.assert_close(kept[[0, 2]], rebuilt[[0, 2]])


def test_a_batch_of_only_truncated_rows_has_an_empty_loss_mask(monkeypatch):
    mask = _loss_mask(monkeypatch, torch.zeros_like(TRL_MASK), mask_truncated_completions = True)
    assert mask.sum() == 0


@pytest.mark.parametrize("trainer_attrs", [{}, {"mask_truncated_completions": False}])
def test_without_the_option_the_rebuilt_mask_is_untouched(monkeypatch, trainer_attrs):
    left_pad = _rl.calculate_pad_tokens_in_prompt(INPUT_IDS, 3, 0)
    max_left_pad = int(left_pad.max())
    expected = _rl.create_completion_attention_mask(
        _rl.left_pack_padding(INPUT_IDS, 0)[:, -(3 + max_left_pad):], left_pad, max_left_pad, 0,
    )
    mask = _loss_mask(monkeypatch, TRL_MASK, **trainer_attrs)
    assert torch.equal(mask.bool(), expected.bool())


def test_the_kl_metric_leaves_dropped_rows_out():
    """A dropped row must not drag the logged KL toward 0, and an all-dropped batch logs 0."""
    torch.manual_seed(0)
    B, T = 4, 5
    new = torch.randn(B, T, dtype = torch.float64)
    ref = new + 0.3 * torch.randn(B, T, dtype = torch.float64)
    mask = torch.ones(B, T, dtype = torch.float64)
    kwargs = dict(loss_type = "dapo", num_items_in_batch = float(mask.sum()), num_processes = 1,
                  current_gradient_accumulation_steps = 1, max_completion_length = T)

    def kl(m):
        return _rl.grpo_compute_loss(ref, new, new, None, torch.zeros(B, T, dtype = torch.long),
                                     m, 0.1, torch.zeros(B, dtype = torch.float64), **kwargs)[2]

    full = kl(mask)
    kept = mask.clone()
    kept[1:] = 0
    only_row0 = kl(mask[:1].expand(B, T) * kept)
    per_row = (torch.exp(ref - new) - (ref - new) - 1).mean(1)
    torch.testing.assert_close(full, per_row.mean())
    torch.testing.assert_close(only_row0, per_row[0])
    assert kl(torch.zeros_like(mask)).item() == 0.0
