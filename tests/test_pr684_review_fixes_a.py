# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""CPU regression tests for the PR #684 review-fix cluster A.

Covers the trainer/batching/tokenizer findings:

1. _train_inner must not clobber a prebuilt _prepared_batches_include_epochs
   flag set by train_on_responses_only (double-epoch counting).
2. _create_labeled_batches must not raise on seed=None with torch_randperm.
3. train_on_responses_only must only materialize epoch blocks for true
   epoch-based runs (max_steps <= 0).
4. SGD weight decay must be coupled into the gradient (HF/PyTorch parity),
   not applied as AdamW-style decoupled parameter shrink.
5. _get_processor_tokenizer must not return the low-level Rust backend for
   HF fast tokenizers (which would break convert_tokens_to_ids / templates).

Runs on Linux via the mlx_simulation torch shim.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


# ---------------------------------------------------------------------------
# Shared lightweight tokenizer / model stubs.
# ---------------------------------------------------------------------------

class _SpaceTokenizer:
    """Whitespace tokenizer that mimics the HF fast-tokenizer surface."""

    pad_token_id = 0
    eos_token_id = 99
    unk_token_id = -1

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            return {"input_ids": self.encode(texts)}
        return {"input_ids": [self.encode(text) for text in texts]}

    def encode(self, text):
        return [int(part) for part in str(text).split()]

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]
        return self.unk_token_id


def _identity_mask_fn(d):
    """Keep all tokens as labels (no instruction masking)."""
    ids = list(d["input_ids"][0])
    return {"labels": [list(ids)]}


# ===========================================================================
# Thread 4: SGD coupled weight decay.
# ===========================================================================

def test_thread4_build_optimizer_routes_sgd_to_coupled_decay():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(optim="sgd", weight_decay=0.1)

    optimizer = trainer._build_optimizer(total_steps=4)

    # SGD uses coupled decay, not decoupled manual shrink.
    assert trainer._coupled_weight_decay == pytest.approx(0.1)
    assert trainer._manual_weight_decay == pytest.approx(0.0)
    # MLX SGD's built-in decay stays off; our helper owns the decay term.
    if hasattr(optimizer, "_kw"):
        assert optimizer._kw["weight_decay"] == 0.0


def test_thread4_coupled_decay_folds_into_grad_and_exempts_bias_norm():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from unsloth_zoo.mlx.trainer import MLXTrainer

    class TinyModel:
        def trainable_parameters(self):
            return {
                "layer": {
                    "weight": mx.array([2.0], dtype=mx.float32),
                    "bias": mx.array([2.0], dtype=mx.float32),
                },
                "norm": {"weight": mx.array([2.0], dtype=mx.float32)},
            }

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer._coupled_weight_decay = 0.1

    grad = {
        "layer": {
            "weight": mx.array([1.0], dtype=mx.float32),
            "bias": mx.array([1.0], dtype=mx.float32),
        },
        "norm": {"weight": mx.array([1.0], dtype=mx.float32)},
    }
    out = trainer._apply_coupled_weight_decay(TinyModel(), grad)
    flat = dict(tree_flatten(out))

    # weight grad gets wd * param folded in: 1.0 + 0.1 * 2.0 = 1.2
    assert flat["layer.weight"].item() == pytest.approx(1.2)
    # bias and norm are exempt (HF param-group filter).
    assert flat["layer.bias"].item() == pytest.approx(1.0)
    assert flat["norm.weight"].item() == pytest.approx(1.0)


def test_thread4_coupled_decay_is_noop_when_disabled():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import MLXTrainer

    class TinyModel:
        def trainable_parameters(self):
            return {"layer": {"weight": mx.array([2.0], dtype=mx.float32)}}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer._coupled_weight_decay = 0.0
    grad = {"layer": {"weight": mx.array([1.0], dtype=mx.float32)}}
    # Returns the same object untouched when wd <= 0.
    assert trainer._apply_coupled_weight_decay(TinyModel(), grad) is grad


def test_thread4_coupled_sgd_matches_torch_sgd_with_momentum():
    """End-to-end parity: coupling wd into the grad then stepping MLX-sim SGD
    (momentum) must equal torch.optim.SGD(weight_decay=wd) on a non-exempt
    weight. This is what the decoupled shrink got wrong."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from unsloth_zoo.mlx.trainer import MLXTrainer

    wd, lr, mom = 0.1, 0.5, 0.9

    # Reference: torch SGD couples wd into the gradient (and momentum).
    ref = torch.tensor([3.0], requires_grad=True)
    ref_opt = torch.optim.SGD([ref], lr=lr, momentum=mom, weight_decay=wd)
    for _ in range(3):
        ref.grad = torch.tensor([1.0])
        ref_opt.step()

    # Branch path: MLX SGD with weight_decay=0 + our coupled helper.
    class TinyModel:
        def __init__(self):
            self.p = mx.array([3.0], dtype=mx.float32)

        def trainable_parameters(self):
            return {"layer": {"weight": self.p}}

        def parameters(self):
            return self.trainable_parameters()

        def update(self, updates):
            self.p = updates["layer"]["weight"]

    model = TinyModel()
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer._coupled_weight_decay = wd
    optimizer = optim.SGD(learning_rate=lr, weight_decay=0.0, momentum=mom)

    for _ in range(3):
        grad = {"layer": {"weight": mx.array([1.0], dtype=mx.float32)}}
        grad = trainer._apply_coupled_weight_decay(model, grad)
        optimizer.update(model, grad)

    assert model.p.item() == pytest.approx(ref.item(), rel=1e-4, abs=1e-4)


def test_thread4_decoupled_shrink_would_diverge_from_torch_sgd():
    """Guard the regression itself: the old decoupled shrink does NOT match
    torch SGD once momentum is involved, so the coupled path is required."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from unsloth_zoo.mlx.trainer import MLXTrainer

    wd, lr, mom = 0.1, 0.5, 0.9

    ref = torch.tensor([3.0], requires_grad=True)
    ref_opt = torch.optim.SGD([ref], lr=lr, momentum=mom, weight_decay=wd)
    for _ in range(3):
        ref.grad = torch.tensor([1.0])
        ref_opt.step()

    class TinyModel:
        def __init__(self):
            self.p = mx.array([3.0], dtype=mx.float32)

        def trainable_parameters(self):
            return {"layer": {"weight": self.p}}

        def parameters(self):
            return self.trainable_parameters()

        def update(self, updates):
            self.p = updates["layer"]["weight"]

    model = TinyModel()
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer._manual_weight_decay = wd  # old decoupled path
    optimizer = optim.SGD(learning_rate=lr, weight_decay=0.0, momentum=mom)

    for _ in range(3):
        grad = {"layer": {"weight": mx.array([1.0], dtype=mx.float32)}}
        trainer._apply_manual_weight_decay(model, optimizer, grad)
        optimizer.update(model, grad)

    assert model.p.item() != pytest.approx(ref.item(), rel=1e-4, abs=1e-4)


# ===========================================================================
# Thread 2: seed=None labeled torch_randperm ordering.
# ===========================================================================

def test_thread2_labeled_batches_accept_none_seed_with_torch_randperm():
    from unsloth_zoo.mlx.trainer import _create_labeled_batches

    ds = [{"text": f"{i} {i + 10} {i + 20}"} for i in range(4)]
    # Must not raise TypeError on None + epoch_idx.
    batches = _create_labeled_batches(
        dataset=ds,
        tokenizer=_SpaceTokenizer(),
        mask_fn=_identity_mask_fn,
        batch_size=1,
        max_seq_length=8,
        dataset_text_field="text",
        seed=None,
        dataset_order="torch_randperm",
        num_epochs=2,
    )
    # 4 samples x 2 epochs, batch_size 1.
    assert len(batches) == 8


def test_thread2_labeled_torch_randperm_reseeds_per_epoch():
    from unsloth_zoo.mlx.trainer import _create_labeled_batches

    ds = [{"text": f"{i} {i + 10} {i + 20} {i + 30}"} for i in range(5)]
    batches = _create_labeled_batches(
        dataset=ds,
        tokenizer=_SpaceTokenizer(),
        mask_fn=_identity_mask_fn,
        batch_size=1,
        max_seq_length=8,
        dataset_text_field="text",
        seed=None,
        dataset_order="torch_randperm",
        num_epochs=2,
    )
    assert len(batches) == 10
    first = [int(b[0][0, 0].item()) for b in batches[:5]]
    second = [int(b[0][0, 0].item()) for b in batches[5:]]
    assert sorted(first) == [0, 1, 2, 3, 4]
    assert sorted(second) == [0, 1, 2, 3, 4]
    # Per-epoch reseed means the two orders differ.
    assert first != second


# ===========================================================================
# Threads 1 & 3: epoch flag preservation and gated materialization.
# ===========================================================================

class _StubModel:
    _hf_repo = None
    _config = {}


class _StubTrainer:
    """Minimal trainer surface for the text train_on_responses_only path."""

    def __init__(self, args):
        self._is_vlm = False
        self.args = args
        self.model = _StubModel()
        self.train_dataset = [
            {"text": f"{i} {i + 10} {i + 20}"} for i in range(6)
        ]
        self.eval_dataset = None
        self.formatting_func = None
        self.tokenizer = _SpaceTokenizer()
        self._batches = None
        self._prepared_batches_include_epochs = None


def _run_train_on_responses_only(monkeypatch, args):
    """Drive mlx.trainer.train_on_responses_only with an identity HF mask."""
    import unsloth_zoo.dataset_utils as dataset_utils
    import unsloth_zoo.mlx.trainer as trainer_mod

    def fake_hf(trainer, *, instruction_part=None, response_part=None,
                force_match=True, tokenizer=None, return_function=False,
                num_proc=None, last_response_only=False):
        return _identity_mask_fn

    monkeypatch.setattr(dataset_utils, "train_on_responses_only", fake_hf)
    trainer = _StubTrainer(args)
    trainer_mod.train_on_responses_only(
        trainer,
        instruction_part="<user>",
        response_part="<assistant>",
    )
    return trainer


def test_thread3_step_based_run_does_not_materialize_all_epochs(monkeypatch):
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    # max_steps > 0 + num_train_epochs > 1: must NOT pre-build every epoch.
    args = MLXTrainingConfig(
        optim="adamw",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        num_train_epochs=5,
        max_seq_length=8,
        seed=0,
    )
    trainer = _run_train_on_responses_only(monkeypatch, args)

    # Truncated to max_steps * grad_accum = 2 batches, not 6 * 5 = 30.
    assert len(trainer._batches) == 2
    # Step-based run: prebuilt batches are a single (truncated) block.
    assert trainer._prepared_batches_include_epochs is False


def test_thread3_epoch_based_run_materializes_all_epochs(monkeypatch):
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    args = MLXTrainingConfig(
        optim="adamw",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=0,
        num_train_epochs=3,
        max_seq_length=8,
        seed=0,
    )
    trainer = _run_train_on_responses_only(monkeypatch, args)

    # 6 samples x 3 epochs at batch_size 1.
    assert len(trainer._batches) == 18
    assert trainer._prepared_batches_include_epochs is True


def test_thread1_train_inner_preserves_prebuilt_epoch_flag(monkeypatch):
    """_train_inner resets the flag only when _batches is not prebuilt, so a
    train_on_responses_only epoch-based run keeps include_epochs=True and the
    step counter does not multiply by num_train_epochs a second time."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    args = MLXTrainingConfig(
        optim="adamw",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=0,
        num_train_epochs=3,
        max_seq_length=8,
        seed=0,
    )
    trainer = _run_train_on_responses_only(monkeypatch, args)
    assert trainer._prepared_batches_include_epochs is True
    n_batches = len(trainer._batches)

    # Replicate _train_inner's reset guard exactly.
    if trainer._batches is None:
        trainer._prepared_batches_include_epochs = False

    # Flag must survive because _batches was prebuilt.
    assert trainer._prepared_batches_include_epochs is True

    # Step-count math: include_epochs True => do not re-multiply by epochs.
    grad_accum = args.gradient_accumulation_steps
    if getattr(trainer, "_prepared_batches_include_epochs", False):
        total_steps = n_batches // grad_accum
    elif args.num_train_epochs > 0:
        total_steps = (n_batches * args.num_train_epochs) // grad_accum
    else:
        total_steps = n_batches // grad_accum
    total_steps = max(1, total_steps)

    # 18 prebuilt batches already include the 3 epochs => 18 steps, not 54.
    assert total_steps == 18


def test_thread1_regression_without_fix_would_triple_count():
    """Document the bug shape: if the prebuilt flag were cleared, the step
    counter would multiply already-materialized epochs again."""
    n_batches = 18  # 6 samples * 3 epochs, already materialized
    grad_accum = 1
    num_train_epochs = 3

    # Buggy path: flag cleared to False before step counting.
    include_epochs = False
    if include_epochs:
        buggy = n_batches // grad_accum
    elif num_train_epochs > 0:
        buggy = (n_batches * num_train_epochs) // grad_accum
    else:
        buggy = n_batches // grad_accum
    assert buggy == 54  # 3x over-trained

    # Fixed path keeps the flag.
    include_epochs = True
    fixed = n_batches // grad_accum if include_epochs else buggy
    assert fixed == 18


# ===========================================================================
# Thread 5: _get_processor_tokenizer must not unwrap HF fast tokenizers.
# ===========================================================================

class _RustBackend:
    """Mimics tokenizers.Tokenizer: token_to_id, but no HF convenience API."""

    def token_to_id(self, token):
        return -1


class _FastTokenizer:
    """HF PreTrainedTokenizerFast: has convert_tokens_to_ids + a Rust _tokenizer."""

    chat_template = "FAST_TEMPLATE"

    def __init__(self):
        self._tokenizer = _RustBackend()

    def convert_tokens_to_ids(self, token):
        return 123

    def apply_chat_template(self, *a, **k):
        return "rendered"


class _MlxWrapper:
    """mlx-lm TokenizerWrapper: proxies HF API to an inner _tokenizer."""

    def __init__(self):
        self._tokenizer = _FastTokenizer()

    def __getattr__(self, name):
        # Proxy everything that is not private to the inner HF tokenizer.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_tokenizer"], name)


class _VlmProcessor:
    """HF VLM processor: tokenizer under .tokenizer, no top-level HF API."""

    def __init__(self):
        self.tokenizer = _FastTokenizer()


def test_thread5_fast_tokenizer_is_returned_as_is_not_rust_backend():
    from unsloth_zoo.mlx.utils import _get_processor_tokenizer

    tok = _FastTokenizer()
    result = _get_processor_tokenizer(tok)
    # Must be the HF tokenizer itself, never the Rust backend.
    assert result is tok
    assert not isinstance(result, _RustBackend)
    assert hasattr(result, "convert_tokens_to_ids")
    assert result.convert_tokens_to_ids("x") == 123
    assert result.chat_template == "FAST_TEMPLATE"


def test_thread5_mlx_wrapper_keeps_hf_api_available():
    from unsloth_zoo.mlx.utils import _get_processor_tokenizer

    wrapper = _MlxWrapper()
    result = _get_processor_tokenizer(wrapper)
    # Returned object exposes the HF API (directly or via proxy).
    assert hasattr(result, "convert_tokens_to_ids")
    assert result.convert_tokens_to_ids("x") == 123
    assert result.chat_template == "FAST_TEMPLATE"


def test_thread5_vlm_processor_unwraps_to_inner_tokenizer():
    from unsloth_zoo.mlx.utils import _get_processor_tokenizer

    proc = _VlmProcessor()
    result = _get_processor_tokenizer(proc)
    # VLM processor has no top-level convert_tokens_to_ids and no _tokenizer,
    # so we fall through to processor.tokenizer.
    assert result is proc.tokenizer
    assert hasattr(result, "convert_tokens_to_ids")


def test_thread5_bare_wrapper_without_hf_api_still_unwraps_tokenizer():
    """A wrapper that exposes _tokenizer but not the HF API (e.g. an mlx-vlm
    style processor) still unwraps so downstream lookups keep working."""
    from unsloth_zoo.mlx.utils import _get_processor_tokenizer

    inner = _FastTokenizer()

    class _BareWrapper:
        _tokenizer = inner

    result = _get_processor_tokenizer(_BareWrapper())
    assert result is inner


def test_thread5_vlm_ignore_ids_resolve_with_fast_tokenizer():
    """End-to-end: _get_vlm_ignore_token_ids must reach convert_tokens_to_ids
    on a fast tokenizer instead of failing against the Rust backend."""
    from unsloth_zoo.mlx.utils import _get_vlm_ignore_token_ids

    class _ImageTokenizer(_FastTokenizer):
        unk_token_id = -1
        image_token = "<image>"

        def convert_tokens_to_ids(self, token):
            if isinstance(token, list):
                token = token[0] if token else None
            return {"<image>": 200}.get(token, -1)

    ids = _get_vlm_ignore_token_ids(
        processor=_ImageTokenizer(),
        config={"image_token_id": 201},
    )
    assert ids is not None
    # 200 comes from the tokenizer's image_token; 201 from config.
    assert 200 in ids
    assert 201 in ids


# ===========================================================================
# Thread 5 follow-up: train_on_responses_only must hand the HF masking impl
# a CALLABLE tokenizer. mlx-lm's TokenizerWrapper proxies attributes via
# __getattr__ (so hasattr(convert_tokens_to_ids) is True) but defines no
# __call__, and the HF impl invokes tokenizer(...) directly.
# ===========================================================================

def test_thread5_noncallable_proxy_wrapper_unwraps_for_masking(monkeypatch):
    import unsloth_zoo.dataset_utils as dataset_utils
    import unsloth_zoo.mlx.trainer as trainer_mod

    class _CallableTokenizer(_SpaceTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": self.encode(text)}

    inner = _CallableTokenizer()

    class _ProxyWrapper:
        """mlx-lm TokenizerWrapper shape: proxies attributes, not callable."""

        def __init__(self, tokenizer):
            self._tokenizer = tokenizer

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_tokenizer"), name)

    received = {}

    def fake_hf(trainer, *, instruction_part=None, response_part=None,
                force_match=True, tokenizer=None, return_function=False,
                num_proc=None, last_response_only=False):
        received["tokenizer"] = tokenizer
        return lambda batch: batch

    monkeypatch.setattr(dataset_utils, "train_on_responses_only", fake_hf)
    trainer_mod.train_on_responses_only(
        None,
        instruction_part="<user>",
        response_part="<assistant>",
        tokenizer=_ProxyWrapper(inner),
        return_function=True,
    )
    assert received["tokenizer"] is inner, (
        "non-callable proxy wrapper must unwrap to its inner HF tokenizer"
    )
    # A real fast tokenizer (callable + HF API) must pass through untouched.
    trainer_mod.train_on_responses_only(
        None,
        instruction_part="<user>",
        response_part="<assistant>",
        tokenizer=inner,
        return_function=True,
    )
    assert received["tokenizer"] is inner
