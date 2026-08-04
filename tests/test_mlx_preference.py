"""Focused behavioral coverage for MLX ORPO and DPO."""

from __future__ import annotations

import math
import sys
import types

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    from mlx_simulation import simulate_mlx_on_torch
    from mlx_simulation.mlx_stub import _MLXFinder
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_modules)


class Tokenizer:
    bos_token = None
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [3 + (ord(character) % 43) for character in text]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False,
        continue_final_message=False, **kwargs,
    ):
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class MappingTokenizer:
    eos_token_id = None
    pad_token_id = 0

    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=True):
        return self.mapping[text]


def rows(count=6):
    return [
        {"prompt": f"question {index}: ", "chosen": "yes", "rejected": "no"}
        for index in range(count)
    ]


def build_plan(**kwargs):
    from unsloth_zoo.mlx.preference import create_preference_batch_plan
    options = dict(
        batch_size=2, max_seq_length=64, dataset_order="sequential",
        grad_accum=2,
    )
    options.update(kwargs)
    return create_preference_batch_plan(rows(), Tokenizer(), **options)


def test_configs_are_objective_specific_and_keyword_only():
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXORPOConfig, MLXTrainingConfig

    assert not hasattr(MLXTrainingConfig(), "beta")
    assert MLXORPOConfig(beta=0.2).beta == 0.2
    assert MLXDPOConfig(beta=0.3, label_smoothing=0.1).reference_free is False
    with pytest.raises(TypeError):
        MLXORPOConfig(*([None] * 100))


@pytest.mark.parametrize(
    "epochs,expected_batches,expected_steps",
    [(0.5, 2, 1), (1.0, 3, 2), (1.5, 5, 3), (2.0, 6, 4)],
)
def test_fractional_epoch_budget_matches_finite_sft_contract(
    epochs, expected_batches, expected_steps,
):
    from unsloth_zoo.mlx.trainer import _resolve_training_steps

    plan = build_plan(num_epochs=epochs, dataset_order="torch_randperm", seed=7)
    args = types.SimpleNamespace(
        max_steps=0, num_train_epochs=epochs, gradient_accumulation_steps=2,
    )
    assert len(plan) == expected_batches
    assert plan.cycle_length == 3
    assert _resolve_training_steps(
        args, plan, None, includes_epochs=True,
    ) == expected_steps


def test_plan_is_lazy_and_concatenates_chosen_then_rejected():
    import mlx.core as mx

    plan = build_plan(num_batches=2)
    assert all(not isinstance(value, mx.array) for value in plan.rows)
    batch, lengths, normalizers = plan[0]
    assert batch.shape[0] == 4
    assert lengths.tolist()[0][0] == lengths.tolist()[2][0]
    assert normalizers.tolist()[1:] == [4, 2]


def test_default_order_randomizes_length_batches_from_first_visit():
    from unsloth_zoo.mlx.preference import create_preference_batch_plan

    dataset = [
        {"prompt": "p", "chosen": "x" * size, "rejected": "y" * size}
        for size in range(1, 8)
    ]
    plan = create_preference_batch_plan(
        dataset, Tokenizer(), batch_size=2, max_seq_length=64,
        num_batches=4, grad_accum=1, dataset_order="default", seed=1,
    )
    assert plan.schedule == ((6,), (4, 5), (0, 1), (2, 3))


def test_prompt_only_eos_does_not_consume_first_response_token():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    class PromptEOSTokenizer(Tokenizer):
        def encode(self, text, add_special_tokens=True):
            values = super().encode(text, add_special_tokens)
            if text.endswith(": "):
                values.append(self.eos_token_id)
            return values

    tokenized = tokenize_preference_row(
        PromptEOSTokenizer(), rows(1)[0], max_seq_length=64,
    )
    expected_first = Tokenizer().encode("question 0: yes")[len(
        Tokenizer().encode("question 0: ")
    )]
    assert tokenized.chosen_ids[0] == expected_first


@pytest.mark.parametrize(
    "mapping,row,expected",
    [
        (
            {"ab": [1, 2], "abc": [1, 3, 4], "abd": [1, 5, 6]},
            {"prompt": "ab", "chosen": "c", "rejected": "d"},
            ((1, 3, 4), (1, 5, 6), (1, 1)),
        ),
        (
            {"a": [1], "ac": [2, 3], "ad": [4, 5]},
            {"prompt": "a", "chosen": "c", "rejected": "d"},
            ((2, 3), (4, 5), (0, 0)),
        ),
        (
            {"ab": [1, 2], "abc": [1, 2, 3], "abd": [1, 4, 5]},
            {"prompt": "ab", "chosen": "c", "rejected": "d"},
            ((1, 2, 3), (1, 4, 5), (2, 1)),
        ),
    ],
)
def test_boundary_merges_preserve_each_branch(mapping, row, expected):
    from unsloth_zoo.mlx.preference import (
        create_preference_batch_plan,
        tokenize_preference_row,
    )

    tokenizer = MappingTokenizer(mapping)
    tokenized = tokenize_preference_row(
        tokenizer, row, max_seq_length=8,
    )
    chosen, rejected, prompt_lengths = expected
    assert tokenized.chosen == chosen
    assert tokenized.rejected == rejected

    _, lengths, _ = create_preference_batch_plan(
        [row], tokenizer, batch_size=1, max_seq_length=8,
        num_batches=1, grad_accum=1, dataset_order="sequential",
    )[0]
    assert lengths.tolist() == [
        [prompt_lengths[0], len(chosen)],
        [prompt_lengths[1], len(rejected)],
    ]


def test_sft_and_preference_share_boundary_merge_tolerance():
    from unsloth_zoo.mlx.utils import _tokenize_mlx_prompt_completion

    input_ids, labels = _tokenize_mlx_prompt_completion(
        MappingTokenizer({"ab": [1, 2], "abc": [1, 3, 4]}),
        "ab", "c", completion_only_loss=True,
    )
    assert input_ids == [1, 3, 4]
    assert labels == [-100, 3, 4]


def test_prompt_mismatch_before_the_boundary_is_rejected():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    tokenizer = MappingTokenizer(
        {
                "abc": [1, 2, 3],
                "abcd": [9, 2, 4, 5],
                "abce": [8, 2, 6, 7],
        }
    )

    with pytest.raises(ValueError, match="differ before the final prompt token"):
        tokenize_preference_row(
            tokenizer,
            {"prompt": "abc", "chosen": "d", "rejected": "e"},
            max_seq_length=8,
        )


def test_long_prompt_truncation_keeps_response_and_prompt_end():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    row = {"prompt": "abcdefghijklmnopqrstuvwxyz", "chosen": "GOOD", "rejected": "BAD"}
    tokenized = tokenize_preference_row(Tokenizer(), row, max_seq_length=8)
    assert tokenized.chosen_prompt_ids and tokenized.rejected_prompt_ids
    assert tokenized.chosen_ids and tokenized.rejected_ids
    assert len(tokenized.chosen) <= 8
    assert len(tokenized.rejected) <= 8


def test_content_parts_and_chat_template_options_are_preserved():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    class CapturingTokenizer(Tokenizer):
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return super().apply_chat_template(messages, **kwargs)

    tokenizer = CapturingTokenizer()
    row = {
        "prompt": [{"role": "user", "content": [{"type": "text", "text": "Q"}]}],
        "chosen": [{"role": "assistant", "content": [{"type": "text", "text": "A"}]}],
        "rejected": [{"role": "assistant", "content": "B"}],
        "tools": [],
        "chat_template_kwargs": {"flag": 1},
    }
    result = tokenize_preference_row(tokenizer, row, max_seq_length=64)
    assert result.chosen != result.rejected
    assert all(call[1]["tools"] == [] and call[1]["flag"] == 1 for call in tokenizer.calls)
    assert tokenizer.calls[0][0][0]["content"] == "Q"


def test_assistant_ended_prompt_continues_the_same_message():
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    tokenizer = Tokenizer()
    row = {
        "prompt": [{"role": "assistant", "content": "prefix"}],
        "chosen": [{"role": "assistant", "content": " chosen"}],
        "rejected": [{"role": "assistant", "content": " rejected"}],
    }
    result = tokenize_preference_row(tokenizer, row, max_seq_length=64)
    assert (
        result.chosen_prompt_ids
        and result.rejected_prompt_ids
        and result.chosen_ids
        and result.rejected_ids
    )


@pytest.mark.parametrize(
    "row",
    [
        {"chosen": "a", "rejected": "b"},
        {"prompt": "p", "chosen": [{"role": "assistant", "content": "a"}], "rejected": "b"},
        {"prompt": [{"role": "system", "content": "p"}], "chosen": [], "rejected": []},
    ],
)
def test_ambiguous_rows_are_rejected(row):
    from unsloth_zoo.mlx.preference import tokenize_preference_row

    with pytest.raises(ValueError):
        tokenize_preference_row(Tokenizer(), row, max_seq_length=64)


class TinyModel:
    def __init__(self):
        import mlx.nn as nn
        self.embedding = nn.Embedding(64, 8)
        self.output = nn.Linear(8, 64, bias=False)

    def __call__(self, tokens):
        return self.output(self.embedding(tokens))


@pytest.mark.parametrize("objective", ["orpo", "dpo"])
def test_microbatch_loss_matches_one_logical_batch(objective):
    from unsloth_zoo.mlx.preference import make_dpo_loss_fn, make_orpo_loss_fn

    dataset = rows(3)
    from unsloth_zoo.mlx.preference import create_preference_batch_plan
    micro = create_preference_batch_plan(
        dataset, Tokenizer(), batch_size=2, max_seq_length=64,
        num_batches=2, grad_accum=2, dataset_order="sequential",
    )
    whole = create_preference_batch_plan(
        dataset, Tokenizer(), batch_size=3, max_seq_length=64,
        num_batches=1, grad_accum=1, dataset_order="sequential",
    )
    model = TinyModel()
    loss_fn = (
        make_orpo_loss_fn(beta=0.1)
        if objective == "orpo"
        else make_dpo_loss_fn(beta=0.1, reference_free=True)
    )
    micro_value = sum(float(loss_fn(model, *micro[index])[0]) for index in range(2)) / 2
    whole_value = float(loss_fn(model, *whole[0])[0])
    assert math.isclose(micro_value, whole_value, rel_tol=2e-5, abs_tol=2e-5)


def test_orpo_odds_term_is_finite_for_perfect_float16_logp():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import _orpo_terms

    value = _orpo_terms(
        mx.array([0.0], dtype=mx.float16),
        mx.array([-1.0], dtype=mx.float16),
    )
    assert math.isfinite(float(value[0]))
    assert value.dtype == mx.float32


def test_dpo_label_smoothing_matches_conservative_formula():
    from unsloth_zoo.mlx import preference

    plan = build_plan(num_batches=1, grad_accum=1)
    batch, lengths, normalizers = plan[0]
    model = TinyModel()
    epsilon = 0.2
    beta = 0.3
    policy = preference._response_logps(model, batch, lengths)
    pairs = batch.shape[0] // 2
    logits = beta * (policy[:pairs] - policy[pairs:])
    expected = -(
        (1 - epsilon) * preference._log_sigmoid(logits)
        + epsilon * preference._log_sigmoid(-logits)
    ).mean()
    actual, weight = preference.make_dpo_loss_fn(
        beta=beta, label_smoothing=epsilon, reference_free=True,
    )(model, batch, lengths, normalizers)
    assert float(actual) == pytest.approx(float(expected), rel=1e-6)
    assert int(weight) == 1


def test_orpo_nll_covers_the_full_chosen_sequence():
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx import preference

    plan = build_plan(num_batches=1, grad_accum=1)
    batch, lengths, normalizers = plan[0]
    mx.random.seed(0)
    model = TinyModel()
    beta = 0.15
    targets = batch[:, 1:]
    ce = nn.losses.cross_entropy(
        model(batch[:, :-1]), targets, reduction="none",
    ).reshape(targets.shape)
    pairs = batch.shape[0] // 2
    steps = mx.arange(1, targets.shape[1] + 1)
    response_mask = preference._response_mask(targets, lengths)
    response_logp = -(ce * response_mask).sum(axis=1) / mx.maximum(
        response_mask.sum(axis=1), mx.array(1.0),
    )
    odds = preference._orpo_terms(
        response_logp[:pairs], response_logp[pairs:],
    ).mean()
    full_mask = (steps < lengths[:pairs, 1:]).astype(mx.float32)
    full_nll = (ce[:pairs] * full_mask).sum() / full_mask.sum()
    response_nll = (
        ce[:pairs] * response_mask[:pairs]
    ).sum() / response_mask[:pairs].sum()
    actual, _ = preference.make_orpo_loss_fn(beta)(
        model, batch, lengths, normalizers,
    )
    assert float(actual) == pytest.approx(float(full_nll + beta * odds), rel=1e-6)
    assert not math.isclose(
        float(actual), float(response_nll + beta * odds), rel_tol=1e-4,
    )


def test_dropout_context_restores_exact_state():
    from unsloth_zoo.mlx.preference import PreferenceRunContext
    from unsloth_zoo.mlx.utils import _read_mlx_lora_dropout

    class Dropout:
        def __init__(self):
            self._p_1 = 0.75
            self.p = 0.25

    class Model:
        def __init__(self):
            self.dropout = Dropout()
            self.probability = types.SimpleNamespace(p=0.6)

        def named_modules(self):
            return [
                ("", self), ("dropout", self.dropout),
                ("probability", self.probability),
            ]

    model = Model()
    adapter = types.SimpleNamespace(dropout=model.dropout)
    before = (model.dropout._p_1, model.dropout.p)
    context = PreferenceRunContext(model)
    assert model.dropout._p_1 == 1.0
    assert model.probability.p == 0.6
    assert _read_mlx_lora_dropout(adapter) == 0.25
    context.restore()
    assert (model.dropout._p_1, model.dropout.p) == before


def test_adapter_config_keeps_dropout_while_preference_context_is_active():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import PreferenceRunContext
    from unsloth_zoo.mlx.utils import _enrich_mlx_adapter_config

    class Dropout:
        def __init__(self):
            self._p_1 = 0.7
            self.p = 0.3

    class Adapter:
        def __init__(self):
            self.lora_a = mx.zeros((8, 4))
            self.lora_b = mx.zeros((4, 16))
            self.scale = 2.0
            self.dropout = Dropout()

    class Model:
        def __init__(self):
            self.adapter = Adapter()

        def named_modules(self):
            return [
                ("", self), ("q_proj", self.adapter),
                ("q_proj.dropout", self.adapter.dropout),
            ]

    model = Model()
    context = PreferenceRunContext(model)
    try:
        config = _enrich_mlx_adapter_config(model, {})
        assert config["lora_parameters"]["dropout"] == pytest.approx(0.3)
        assert config["dropout"] == pytest.approx(0.3)
    finally:
        context.restore()


def test_referenced_dpo_rejects_non_lora_and_reference_free_accepts_it():
    from unsloth_zoo.mlx.preference import build_reference_policy

    class Model:
        _hf_repo = "base"

        def named_modules(self):
            return [("", self)]

        def parameters(self):
            return {}

        def trainable_parameters(self):
            return {}

    with pytest.raises(ValueError, match="requires plain LoRA"):
        build_reference_policy(
            Model(), reference_free=False, resume_provenance=None,
        )
    policy, provenance = build_reference_policy(
        Model(), reference_free=True, resume_provenance=None,
    )
    assert policy is None and provenance == {"kind": "reference_free"}


def test_reference_forward_restores_scale_and_neftune_after_failure(monkeypatch):
    from unsloth_zoo.mlx import preference

    adapter = types.SimpleNamespace(scale=0.75)
    neftune = types.SimpleNamespace(_neftune_noise_enabled=True)
    policy = preference.LoRAReferencePolicy([adapter], [neftune])
    monkeypatch.setattr(
        preference, "_response_logps",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("forward failed")),
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        policy.forward(object(), object(), object())
    assert adapter.scale == 0.75
    assert neftune._neftune_noise_enabled is True


def test_referenced_dpo_accepts_only_fresh_or_same_run_plain_lora():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import build_reference_policy

    class PlainLoRA:
        def __init__(self, second_nonzero=False):
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[1.0 if second_nonzero else 0.0]])
            self.scale = 1.0

    class Model:
        _hf_repo = "base/repo"
        _unsloth_base_revision = "main"
        _unsloth_base_commit_hash = "abc123"

        def __init__(self, adapter, extra=False):
            self.adapter = adapter
            self.extra = extra

        def named_modules(self):
            return [("", self), ("q_proj", self.adapter)]

        def parameters(self):
            values = {
                "q_proj": {
                    "lora_a": self.adapter.lora_a,
                    "lora_b": self.adapter.lora_b,
                }
            }
            if self.extra:
                values["norm"] = mx.array([1.0])
            return values

        def trainable_parameters(self):
            return self.parameters()

    fresh = Model(PlainLoRA())
    policy, provenance = build_reference_policy(
        fresh, reference_free=False, resume_provenance=None,
    )
    assert policy.modules == (fresh.adapter,)
    assert provenance["base_commit"] == "abc123"

    resumed = Model(PlainLoRA(second_nonzero=True))
    resumed_policy, resumed_provenance = build_reference_policy(
        resumed, reference_free=False, resume_provenance=provenance,
    )
    assert resumed_policy.modules == (resumed.adapter,)
    assert resumed_provenance == provenance

    with pytest.raises(ValueError, match="fresh zero-delta"):
        build_reference_policy(
            Model(PlainLoRA(second_nonzero=True)),
            reference_free=False,
            resume_provenance=None,
        )
    with pytest.raises(ValueError, match="LoRA-only trainable"):
        build_reference_policy(
            Model(PlainLoRA(), extra=True),
            reference_free=False,
            resume_provenance=None,
        )


def test_referenced_dpo_rejects_dora():
    import mlx.core as mx
    from unsloth_zoo.mlx.preference import build_reference_policy

    DoRA = type("DoRALinear", (), {})
    adapter = DoRA()
    adapter.lora_a = mx.array([[1.0]])
    adapter.lora_b = mx.array([[0.0]])
    adapter.scale = 1.0

    class Model:
        def named_modules(self):
            return [("", self), ("q_proj", adapter)]

        def parameters(self):
            return {"q_proj": {"lora_a": adapter.lora_a, "lora_b": adapter.lora_b}}

        trainable_parameters = parameters

    with pytest.raises(ValueError, match="does not support DoRA"):
        build_reference_policy(
            Model(), reference_free=False, resume_provenance=None,
        )


def test_preference_trainers_forward_shared_constructor_state():
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig, MLXDPOTrainer, MLXORPOConfig, MLXORPOTrainer,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._config = {"model_type": "tiny"}

    def formatter(row):
        return row
    processor = object()
    for trainer_class, config in (
        (MLXORPOTrainer, MLXORPOConfig()),
        (MLXDPOTrainer, MLXDPOConfig(reference_free=True)),
    ):
        original = config.to_dict()
        trainer = trainer_class(
            Model(), Tokenizer(), rows(1), args=config,
            formatting_func=formatter, processor=processor,
            max_seq_length=96,
        )
        assert trainer.args is not config
        assert config.to_dict() == original
        assert trainer.args.beta == config.beta
        assert trainer.args.max_seq_length == 96
        assert config.max_seq_length == original["max_seq_length"] == 2048
        assert trainer.formatting_func is formatter
        assert trainer.processor is processor


@pytest.mark.parametrize("objective", ["orpo", "dpo"])
def test_preference_trainer_runs_through_shared_training_loop(
    objective, monkeypatch, tmp_path,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.trainer import (
        MLXDPOConfig, MLXDPOTrainer, MLXORPOConfig, MLXORPOTrainer,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 4)
            self.proj = nn.Linear(4, 64, bias=False)
            self._config = {"model_type": "tiny"}

        def __call__(self, tokens):
            return self.proj(self.embed(tokens))

        def train(self, mode=True):
            return self

        @property
        def state(self):
            return []

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    common = dict(
        max_steps=1,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=2,
        compile=True,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        disable_memory_limits=True,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        logging_steps=1,
        output_dir=str(tmp_path),
    )
    if objective == "orpo":
        trainer = MLXORPOTrainer(
            Model(), Tokenizer(), rows(3), args=MLXORPOConfig(**common),
        )
    else:
        trainer = MLXDPOTrainer(
            Model(), Tokenizer(), rows(3),
            args=MLXDPOConfig(reference_free=True, **common),
        )
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None
    result = trainer.train()
    assert result["train_steps"] == 1
    assert result["trained_tokens"] > 0


def test_referenced_dpo_freezes_accidental_norm_before_reference_validation(
    monkeypatch, tmp_path,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    class Adapter:
        def __init__(self):
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[0.0]])
            self.scale = 1.0

    class Norm:
        def __init__(self):
            self.weight = mx.array([1.0])
            self.frozen = False

        def freeze(self, keys=None, recurse=False):
            assert keys == ["weight"] and recurse is False
            self.frozen = True

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 4)
            self.proj = nn.Linear(4, 64, bias=False)
            self.adapter = Adapter()
            self.norm = Norm()
            self._config = {"model_type": "tiny"}

        def __call__(self, tokens):
            return self.proj(self.embed(tokens))

        def train(self, mode=True):
            return self

        @property
        def state(self):
            return []

        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": self.adapter.lora_a,
                    "lora_b": self.adapter.lora_b,
                },
                "norm": {"weight": self.norm.weight},
            }

        def trainable_parameters(self):
            parameters = self.parameters()
            if self.norm.frozen:
                parameters.pop("norm")
            return parameters

        def named_modules(self):
            return [("", self), ("q_proj", self.adapter), ("norm", self.norm)]

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    model = Model()
    trainer = MLXDPOTrainer(
        model,
        Tokenizer(),
        rows(2),
        args=MLXDPOConfig(
            max_steps=1,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=1,
            compile=True,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            disable_memory_limits=True,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            logging_steps=1,
            output_dir=str(tmp_path),
        ),
    )
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None

    result = trainer.train()

    assert model.norm.frozen is True
    assert result["train_steps"] == 1


def test_final_loss_averages_logical_updates_with_ragged_epoch_tail(
    monkeypatch, tmp_path,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.preference import make_orpo_loss_fn
    from unsloth_zoo.mlx.trainer import MLXORPOConfig, MLXORPOTrainer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 4)
            self.proj = nn.Linear(4, 64, bias=False)
            self._config = {"model_type": "tiny"}

        def __call__(self, tokens):
            return self.proj(self.embed(tokens))

        def train(self, mode=True):
            return self

        @property
        def state(self):
            return []

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    config = MLXORPOConfig(
        max_steps=2, gradient_accumulation_steps=2,
        per_device_train_batch_size=2, dataset_order="sequential",
        compile=False, gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False, disable_memory_limits=True,
        max_grad_norm=0.0, max_grad_leaf_norm=0.0, logging_steps=10,
        output_dir=str(tmp_path),
    )
    model = Model()
    trainer = MLXORPOTrainer(model, Tokenizer(), rows(5), args=config)
    plan, _ = trainer._prepare_data(False)
    objective = make_orpo_loss_fn(config.beta)
    values = [float(objective(model, *plan[index])[0]) for index in range(3)]
    expected = ((values[0] + values[1]) / 2 + values[2]) / 2
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None
    result = trainer.train()
    assert result["train_steps"] == 2
    assert result["train_loss"] == pytest.approx(expected, rel=1e-6)


def test_preference_capabilities_fail_before_model_setup():
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXORPOConfig, MLXORPOTrainer

    class CapabilityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._config = {"model_type": "tiny"}

    trainer = MLXORPOTrainer(
        CapabilityModel(), Tokenizer(), rows(1), eval_dataset=rows(1),
        args=MLXORPOConfig(eval_steps=1),
    )
    with pytest.raises(ValueError, match="evaluation, best-model loading"):
        trainer._prepare_data(False)


def test_trainer_applies_preference_formatter_once_per_row():
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXORPOConfig, MLXORPOTrainer

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._config = {"model_type": "tiny"}

    calls = []

    def formatter(row):
        calls.append(row["id"])
        return {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }

    dataset = [dict(row, id=index) for index, row in enumerate(rows(3))]
    trainer = MLXORPOTrainer(
        Model(), Tokenizer(), dataset, formatting_func=formatter,
        args=MLXORPOConfig(max_steps=1, gradient_accumulation_steps=1),
    )
    trainer._prepare_data(False)
    assert calls == [0, 1, 2]


def test_mismatched_referenced_resume_rejects_before_adapter_hydration(tmp_path):
    import json
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, MLXDPOTrainer

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[0.0]])
            self.scale = 1.0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = Adapter()
            self._config = {"model_type": "tiny"}
            self._hf_repo = "base/repo"
            self._unsloth_base_revision = "main"
            self._unsloth_base_commit_hash = "abc123"
            self.loaded = False

        def named_modules(self):
            return [("", self), ("q_proj", self.adapter)]

        def parameters(self):
            return {
                "q_proj": {
                    "lora_a": self.adapter.lora_a,
                    "lora_b": self.adapter.lora_b,
                }
            }

        trainable_parameters = parameters

        def load_weights(self, *_args, **_kwargs):
            self.loaded = True
            self.adapter.lora_b = mx.array([[9.0]])

    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "adapters.safetensors").touch()
    (checkpoint / "optimizer_state.safetensors").touch()
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": 1,
        "preference_reference": {"kind": "reference_free"},
    }))
    model = Model()
    trainer = MLXDPOTrainer(
        model, Tokenizer(), rows(1),
        args=MLXDPOConfig(
            max_steps=2, compile=False, gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False, disable_memory_limits=True,
            output_dir=str(tmp_path / "output"),
        ),
    )
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda *_args: None,
    )
    with pytest.raises(ValueError, match="provenance does not match"):
        trainer.train(resume_from_checkpoint=str(checkpoint))
    assert model.loaded is False
    assert model.adapter.lora_b.tolist() == [[0.0]]
