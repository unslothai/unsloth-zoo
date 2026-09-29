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

"""Behavioral coverage for MLX GRPO on the torch-backed MLX shim."""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest

# The MLX simulation runs on torch; hosts with real MLX and no torch cannot build it.
pytest.importorskip("torch")


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    prefixes = ("mlx", "mlx_lm", "mlx_vlm")

    def _owned(name):
        return (
            name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx.")
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        )

    from mlx_simulation import restore_modules, simulate_mlx_on_torch, snapshot_modules
    from mlx_simulation.mlx_stub import _MLXFinder

    real_modules = snapshot_modules(_owned)
    simulate_mlx_on_torch()
    for name in list(sys.modules):
        if name == "unsloth_zoo.mlx" or name.startswith("unsloth_zoo.mlx."):
            sys.modules.pop(name, None)
    yield
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not isinstance(finder, _MLXFinder)
    ]
    restore_modules(real_modules, _owned)


EOS = 2


class Tokenizer:
    bos_token = None
    eos_token_id = EOS
    pad_token_id = 0

    def __init__(self):
        self.special_flags = []

    def encode(self, text, add_special_tokens=True):
        self.special_flags.append(add_special_tokens)
        return [3 + (ord(character) % 43) for character in text]

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False,
        continue_final_message=False, **kwargs,
    ):
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")


class FakeGenerator:
    """Stands in for generate_batch: one scripted completion per request, in order."""

    def __init__(self, script=None):
        self.script = script
        self.calls = []

    def __call__(self, model, tokenizer, requests, *, defaults=None):
        from unsloth_zoo.mlx.generate import GenerationResult

        self.calls.append(requests)
        results = []
        for index, request in enumerate(requests):
            if self.script is not None:
                ids, reason = self.script[index % len(self.script)]
            else:
                ids = [5 + (request.sampling.seed + step) % 30 for step in range(1 + index % 3)]
                reason = "stop"
            ids = list(ids)[: request.max_tokens]
            results.append(GenerationResult(
                token_ids=ids,
                text="".join(chr(97 + token % 26) for token in ids),
                logprobs=[0.0] * len(ids),
                finish_reason=reason,
                stop_token_id=EOS if reason == "stop" else None,
            ))
        return results


def rows(count=3):
    return [{"prompt": f"q{index}:", "answer": str(index)} for index in range(count)]


def _tiny_model(lora=False):
    import mlx.nn as nn
    import mlx.core as mx

    class Adapter:
        def __init__(self):
            self.lora_a = mx.array([[1.0]])
            self.lora_b = mx.array([[0.5]])
            self.scale = 1.0

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 4)
            self.proj = nn.Linear(4, 64, bias=False)
            self._config = {"model_type": "tiny"}
            if lora:
                self.q_proj = Adapter()

        def __call__(self, tokens):
            hidden = self.embed(tokens)
            if lora:
                hidden = hidden * (1 + self.q_proj.scale * (self.q_proj.lora_a * self.q_proj.lora_b).sum())
            return self.proj(hidden)

        training = True

        def train(self, mode=True):
            self.training = mode
            return self

        def eval(self):
            self.training = False

        @property
        def state(self):
            return []

        if lora:
            def named_modules(self):
                return [("", self), ("q_proj", self.q_proj)]

            def parameters(self):
                return {"q_proj": {"lora_a": self.q_proj.lora_a, "lora_b": self.q_proj.lora_b}}

            def trainable_parameters(self):
                return self.parameters()

    return Model()


def _config(tmp_path, **overrides):
    from unsloth_zoo.mlx.trainer import MLXGRPOConfig

    options = dict(
        max_steps=2, gradient_accumulation_steps=1, per_device_train_batch_size=2,
        num_generations=2, max_completion_length=4, max_seq_length=32,
        compile=False, gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False, disable_memory_limits=True,
        max_grad_norm=0.0, max_grad_leaf_norm=0.0, logging_steps=1,
        output_dir=str(tmp_path),
    )
    options.update(overrides)
    return MLXGRPOConfig(**options)


def _trainer(tmp_path, reward=None, dataset=None, model=None, **overrides):
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    def length_reward(completions, **kwargs):
        return [float(len(text)) for text in completions]

    return MLXGRPOTrainer(
        model or _tiny_model(), Tokenizer(), dataset or rows(),
        reward or length_reward, args=_config(tmp_path, **overrides),
    )


def _stub_optimizer(trainer, monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    trainer._build_optimizer = lambda _steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5), state={}, update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None


def test_group_advantages_use_the_unbiased_std_per_group():
    from unsloth_zoo.mlx.grpo import group_advantages

    rewards = [1.0, 2.0, 4.0, 3.0, 3.0, 3.0]
    got = group_advantages(rewards, 3)
    first = np.array(rewards[:3])
    expected = (first - first.mean()) / (first.std(ddof=1) + 1e-4)
    np.testing.assert_allclose(got[:3], expected)
    np.testing.assert_array_equal(got[3:], 0.0)


def _oracle_loss(logps, ref_logps, mask, advantages, beta):
    """fp64 TRL loss_type='grpo' at num_iterations=1 (ratio == 1)."""
    per_token = -advantages[:, None] * np.ones_like(logps)
    if beta:
        delta = ref_logps - logps
        per_token = per_token + beta * (np.expm1(delta) - delta)
    return float(((per_token * mask).sum(1) / np.maximum(mask.sum(1), 1)).mean())


@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_loss_matches_the_trl_formula(beta):
    import mlx.core as mx
    from unsloth_zoo.mlx import grpo
    from unsloth_zoo.mlx.preference import build_reference_policy

    model = _tiny_model(lora=True)
    reference = build_reference_policy(model, reference_free=False, resume_provenance=None)[0] if beta else None
    loss_fn = grpo.make_grpo_loss_fn(
        beta=beta, epsilon_low=0.2, epsilon_high=0.2, temperature=0.7,
        reference_policy=reference,
    )
    batch = mx.array([[3, 4, 5, 6, 0], [3, 4, 7, 8, 9]], dtype=mx.int32)
    lengths = mx.array([[2, 4], [2, 5]], dtype=mx.int32)
    advantages = mx.array([0.7, -0.7])
    rewards = mx.array([1.0, 0.0])
    loss, rows_, stats = loss_fn(model, batch, lengths, advantages, rewards)

    logps = np.array(grpo._token_logps(model, batch, 0.7).tolist(), dtype=np.float64)
    ref_logps = logps
    if beta:
        with reference.activate(model) as base:
            ref_logps = np.array(grpo._token_logps(base, batch, 0.7).tolist(), dtype=np.float64)
        assert not np.allclose(ref_logps, logps)
    mask = np.array([[0, 1, 1, 0], [0, 1, 1, 1]], dtype=np.float64)
    expected = _oracle_loss(logps, ref_logps, mask, np.array([0.7, -0.7]), beta)
    assert math.isclose(float(loss.item()), expected, rel_tol=1e-5, abs_tol=1e-6)
    assert int(rows_.item()) == 2
    names = loss_fn._unsloth_preference_metrics
    values = stats.tolist()
    assert values[names.index("reward")] == 1.0
    assert values[-1 if not beta else 3] == 2.0
    if beta:
        delta = ref_logps - logps
        kl = ((np.expm1(delta) - delta) * mask).sum()
        assert math.isclose(values[1], kl, rel_tol=1e-4)


def test_temperature_scales_the_scored_logits():
    import mlx.core as mx
    from unsloth_zoo.mlx import grpo

    model = _tiny_model()
    batch = mx.array([[3, 4, 5, 6]], dtype=mx.int32)
    logits = np.array(model(batch[:, :-1]).tolist(), dtype=np.float64) / 0.5
    targets = [4, 5, 6]
    expected = [
        logits[0, i, t] - np.log(np.exp(logits[0, i]).sum()) for i, t in enumerate(targets)
    ]
    np.testing.assert_allclose(
        np.array(grpo._token_logps(model, batch, 0.5).tolist())[0], expected, rtol=1e-5,
    )


def test_prompt_indices_cover_each_epoch_once_across_ranks():
    from unsloth_zoo.mlx.grpo import rollout_prompt_indices

    options = dict(num_prompts=8, prompts_per_batch=2, world=2, seed=3, shuffle=True)
    for epoch in range(2):
        seen = []
        for step in range(2):
            for rank in range(2):
                seen += rollout_prompt_indices(epoch * 2 + step, rank=rank, **options)
        assert sorted(seen) == list(range(8))
    assert rollout_prompt_indices(0, rank=0, **options) != rollout_prompt_indices(2, rank=0, **options)
    sequential = dict(options, shuffle=False, world=1)
    assert rollout_prompt_indices(4, rank=0, **sequential) == [0, 1]


def test_rollout_keeps_the_stop_token_and_passes_trl_reward_kwargs(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    fake = FakeGenerator(script=[([7, 8], "stop"), ([9, 10, 11, 12], "length")])
    monkeypatch.setattr(grpo, "generate_batch", fake)
    seen = {}

    def strict(completions, prompts, answer):
        seen["strict"] = (list(completions), list(prompts), list(answer))
        return [1.0, None]

    def open_kwargs(completions, completion_ids, trainer_state, **kwargs):
        seen["open"] = (completion_ids, trainer_state, sorted(kwargs))
        return [0.5, 2.0]

    trainer = _trainer(tmp_path, reward=[strict, open_kwargs], dataset=rows(1))
    trainer.args.reward_weights = None
    batch, lengths, advantages, rewards = next(trainer._prepare_data(False)[1])
    prompt = [3 + ord(c) % 43 for c in "q0:"]
    assert batch.tolist()[0][:6] == prompt + [7, 8, EOS]
    assert batch.tolist()[1][:7] == prompt + [9, 10, 11, 12]
    assert lengths.tolist() == [[3, 6], [3, 7]]
    assert rewards.tolist() == [1.5, 2.0]
    assert seen["strict"][1:] == (["q0:", "q0:"], ["0", "0"])
    assert seen["open"][0] == [[7, 8, EOS], [9, 10, 11, 12]]
    assert seen["open"][1] is trainer.state
    assert seen["open"][2] == ["answer", "prompts"]
    # Latest TRL tokenizes a plain-text prompt with the tokenizer's specials.
    assert all(flag is True for flag in trainer.tokenizer.special_flags)
    assert batch.shape[1] == 32
    requests = fake.calls[0]
    assert {r.sampling.seed for r in requests} and len({r.sampling.seed for r in requests}) == 2


def test_truncated_completions_can_be_masked(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    monkeypatch.setattr(grpo, "generate_batch", FakeGenerator(
        script=[([7], "stop"), ([9, 10, 11, 12], "length")],
    ))
    trainer = _trainer(tmp_path, dataset=rows(1), mask_truncated_completions=True)
    _, lengths, _, _ = next(trainer._prepare_data(False)[1])
    assert lengths.tolist() == [[3, 5], [7, 7]]


def test_chat_prompts_render_and_reward_as_messages(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    monkeypatch.setattr(grpo, "generate_batch", FakeGenerator(script=[([7], "stop")]))
    seen = []

    def reward(completions, prompts, **kwargs):
        seen.append((completions, prompts))
        return [0.0, 1.0]

    chat = [{"role": "user", "content": "hi"}]
    trainer = _trainer(tmp_path, reward=reward, dataset=[{"prompt": chat}])
    batch, lengths, _, _ = next(trainer._prepare_data(False)[1])
    rendered = [3 + ord(c) % 43 for c in "<user>hi<assistant>"]
    assert batch.tolist()[0][:len(rendered)] == rendered
    assert trainer.tokenizer.special_flags == [False]
    completions, prompts = seen[0]
    assert completions[0] == [{"role": "assistant", "content": "h"}]
    assert prompts == [chat, chat]


def test_reward_functions_must_score_every_completion(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    monkeypatch.setattr(grpo, "generate_batch", FakeGenerator())
    trainer = _trainer(tmp_path, reward=lambda completions, **kw: [1.0])
    with pytest.raises(ValueError, match="one per completion"):
        next(trainer._prepare_data(False)[1])


@pytest.mark.parametrize("overrides,message", [
    ({"num_generations": 1}, "num_generations"),
    ({"temperature": 0.0}, "temperature"),
    ({"loss_type": "dapo"}, "loss_type"),
    ({"scale_rewards": "batch"}, "scale_rewards"),
    ({"per_device_train_batch_size": 3}, "multiple of num_generations"),
    ({"reward_weights": [1.0, 2.0]}, "reward_weights"),
])
def test_unsupported_configurations_fail_at_construction(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _trainer(tmp_path, **overrides)


def test_trainer_requires_callable_rewards_and_no_eval(tmp_path):
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    with pytest.raises(ValueError, match="callables"):
        MLXGRPOTrainer(_tiny_model(), Tokenizer(), rows(), [], args=_config(tmp_path))
    with pytest.raises(NotImplementedError, match="evaluation"):
        MLXGRPOTrainer(
            _tiny_model(), Tokenizer(), rows(), len, eval_dataset=rows(),
            args=_config(tmp_path),
        )


def test_the_dataset_is_not_rewritten_into_sft_rows(tmp_path):
    trainer = _trainer(tmp_path)
    assert trainer.train_dataset == rows()


def test_train_on_responses_only_is_rejected(tmp_path):
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    with pytest.raises(ValueError, match="GRPO"):
        train_on_responses_only(
            _trainer(tmp_path), instruction_part="<user>", response_part="<assistant>",
        )


def test_trainer_runs_and_logs_rewards(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    fake = FakeGenerator()
    monkeypatch.setattr(grpo, "generate_batch", fake)
    trainer = _trainer(tmp_path)
    _stub_optimizer(trainer, monkeypatch)
    result = trainer.train()
    assert result["train_steps"] == 2
    assert len(fake.calls) == 2
    logged = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert logged and all("reward" in entry and "kl" not in entry for entry in logged)
    assert all(entry["completions/mean_length"] > 0 for entry in logged)


def test_referenced_run_logs_kl(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    monkeypatch.setattr(grpo, "generate_batch", FakeGenerator())
    trainer = _trainer(tmp_path, model=_tiny_model(lora=True), beta=0.1)
    _stub_optimizer(trainer, monkeypatch)
    trainer.train()
    logged = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert logged and all(entry["kl"] >= 0 for entry in logged)


def test_epoch_budget_ceils_a_ragged_tail(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    fake = FakeGenerator()
    monkeypatch.setattr(grpo, "generate_batch", fake)
    trainer = _trainer(tmp_path, max_steps=0, num_train_epochs=1, gradient_accumulation_steps=2)
    _stub_optimizer(trainer, monkeypatch)
    result = trainer.train()
    assert result["train_steps"] == 2
    assert len(fake.calls) == 3


def test_a_resumed_stream_starts_where_the_run_stopped(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    monkeypatch.setattr(grpo, "generate_batch", FakeGenerator())
    trainer = _trainer(tmp_path, dataset=rows(5))
    _, stream = trainer._prepare_data(False)
    fresh = [next(stream) for _ in range(4)]
    resumed = next(trainer._rollout_batches(3))
    for got, expected in zip(resumed, fresh[3]):
        assert got.tolist() == expected.tolist()


def test_unscorable_completions_leave_the_baseline_and_get_no_advantage():
    from unsloth_zoo.mlx.grpo import group_advantages, score_rewards

    nan = float("nan")
    got = group_advantages([1.0, nan, 3.0, 2.0, nan, nan], 3)
    scored = np.array([1.0, 3.0])
    expected = (scored - scored.mean()) / (scored.std(ddof=1) + 1e-4)
    np.testing.assert_allclose(got[[0, 2]], expected)
    assert got[1] == 0.0
    np.testing.assert_array_equal(got[3:], 0.0)

    def first(completions, **kwargs):
        return [None, 1.0]

    def second(completions, **kwargs):
        return [None, None]

    rows = [{"prompt": "p"}]
    total = score_rewards([first, second], [1.0, 1.0], rows, [["a", "b"]], [[[1], [2]]], None)
    assert np.isnan(total[0]) and total[1] == 1.0


def test_completion_indices_cover_exactly_the_completion_targets():
    from unsloth_zoo.mlx.grpo import completion_indices

    got = completion_indices([[2, 4], [3, 3], [1, 5]], width=6, rows_multiple=8).tolist()
    assert got == [1, 2, 10, 11, 12, 13, -1, -1]


def test_a_scorer_scores_only_completions_on_the_tempered_head():
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx import grpo

    model = _tiny_model()
    calls = []

    def scorer(model_, batch, supervised, indices=None, *, hidden_scale=None):
        calls.append((supervised.tolist(), None if indices is None else indices.tolist(), hidden_scale))
        logits = model_(batch[:, :-1]) * (1.0 if hidden_scale is None else hidden_scale)
        ce = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none").reshape(supervised.shape)
        return ce * supervised, None

    scorer.compaction = True
    batch = mx.array([[3, 4, 5, 6, 0], [3, 4, 7, 8, 9]], dtype=mx.int32)
    lengths = mx.array([[2, 4], [2, 5]], dtype=mx.int32)
    args = (batch, lengths, mx.array([0.7, -0.7]), mx.array([1.0, 0.0]))
    options = dict(beta=0.0, epsilon_low=0.2, epsilon_high=0.2, temperature=0.5)
    dense = grpo.make_grpo_loss_fn(**options)(model, *args)[0]
    fn = grpo.make_grpo_loss_fn(scorer=scorer, **options)
    indices = grpo.completion_indices(lengths.tolist(), 5, rows_multiple=8)
    scored = fn(model, *args, indices)[0]
    assert math.isclose(float(scored.item()), float(dense.item()), rel_tol=1e-6, abs_tol=1e-7)
    assert fn._unsloth_cce_compaction and fn._unsloth_cce_backend == "runtime-cce"
    supervised, got_indices, hidden_scale = calls[0]
    assert supervised == [[False, True, True, False], [False, True, True, True]]
    assert got_indices == [1, 2, 5, 6, 7, -1, -1, -1] and hidden_scale == 2.0


def test_softcapped_heads_keep_the_dense_path_away_from_temperature_one(monkeypatch):
    from unsloth_zoo.mlx import grpo

    fake = types.SimpleNamespace(softcap=30.0)
    monkeypatch.setattr(grpo, "_make_preference_cce_scorer", lambda model: fake)
    assert grpo.make_grpo_scorer(object(), 0.7) is None
    assert grpo.make_grpo_scorer(object(), 1.0) is fake


def test_a_micro_batch_with_every_completion_masked_is_a_zero_step(tmp_path, monkeypatch):
    from unsloth_zoo.mlx import grpo

    monkeypatch.setattr(grpo, "generate_batch", FakeGenerator(script=[([9, 10, 11, 12], "length")]))
    trainer = _trainer(tmp_path, mask_truncated_completions=True)
    _stub_optimizer(trainer, monkeypatch)
    assert trainer.train()["train_steps"] == 2


@pytest.mark.parametrize("state,beta", [
    ({"global_step": 1}, 0.0),
    ({"global_step": 1, "preference_reference": {"kind": "reference_free"}}, 0.0),
    ({"global_step": 1, "preference_reference": {"kind": "grpo_no_reference", "objective": "grpo"}}, 0.1),
])
def test_resume_needs_a_grpo_checkpoint_with_the_same_reference_mode(tmp_path, monkeypatch, state, beta):
    import json

    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "adapters.safetensors").touch()
    (checkpoint / "optimizer_state.safetensors").touch()
    (checkpoint / "trainer_state.json").write_text(json.dumps(state))
    model = _tiny_model(lora=True)
    loaded = []
    model.load_weights = lambda *args, **kwargs: loaded.append(args)
    trainer = _trainer(tmp_path, model=model, beta=beta)
    _stub_optimizer(trainer, monkeypatch)
    with pytest.raises(ValueError, match="GRPO run"):
        trainer.train(resume_from_checkpoint=str(checkpoint))
    assert loaded == []
