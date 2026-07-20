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

"""Deeper MLX component exercises: trainer, compile discovery,
cce backward, and quantization helpers, beyond just imports.

If a test fails, the failing component identifies the next gap.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _install_shim():
    import sys
    shim_prefixes = ("mlx", "mlx_lm", "mlx_vlm")
    real_mlx_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
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
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in shim_prefixes)
        ):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if not isinstance(finder, _MLXFinder)
    ]
    sys.modules.update(real_mlx_modules)


# ---------------------------------------------------------------------------
# 1. MLXTrainingConfig: full surface check.
# ---------------------------------------------------------------------------

def test_mlx_training_config_is_dataclass_with_all_fields():
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig
    assert dataclasses.is_dataclass(MLXTrainingConfig)
    field_names = [f.name for f in dataclasses.fields(MLXTrainingConfig)]
    fields = set(field_names)
    # Required SFT-compat fields
    for must_have in (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_steps",
        "warmup_ratio",
        "learning_rate",
        "lr_scheduler_type",
        "optim",
        "weight_decay",
        "max_grad_norm",
        "max_grad_leaf_norm",
        "seed",
        "logging_steps",
        "output_dir",
        "max_seq_length",
        "use_cce",
        "compile",
        "gradient_checkpointing",
        "dataset_order",
        "preserve_dataset_order",
        "completion_only_loss",
        "assistant_only_loss",
    ):
        assert must_have in fields, f"missing field: {must_have}"
    # dataset_text_field follows the eval block; newer eval knobs (eg load_best_model_at_end)
    # may sit between them, so assert relative order rather than strict adjacency.
    assert field_names.index("dataset_text_field") > field_names.index("eval_steps")
    assert field_names[field_names.index("append_eos") + 1] == "train_on_completions"
    assert field_names.index("per_device_eval_batch_size") > field_names.index("vlm_chat_template")
    assert field_names.index("image_size") > field_names.index("vlm_chat_template")


def test_mlx_training_config_exposes_completion_only_loss():
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _text_assistant_only_loss_arg,
        _text_completion_only_loss_arg,
    )

    assert _text_completion_only_loss_arg(
        MLXTrainingConfig(completion_only_loss=False)
    ) is False
    assert _text_completion_only_loss_arg(
        MLXTrainingConfig(completion_only_loss=True)
    ) is True
    assert _text_completion_only_loss_arg(
        MLXTrainingConfig(train_on_completions=True)
    ) is True
    assert _text_assistant_only_loss_arg(
        MLXTrainingConfig(assistant_only_loss=True)
    ) is True
    assert _text_assistant_only_loss_arg(MLXTrainingConfig()) is False


def test_mlx_trainer_distributed_defaults_world_size_one():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self): return {}

    trainer = MLXTrainer(DummyModel(), None, [], args=MLXTrainingConfig())

    assert trainer._distributed_initialized is False
    assert trainer.distributed_rank == 0
    assert trainer.distributed_world_size == 1
    assert trainer.is_main_process is True
    assert trainer._distributed_result_fields() == {
        "distributed_world_size": 1,
        "distributed_rank": 0,
        "distributed_is_main_process": True,
    }


def test_mlx_trainer_distributed_state_uses_cached_group(monkeypatch):
    import unsloth_zoo.mlx.trainer as trainer_mod

    class FakeWorld:
        def rank(self): return 1
        def size(self): return 2

    calls = []
    def fake_init():
        calls.append("init")
        return FakeWorld()

    monkeypatch.setattr(trainer_mod.mx.distributed, "init", fake_init)
    trainer = trainer_mod.MLXTrainer.__new__(trainer_mod.MLXTrainer)

    assert trainer.distributed_world is trainer.distributed_world
    assert calls == ["init"]
    assert trainer.distributed_rank == 1
    assert trainer.distributed_world_size == 2
    assert trainer.is_main_process is False
    assert trainer._distributed_result_fields() == {
        "distributed_world_size": 2,
        "distributed_rank": 1,
        "distributed_is_main_process": False,
    }


@pytest.mark.parametrize("accepts_backend", [True, False])
def test_mlx_trainer_distributed_state_selects_jaccl_backend(monkeypatch, accepts_backend):
    import unsloth_zoo.mlx.trainer as trainer_mod

    class FakeWorld:
        def rank(self): return 1
        def size(self): return 2

    calls = []
    def fake_init(**kwargs):
        calls.append(kwargs)
        if kwargs and not accepts_backend:
            raise TypeError("init() got an unexpected keyword argument 'backend'")
        return FakeWorld()

    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "127.0.0.1:12345")
    monkeypatch.setenv("MLX_IBV_DEVICES", "/tmp/mlx-devices.json")
    monkeypatch.setattr(trainer_mod.mx.distributed, "init", fake_init)
    trainer = trainer_mod.MLXTrainer.__new__(trainer_mod.MLXTrainer)

    assert trainer.distributed_world is trainer.distributed_world
    assert trainer.distributed_rank == 1
    assert trainer.distributed_world_size == 2
    if accepts_backend:
        assert calls == [{"backend": "jaccl"}]
    else:
        assert calls == [{"backend": "jaccl"}, {}]


def test_distributed_text_batches_use_tokenizer_pad_without_global_rng():
    import numpy as np
    from unsloth_zoo.mlx.utils import _create_distributed_text_batches

    class FakeWorld:
        def rank(self): return 0
        def size(self): return 2

    class Tokenizer:
        pad_token_id = 99

    # Shortest row has 2 tokens so it survives the sub-two-token filter while
    # still being padded out to the block length, exercising the pad id path.
    dataset = [([5, 6], 0), ([7, 8, 9], 0)]
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)

    batches = _create_distributed_text_batches(
        dataset,
        batch_size=2,
        max_seq_length=8,
        seed=7,
        comm_group=FakeWorld(),
        tokenizer=Tokenizer(),
    )

    assert np.random.random(3) == pytest.approx(expected)
    rows = batches[0][0].tolist()
    assert rows[0][:2] == [5, 6]
    assert rows[0][2:] == [99] * (len(rows[0]) - 2)


def test_distributed_text_batches_filter_sub_two_token_rows():
    from unsloth_zoo.mlx.utils import _create_distributed_text_batches

    class FakeWorld:
        def rank(self): return 0
        def size(self): return 2

    class Tokenizer:
        pad_token_id = 99

    # The length-1 row (token 5) has no causal target and must be filtered, so
    # every batch is drawn only from the length-2 row (tokens 6, 7).
    dataset = [([5], 0), ([6, 7], 0)]
    batches = _create_distributed_text_batches(
        dataset,
        batch_size=2,
        max_seq_length=8,
        num_batches=3,
        seed=7,
        comm_group=FakeWorld(),
        tokenizer=Tokenizer(),
    )

    assert len(batches) == 3
    for batch in batches:
        for row in batch[0].tolist():
            content = [tok for tok in row if tok != 99]
            assert content == [6, 7]


def test_distributed_text_batches_use_token_length_not_cache_itemlen(monkeypatch):
    # Regression: real mlx_lm CacheDataset.itemlen returns len(raw_row); for the
    # {"text": ...} rows _prepare_dataset builds that is the dict key count (1),
    # so an itemlen-based sub-two-token filter would drop every row and raise.
    # The filter must measure the processed token length instead.
    import sys

    from unsloth_zoo.mlx.utils import _create_distributed_text_batches

    class FakeWorld:
        def rank(self): return 0
        def size(self): return 2

    class Tokenizer:
        pad_token_id = 99

    class CacheDataset:
        def __init__(self, rows):
            self._rows = rows
            self._proc = {}

        def __len__(self):
            return len(self._rows)

        def itemlen(self, idx):
            # Matches real mlx_lm: length of the RAW row (dict key count == 1).
            return len(self._rows[idx])

        def __getitem__(self, idx):
            if idx not in self._proc:
                self._proc[idx] = (self._rows[idx]["ids"], 0)
            return self._proc[idx]

    monkeypatch.setattr(
        sys.modules["mlx_lm.tuner.datasets"], "CacheDataset", CacheDataset
    )

    dataset = CacheDataset([{"ids": [5, 6]}, {"ids": [7, 8, 9]}])
    # itemlen reports 1 for each row; an itemlen-based filter would drop both.
    assert dataset.itemlen(0) == 1

    batches = _create_distributed_text_batches(
        dataset,
        batch_size=2,
        max_seq_length=8,
        num_batches=2,
        seed=7,
        comm_group=FakeWorld(),
        tokenizer=Tokenizer(),
    )

    assert len(batches) == 2
    content = {
        tuple(tok for tok in row if tok != 99)
        for batch in batches
        for row in batch[0].tolist()
    }
    # Rows survived the >=2-token filter (token length, not itemlen).
    assert (5, 6) in content or (7, 8, 9) in content


@pytest.mark.parametrize("optim_name", ["adamw", "adam", "sgd", "adafactor"])
def test_mlx_training_config_each_optim(optim_name):
    """Every supported optim string constructs cleanly in config."""
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig
    cfg = MLXTrainingConfig(optim=optim_name)
    assert cfg.optim == optim_name


def test_trainer_drives_dynamic_lr_outside_optimizer_scheduler():
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
    )

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_steps=5,
    )
    schedule = trainer._build_schedule(total_steps=8)
    def value_at(step):
        value = schedule(step)
        return value.item() if hasattr(value, "item") else float(value)

    assert value_at(0) == pytest.approx(0.0)
    assert value_at(1) > value_at(0)
    assert value_at(4) < trainer.args.learning_rate
    assert value_at(5) == pytest.approx(trainer.args.learning_rate)

    trainer.model = object()
    optimizer = trainer._build_optimizer(total_steps=8)
    assert not callable(optimizer.learning_rate)
    first_lr = float(optimizer.learning_rate)
    trainer._set_optimizer_lr_for_step(optimizer, 1)
    second_lr = float(optimizer.learning_rate)
    assert second_lr > first_lr

    ratio_trainer = MLXTrainer.__new__(MLXTrainer)
    ratio_trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
    )
    ratio_schedule = ratio_trainer._build_schedule(total_steps=8)
    assert ratio_trainer._resolve_warmup_steps(total_steps=8) == 1
    assert ratio_schedule(0).item() < ratio_trainer.args.learning_rate
    assert ratio_schedule(1).item() == pytest.approx(
        ratio_trainer.args.learning_rate,
    )

    copied_ratio_trainer = MLXTrainer.__new__(MLXTrainer)
    copied_ratio_trainer.args = dataclasses.replace(
        MLXTrainingConfig(learning_rate=5e-5, lr_scheduler_type="linear"),
        warmup_ratio=0.1,
    )
    assert copied_ratio_trainer._resolve_warmup_steps(total_steps=100) == 10

    from unsloth_zoo.mlx.trainer import _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    legacy_fields = [
        field for field in dataclasses.fields(MLXTrainingConfig)
        if field.init and field.name not in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    ]
    legacy_values = [getattr(MLXTrainingConfig(), field.name) for field in legacy_fields]
    legacy_values[[field.name for field in legacy_fields].index("warmup_ratio")] = 0.1
    legacy_values[-1] = (128, 256)
    copied_ratio_trainer.args = MLXTrainingConfig(*legacy_values)
    assert copied_ratio_trainer.args.image_size == (128, 256)
    assert copied_ratio_trainer._resolve_warmup_steps(total_steps=100) == 10

    explicit_default_trainer = MLXTrainer.__new__(MLXTrainer)
    explicit_default_trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_steps=5,
        warmup_ratio=0.1,
    )
    assert explicit_default_trainer._resolve_warmup_steps(total_steps=8) == 5

    clamped_trainer = MLXTrainer.__new__(MLXTrainer)
    clamped_trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_ratio=2.0,
    )
    assert clamped_trainer._resolve_warmup_steps(total_steps=8) == 8

    # Explicit warmup_steps=0 must not disable a positive warmup_ratio (HF parity):
    # a zero step count means "use the ratio", not "no warmup".
    zero_steps_ratio_trainer = MLXTrainer.__new__(MLXTrainer)
    zero_steps_ratio_trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_steps=0,
        warmup_ratio=0.1,
    )
    assert zero_steps_ratio_trainer._resolve_warmup_steps(total_steps=100) == 10


def test_adamw_weight_decay_uses_hf_bias_norm_filter():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(
        optim="adamw",
        weight_decay=0.1,
    )

    optimizer = trainer._build_optimizer(total_steps=8)

    assert trainer._manual_weight_decay == pytest.approx(0.1)
    if hasattr(optimizer, "_kw"):
        assert optimizer._kw["weight_decay"] == 0.0
    assert MLXTrainer._should_apply_weight_decay("layers.0.mlp.down_proj.weight")
    assert not MLXTrainer._should_apply_weight_decay("layers.0.mlp.down_proj.bias")
    assert not MLXTrainer._should_apply_weight_decay("layers.0.input_layernorm.weight")
    assert not MLXTrainer._should_apply_weight_decay("vision.blocks.0.norm1.weight")


@pytest.mark.parametrize("optim_name", ["muon", "lion"])
def test_decoupled_optimizers_use_hf_parity_manual_decay(optim_name):
    """Muon and Lion mirror the AdamW pattern: zero out the optimizer's
    built-in `weight_decay` and let `_apply_manual_weight_decay` own the
    decoupled decay so bias and norm params are excluded."""
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(
        optim=optim_name,
        weight_decay=0.05,
    )

    optimizer = trainer._build_optimizer(total_steps=4)

    assert trainer._manual_weight_decay == pytest.approx(0.05)
    assert trainer._coupled_weight_decay == pytest.approx(0.0)
    if hasattr(optimizer, "_kw"):
        assert optimizer._kw["weight_decay"] == 0.0


def test_sgd_weight_decay_is_coupled_not_decoupled():
    """SGD must use coupled decay (folded into the gradient before momentum)
    to match HF/PyTorch SGD, not the AdamW-style decoupled parameter shrink."""
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = DummyModel()
    trainer.args = MLXTrainingConfig(optim="sgd", weight_decay=0.05)

    optimizer = trainer._build_optimizer(total_steps=4)

    assert trainer._coupled_weight_decay == pytest.approx(0.05)
    assert trainer._manual_weight_decay == pytest.approx(0.0)
    if hasattr(optimizer, "_kw"):
        assert optimizer._kw["weight_decay"] == 0.0


def test_norm_clip_dtype_restore_keeps_lora_and_norms_promotable():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    def should_restore_original_dtype(name):
        return (
            not MLXTrainer._is_norm_parameter_name(name)
            and not MLXTrainer._is_lora_parameter_name(name)
        )

    assert should_restore_original_dtype("model.layers.0.mlp.down_proj.weight")
    assert not should_restore_original_dtype("model.layers.0.self_attn.q_proj.lora_a")
    assert not should_restore_original_dtype("model.layers.0.self_attn.q_proj.lora_b")
    assert not should_restore_original_dtype("model.layers.0.input_layernorm.weight")
    assert not should_restore_original_dtype("vision.blocks.0.norm1.weight")


def test_global_norm_clip_reduces_in_float32():
    import inspect

    from unsloth_zoo.mlx.trainer import _clip_grad_norm_fp32

    source = inspect.getsource(_clip_grad_norm_fp32)

    assert "g.astype(mx.float32)" in source
    assert "scale.astype(g.dtype)" in source
    assert "tree_reduce" in source


@pytest.mark.parametrize(
    ("scheduler", "warmup"),
    [
        ("linear", 0),
        ("linear", 5),
        ("cosine", 0),
        ("cosine", 5),
        ("constant", 0),
        ("constant", 5),
    ],
)
def test_scheduler_lr_matches_expected_optimizer_update_steps(scheduler, warmup):
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    total_steps = 8
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(
        learning_rate=5e-5,
        lr_scheduler_type=scheduler,
        warmup_steps=warmup,
    )
    schedule = trainer._build_schedule(total_steps=total_steps)

    if callable(schedule):
        raw_values = [schedule(step) for step in range(total_steps)]
    else:
        raw_values = [schedule] * total_steps
    values = [
        value.item() if hasattr(value, "item") else float(value)
        for value in raw_values
    ]

    if scheduler == "linear" and warmup == 0:
        # Match `transformers.get_scheduler("linear", num_warmup_steps=0,
        # num_training_steps=total_steps)` as seen by optimizer steps across
        # Transformers 4.56.1 through 5.5.0: step 1 uses base LR, then decays.
        lr = trainer.args.learning_rate
        expected = [lr * (total_steps - step) / total_steps for step in range(total_steps)]
        assert values == pytest.approx(expected)
    elif warmup > 0:
        assert values[0] == pytest.approx(0.0)
        assert all(value > 0.0 for value in values[1:])
    else:
        assert all(value > 0.0 for value in values)


def test_mlx_text_dataset_does_not_append_eos(monkeypatch):
    """Unsloth formatting owns EOS decisions; MLX batching must not add one."""
    import sys

    class CacheDataset:
        def __init__(self, data):
            self._data = data
            self._cache = {}

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            if idx not in self._cache:
                self._cache[idx] = self._data.process(self._data[idx])
            return self._cache[idx]

        def itemlen(self, idx):
            return len(self[idx][0])

    monkeypatch.setattr(sys.modules["mlx_lm.tuner.datasets"], "CacheDataset", CacheDataset)

    from unsloth_zoo.mlx.utils import _prepare_dataset

    class Tokenizer:
        eos_token_id = 99
        chat_template = None

        def encode(self, text):
            assert text == "hello"
            return [1, 2, 3]

    # append_eos=False is what Unsloth passes (chat-template renders EOS).
    dataset = _prepare_dataset([{"text": "hello"}], Tokenizer(), append_eos=False)
    assert dataset[0] == ([1, 2, 3], 0)

    # Default (mlx-lm parity for direct MLX text fine-tuning callers)
    # appends the tokenizer EOS so a raw `{"text": str}` row still
    # trains the model to predict EOS.
    dataset_default = _prepare_dataset([{"text": "hello"}], Tokenizer())
    assert dataset_default[0] == ([1, 2, 3, 99], 0)


def test_encode_mlx_text_keeps_raw_text_bos_when_template_has_bos():
    from unsloth_zoo.mlx.utils import encode_mlx_text

    class Tokenizer:
        bos_token = "<s>"
        chat_template = "{{ bos_token }}{{ messages }}"

        def __init__(self):
            self.add_special_tokens_seen = []

        def encode(self, text, add_special_tokens=True):
            self.add_special_tokens_seen.append(add_special_tokens)
            return [1, 2, 3]

    tokenizer = Tokenizer()

    encode_mlx_text(tokenizer, "raw text")
    encode_mlx_text(tokenizer, "<s>rendered text")

    assert tokenizer.add_special_tokens_seen == [True, False]


def _make_mlx_text_trainer(**config_kwargs):
    """Build the smallest MLXTrainer shell needed for data-routing tests."""
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    class Tokenizer:
        chat_template = None

        def encode(self, text, add_special_tokens=True):
            return [1, 2]
    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(**config_kwargs)
    trainer.model = types.SimpleNamespace(_config={})
    trainer.tokenizer = Tokenizer()
    trainer.train_dataset = []
    trainer.formatting_func = None
    trainer._batches = None
    return MLXTrainer, trainer


def test_text_prompt_completion_create_batches_masks_prompt_labels_and_eos():
    from unsloth_zoo.mlx.utils import create_batches

    tokenizer = types.SimpleNamespace(
        chat_template=None,
        eos_token_id=99,
        encode=lambda text, add_special_tokens=True: [
            int(part) for part in str(text).split()
        ],
    )

    batch, _, labels = create_batches(
        dataset=[{"prompt": "1 2", "completion": " 3 4"}],
        tokenizer=tokenizer,
        batch_size=1,
        max_seq_length=8,
        seed=0,
    )[0]

    assert batch.tolist() == [[1, 2, 3, 4, 99]]
    assert labels.tolist() == [[-100, -100, 3, 4, 99]]


def test_text_conversational_prompt_completion_uses_generation_boundary():
    from unsloth_zoo.mlx.utils import create_batches

    class BatchEncoding(dict): pass

    class Tokenizer:
        chat_template = "{{ messages }}"
        eos_token_id = 99

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
            return_dict=False,
            tools=None,
            extra_token=0,
        ):
            ids = ([30] if tools else []) + ([extra_token] if extra_token else [])
            for message in messages:
                ids.append(10 if message["role"] == "user" else 20)
                ids.extend(int(part) for part in message["content"].split())
            if add_generation_prompt:
                ids.append(20)
            return BatchEncoding(input_ids=ids) if return_dict else ids

    batch, _, labels = create_batches(
        dataset=[
            {
                "prompt": [{"role": "user", "content": "1 2"}],
                "completion": [{"role": "assistant", "content": "3 4"}],
                "tools": [{"type": "function"}],
                "chat_template_kwargs": {"extra_token": 5},
            }
        ],
        tokenizer=Tokenizer(),
        batch_size=1,
        max_seq_length=10,
        seed=0,
        append_eos=False,
    )[0]

    assert batch.tolist() == [[30, 5, 10, 1, 2, 20, 3, 4]]
    assert labels.tolist() == [[-100, -100, -100, -100, -100, -100, 3, 4]]


class _AssistantMaskTokenizer:
    chat_template = "{{ messages }}"
    eos_token_id = None
    pad_token_id = 7

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        tools=None,
        add_generation_prompt=False,
        **_kwargs,
    ):
        ids = []
        masks = []
        if tools:
            ids.append(30)
            masks.append(0)
        for message in messages:
            is_assistant = message["role"] == "assistant"
            ids.append(20 if is_assistant else 10)
            masks.append(0)
            ids.extend(int(part) for part in message["content"].split())
            masks.extend([1 if is_assistant else 0] * len(message["content"].split()))
        output = {"input_ids": ids}
        if return_assistant_tokens_mask:
            output["assistant_masks"] = masks
        return output if return_dict else ids


class _NoAssistantMaskTokenizer(_AssistantMaskTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        kwargs["return_assistant_tokens_mask"] = False
        return super().apply_chat_template(*args, **kwargs)


@pytest.mark.parametrize(
    ("dataset", "extra_kwargs"),
    [
        (
            [
                {
                    "messages": [
                        {"role": "user", "content": "1"},
                        {"role": "assistant", "content": "2 3"},
                    ],
                }
            ],
            {},
        ),
        (
            [
                {
                    "prompt": [{"role": "user", "content": "1"}],
                    "completion": [{"role": "assistant", "content": "2 3"}],
                }
            ],
            {"append_eos": False},
        ),
    ],
)
def test_text_assistant_only_loss_masks_non_assistant_tokens(dataset, extra_kwargs):
    from unsloth_zoo.mlx.utils import create_batches

    batch, _, labels = create_batches(
        dataset=dataset,
        tokenizer=_AssistantMaskTokenizer(),
        batch_size=1,
        max_seq_length=8,
        assistant_only_loss=True,
        completion_only_loss=False,
        **extra_kwargs,
    )[0]

    assert batch.tolist() == [[10, 1, 20, 2, 3]]
    assert labels.tolist() == [[-100, -100, -100, 2, 3]]


@pytest.mark.parametrize(
    ("dataset", "tokenizer", "match"),
    [
        ([{"prompt": "Question: ", "completion": "Answer"}], _AssistantMaskTokenizer(), "not conversational"),
        (
            [
                {
                    "messages": [
                        {"role": "user", "content": "1"},
                        {"role": "assistant", "content": "2"},
                    ],
                },
                {"text": "plain text"},
            ],
            _AssistantMaskTokenizer(),
            "not conversational",
        ),
        (
            [
                {
                    "messages": [
                        {"role": "user", "content": "1"},
                        {"role": "assistant", "content": "2"},
                    ],
                }
            ],
            _NoAssistantMaskTokenizer(),
            "no assistant tokens",
        ),
        ([{"input_ids": [1, 2, 3]}], types.SimpleNamespace(), "assistant_masks"),
    ],
)
def test_text_assistant_only_loss_rejects_unsupported_inputs(dataset, tokenizer, match):
    from unsloth_zoo.mlx.utils import create_batches

    with pytest.raises((RuntimeError, ValueError), match=match):
        create_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=1,
            max_seq_length=8,
            assistant_only_loss=True,
            completion_only_loss=False,
        )


def test_text_pretokenized_assistant_masks_build_labels():
    from unsloth_zoo.mlx.utils import create_batches

    _, _, labels = create_batches(
        dataset=[
            {
                "input_ids": [1, 2, 3, 4],
                "assistant_masks": [0, 1, 0, 1],
            }
        ],
        tokenizer=types.SimpleNamespace(),
        batch_size=1,
        max_seq_length=8,
        assistant_only_loss=True,
        completion_only_loss=False,
    )[0]

    assert labels.tolist() == [[-100, 2, -100, 4]]


def test_text_completion_probe_keeps_one_shot_iterables_reusable():
    from unsloth_zoo.mlx.utils import _ensure_reiterable_text_dataset
    def rows():
        yield {"text": "1 2"}

    dataset = _ensure_reiterable_text_dataset(rows())
    assert list(dataset) == [{"text": "1 2"}]
    assert list(dataset) == [{"text": "1 2"}]


class _StreamingTextTokenizer:
    chat_template = None
    eos_token_id = None

    def __init__(self, offset=0, pad_token_id=0):
        self.offset = offset
        self.pad_token_id = pad_token_id

    def encode(self, text, add_special_tokens=True):
        return [int(part) + self.offset for part in str(text).split()]

    def __call__(self, text, **_kwargs):
        return types.SimpleNamespace(
            input_ids=self.encode(text, add_special_tokens=False),
        )

    def apply_chat_template(self, messages, tokenize=False, **_kwargs):
        ids = []
        for message in messages:
            ids.append(20 if message["role"] == "assistant" else 10)
            ids.extend(int(part) for part in message["content"].split())
        return ids if tokenize else " ".join(str(token) for token in ids)


class _MinimalTextModel:
    _config = {}
    def trainable_parameters(self): return {}


class _CountingTextRows:
    def __init__(self, rows, infinite=False):
        self.rows = tuple(rows)
        self.pulls = 0
        self._unsloth_mlx_infinite = infinite

    def __iter__(self):
        while True:
            for row in self.rows:
                self.pulls += 1
                yield row
            if not self._unsloth_mlx_infinite:
                return


class _DeclaredTextRows(_CountingTextRows):
    def __init__(self, rows):
        super().__init__(rows)
        self.epochs = []
    def __len__(self): return len(self.rows)
    def set_epoch(self, epoch): self.epochs.append(epoch)


def _streaming_text_tokenizer(pad_token_id=0):
    return _StreamingTextTokenizer(pad_token_id=pad_token_id)


def _streaming_text_trainer(**kwargs):
    MLXTrainer, trainer = _make_mlx_text_trainer(streaming=True, **kwargs)
    trainer.tokenizer = _streaming_text_tokenizer()
    trainer._distributed_initialized = True
    trainer._distributed_world = None
    trainer._distributed_world_size = 1
    return MLXTrainer, trainer


def _streaming_text_batches(dataset, tokenizer=None, **kwargs):
    from unsloth_zoo.mlx.utils import iterate_training_batches

    options = dict(batch_size=1, max_seq_length=8, dataset_order="sequential")
    return iterate_training_batches(
        dataset, tokenizer or _streaming_text_tokenizer(), **(options | kwargs),
    )


def _streaming_batch_signature(batch):
    tokens, lengths, labels = batch
    return tokens.tolist(), lengths.tolist(), None if labels is None else labels.tolist()


@pytest.mark.parametrize("use_hf", [False, True])
def test_text_streaming_yields_without_sizing_indexing_or_preconsumption(use_hf):
    from datasets import IterableDataset

    class GuardedRows:
        def __init__(self):
            self.pulls = 0

        def __len__(self):
            raise AssertionError("streaming source length must not be requested")

        def __getitem__(self, _index):
            raise AssertionError("streaming source must not be indexed")

        def __iter__(self):
            while True:
                if self.pulls >= 2:
                    raise AssertionError("source was consumed past the first batch")
                self.pulls += 1
                yield {"text": f"{self.pulls} {self.pulls + 10}"}

    guarded = GuardedRows()
    source = IterableDataset.from_generator(lambda: iter(guarded)) if use_hf else guarded
    batch = next(_streaming_text_batches(
        source,
        batch_size=2,
        completion_only_loss=False,
    ))

    assert guarded.pulls == 2
    assert [row[:2] for row in batch[0].tolist()] == [[1, 11], [2, 12]]


def test_streaming_trainer_exposes_lazy_prepared_iterable_view():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class Rows:
        def __init__(self):
            self.pulls = 0; self.features = {"value": "int64"}; self.column_names = ["value"]; self.split = "train"; self.restored = []
        def __len__(self): raise AssertionError("must stay unsized")
        def __getitem__(self, _index): raise AssertionError("must stay unindexed")
        def take(self, _count): return [{"value": "raw"}]
        def state_dict(self): return {"pulls": self.pulls}
        def load_state_dict(self, state): self.restored.append(state)
        def __iter__(self):
            self.pulls += 1
            yield {"value": 1}

    rows = Rows()
    trainer = MLXTrainer(
        _MinimalTextModel(), _streaming_text_tokenizer(), rows,
        formatting_func=lambda row: {"text": f"{row['value']} 2"},
        args=MLXTrainingConfig(
            streaming=True, max_steps=1, completion_only_loss=False,
            max_seq_length=8,
        ),
    )

    assert rows.pulls == 0
    assert not hasattr(trainer.train_dataset, "__len__")
    assert not hasattr(trainer.train_dataset, "__getitem__")
    assert not hasattr(trainer.train_dataset, "take")
    assert not hasattr(trainer.train_dataset, "state_dict")
    assert not hasattr(trainer.train_dataset, "load_state_dict")
    assert not hasattr(trainer.train_dataset, "features")
    assert not hasattr(trainer.train_dataset, "column_names")
    assert trainer.train_dataset.split == "train"
    assert rows.restored == []
    assert next(iter(trainer.train_dataset)) == {"value": 1, "input_ids": [1, 2]}
    assert rows.pulls == 1


def test_train_on_responses_only_masks_unsized_text_lazily():
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer, MLXTrainingConfig, train_on_responses_only,
    )

    source_rows = (
        {"text": "10 1"},
        {"text": "10 1 20 2 3"},
        {"messages": [
            {"role": "user", "content": "4"},
            {"role": "assistant", "content": "5"},
        ]},
    )
    rows = _CountingTextRows(source_rows)
    lazy_eval = _CountingTextRows(source_rows)
    trainer = MLXTrainer(
        _MinimalTextModel(), _StreamingTextTokenizer(offset=100), rows,
        eval_dataset={
            "sized": [{"text": "10 6 20 7"}],
            "lazy": lazy_eval,
        },
        args=MLXTrainingConfig(
            streaming=True, max_steps=1, per_device_train_batch_size=1,
            completion_only_loss=False, dataset_order="sequential",
            max_seq_length=8, chat_template="{{ messages }}",
        ),
    )
    train_on_responses_only(
        trainer, instruction_part="10", response_part="20", force_match=True,
        tokenizer=_StreamingTextTokenizer(),
    )

    assert rows.pulls == 0
    prepared_rows = iter(trainer.train_dataset)
    public_row = next(prepared_rows)
    assert public_row == source_rows[1] | {"input_ids": [10, 1, 20, 2, 3], "labels": [-100, -100, -100, 2, 3]}
    assert rows.pulls == 2  # the fully masked first row is legitimately filtered
    assert next(prepared_rows) == source_rows[2] | {"input_ids": [10, 4, 20, 5], "labels": [-100, -100, -100, 5]}
    assert trainer.eval_dataset["sized"][0]["labels"] == [-100, -100, -100, 7]
    assert lazy_eval.pulls == 0

def test_sized_response_training_defers_lazy_eval_with_override_tokenizer():
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer, MLXTrainingConfig, train_on_responses_only,
    )

    eval_rows = _CountingTextRows([{"text": "10 1 20 2"}])
    trainer = MLXTrainer(
        _MinimalTextModel(), _StreamingTextTokenizer(100),
        [{"text": "10 1 20 2"}],
        eval_dataset=eval_rows,
        args=MLXTrainingConfig(
            streaming=True, max_steps=1, completion_only_loss=False,
            per_device_train_batch_size=1,
        ),
    )
    override = _StreamingTextTokenizer()
    train_on_responses_only(
        trainer, instruction_part="10", response_part="20",
        tokenizer=override,
    )

    assert eval_rows.pulls == 0
    eval_batches = trainer._create_text_eval_batches(
        trainer.eval_dataset, 1, False, False,
    )
    batch = next(iter(eval_batches))
    assert batch[0].tolist() == [[10, 1, 20, 2]]
    assert batch[2].tolist() == [[-100, -100, -100, 2]]


def test_length_declaring_text_stream_supports_epoch_replay():
    MLXTrainer, trainer = _streaming_text_trainer(
        max_steps=0, num_train_epochs=2,
        completion_only_loss=False, dataset_order="sequential",
        per_device_train_batch_size=2, gradient_accumulation_steps=1,
    )
    trainer.train_dataset = _DeclaredTextRows([
        {"text": f"{value} {value + 10}"} for value in range(1, 6)
    ])
    batches, iterator = MLXTrainer._prepare_data(trainer, is_vlm=False)

    assert batches is None
    assert trainer._streaming_epoch_batch_count == 3
    signatures = [_streaming_batch_signature(next(iterator)) for _ in range(4)]
    assert signatures[3] == signatures[0]
    assert trainer.train_dataset.epochs == [0, 1]


def test_raw_text_streaming_matches_sized_sequential_order():
    rows = [{"value": value} for value in range(1, 6)]
    kwargs = {
        "batch_size": 2,
        "completion_only_loss": False,
        "formatting_func": lambda row: {
            "text": f"{row['value']} {row['value'] + 10}"
        },
    }
    expected = _streaming_text_batches(rows, **kwargs)
    actual = _streaming_text_batches(_CountingTextRows(rows), **kwargs)
    assert [_streaming_batch_signature(next(actual)) for _ in range(3)] == [
        _streaming_batch_signature(next(expected)) for _ in range(3)
    ]


def test_streaming_prompt_completion_and_assistant_labels():
    completion_batch = next(_streaming_text_batches(iter([
        {"prompt": "1 2", "completion": " 3"},
        {"prompt": "4", "completion": " 5 6"},
    ]), batch_size=2))
    assert completion_batch[0].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert completion_batch[2].tolist() == [[-100, -100, 3], [-100, 5, 6]]

    assistant_batch = next(_streaming_text_batches(
        iter([{
            "messages": [
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "2 3"},
            ],
        }]),
        tokenizer=_AssistantMaskTokenizer(),
        completion_only_loss=False,
        assistant_only_loss=True,
    ))
    assert assistant_batch[0].tolist() == [[10, 1, 20, 2, 3]]
    assert assistant_batch[2].tolist() == [[-100, -100, -100, 2, 3]]


def test_pretokenized_streaming_preserves_supported_label_fields():
    from unsloth_zoo.mlx.utils import _MLXIterableTokenizedDatasetView

    explicit = next(_streaming_text_batches(
        iter([{
            "input_ids": [1, 2],
            "labels": [-100, 2],
            "attention_mask": [0, 0],
        }]),
        tokenizer=types.SimpleNamespace(pad_token_id=9),
        completion_only_loss=False,
        formatting_func=lambda _row: pytest.fail(
            "pretokenized rows must bypass formatting"
        ),
    ))
    assert explicit[0].tolist() == [[1, 2]]
    assert explicit[2].tolist() == [[-100, 2]]

    masked_row = next(iter(_MLXIterableTokenizedDatasetView(iter([{
        "input_ids": [4, 5, 6], "completion_mask": [1, 1, 1],
        "assistant_masks": [0, 1, 1],
    }]), types.SimpleNamespace(), max_seq_length=2)))
    assert masked_row["completion_mask"] == [1, 1] and masked_row["assistant_masks"] == [0, 1]
    masked = next(_streaming_text_batches(
        iter([masked_row]),
        tokenizer=types.SimpleNamespace(pad_token_id=0),
    ))
    assert masked[2].tolist() == [[-100, 5]]


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([{"text": "1 2"}, {"input_ids": [3, 4]}], "cannot be mixed"),
        (
            [
                {"input_ids": [1, 2], "labels": [-100, 2]},
                {"input_ids": [3, 4]},
            ],
            "must not be mixed",
        ),
    ],
)
def test_text_streaming_rejects_incremental_schema_drift(rows, match):
    batches = _streaming_text_batches(
        iter(rows),
        completion_only_loss=False,
    )

    next(batches)
    with pytest.raises(ValueError, match=match):
        next(batches)

def test_hf_stream_replays_in_source_order_and_sets_epoch():
    from datasets import IterableDataset

    source = IterableDataset.from_generator(
        lambda: iter([{"text": "1"}, {"prompt": "2", "completion": " 3"}])
    )
    epochs = []
    original_set_epoch = source.set_epoch

    def record_epoch(epoch):
        epochs.append(epoch)
        original_set_epoch(epoch)

    source.set_epoch = record_epoch
    batches = _streaming_text_batches(source)
    first = _streaming_batch_signature(next(batches))

    assert first == ([[2, 3]], [[0, 2]], [[-100, 3]])
    assert _streaming_batch_signature(next(batches)) == first
    assert epochs == [0, 1]


def test_one_shot_stream_exhaustion_and_resume_are_actionable():
    batches = _streaming_text_batches(
        iter([{"text": "1 2"}, {"text": "3 4"}]),
        batch_size=2,
        completion_only_loss=False,
    )
    next(batches)
    with pytest.raises(RuntimeError, match="one-shot.*exhausted"):
        next(batches)

    MLXTrainer, trainer = _make_mlx_text_trainer(
        max_steps=2,
        streaming=True,
        dataset_order="sequential",
    )
    trainer.train_dataset = iter([{"text": "1 2"}, {"text": "3 4"}])
    trainer._resume_from_checkpoint = "checkpoint-1"
    _, resumed = MLXTrainer._prepare_data(trainer, is_vlm=False)
    with pytest.raises(RuntimeError, match="replayable iterable"):
        next(resumed)

    MLXTrainer, evaluator = _streaming_text_trainer(
        max_steps=1, completion_only_loss=False,
    )
    source = ({"text": "1 2"} for _ in range(1))
    eval_batches = evaluator._create_text_eval_batches(source, 1, False, False)
    with pytest.raises(RuntimeError, match="replayable iterable"):
        next(iter(eval_batches))
    assert next(source) == {"text": "1 2"}


def test_unsized_stream_rejects_randperm_before_consumption():
    pulls = 0

    def rows():
        nonlocal pulls
        pulls += 1
        yield {"text": "1 2"}

    with pytest.raises(ValueError, match="torch_randperm.*unsized"):
        next(_streaming_text_batches(rows(), dataset_order="torch_randperm"))
    assert pulls == 0

    class SizedRows(torch.utils.data.Dataset):
        rows = [{"text": "1 2"}, {"text": "3 4"}]
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]

    batch = next(_streaming_text_batches(SizedRows(), batch_size=2,
        completion_only_loss=False, dataset_order="torch_randperm"))
    assert sorted(row[0] for row in batch[0].tolist()) == [1, 3]


def test_text_pretokenized_create_batches_preserves_input_ids():
    from unsloth_zoo.mlx.utils import create_batches

    def formatting_func(_item):
        raise AssertionError("formatting_func should be ignored for input_ids rows")

    tokenizer = types.SimpleNamespace(
        pad_token_id=9,
        encode=lambda *_args, **_kwargs: pytest.fail("should not tokenize input_ids")
    )

    batch, lengths, labels = create_batches(
        dataset=[
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
        ],
        tokenizer=tokenizer,
        batch_size=2,
        max_seq_length=8,
        completion_only_loss=False,
        formatting_func=formatting_func,
    )[0]

    assert batch.tolist() == [[4, 5, 9], [1, 2, 3]]
    assert lengths.tolist() == [[0, 2], [0, 3]]
    assert labels is None


def test_text_pretokenized_rejects_mixed_raw_rows():
    from unsloth_zoo.mlx.utils import create_batches

    with pytest.raises(ValueError, match="cannot be mixed"):
        create_batches(
            dataset=[
                {"input_ids": [1, 2, 3]},
                {"text": "4 5 6"},
            ],
            tokenizer=types.SimpleNamespace(),
            batch_size=1,
            max_seq_length=8,
            completion_only_loss=False,
        )


def test_text_pretokenized_rejects_mixed_label_presence():
    from unsloth_zoo.mlx.utils import create_batches

    with pytest.raises(ValueError, match="must not be mixed"):
        create_batches(
            dataset=[
                {"input_ids": [1, 2, 3]},
                {"input_ids": [4, 5, 6], "labels": [-100, 5, 6]},
            ],
            tokenizer=types.SimpleNamespace(),
            batch_size=2,
            max_seq_length=8,
            completion_only_loss=False,
        )


def test_text_pretokenized_completion_mask_requires_completion_only_loss():
    from unsloth_zoo.mlx.utils import create_batches

    tokenizer = types.SimpleNamespace()
    kwargs = dict(tokenizer=tokenizer, batch_size=1, max_seq_length=8)
    row = {
        "input_ids": [1, 2, 3, 4],
        "labels": [11, 12, 13, 14],
        "completion_mask": [0, 1, 0, 1],
    }

    _, _, default_labels = create_batches(dataset=[row], **kwargs)[0]
    batch, _, masked_labels = create_batches(
        dataset=[row],
        completion_only_loss=True,
        **kwargs,
    )[0]

    assert batch.tolist() == [[1, 2, 3, 4]]
    assert default_labels.tolist() == [[11, 12, 13, 14]]
    assert masked_labels.tolist() == [[-100, 12, -100, 14]]


def test_text_pretokenized_ordered_and_streaming_batches_emit_labels():
    from unsloth_zoo.mlx.utils import create_ordered_batches, iterate_training_batches

    tokenizer = types.SimpleNamespace(pad_token_id=7)
    dataset = [
        {"input_ids": [1, 2], "labels": [-100, 2]},
        {"input_ids": [3, 4, 5], "labels": [-100, 4, 5]},
    ]

    batches = [
        create_ordered_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=2,
            max_seq_length=8,
            dataset_order="sequential",
        )[0],
        next(
            iterate_training_batches(
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=2,
                max_seq_length=8,
                seed=0,
            )
        ),
    ]

    for batch, _, labels in batches:
        assert batch.tolist() == [[1, 2, 7], [3, 4, 5]]
        assert labels.tolist() == [[-100, 2, -100], [-100, 4, 5]]


def test_text_prepare_data_passes_completion_only_loss_to_create_batches(monkeypatch):
    from unsloth_zoo.mlx import trainer as mlx_trainer

    received = {}

    def fake_create_batches(**kwargs):
        received.update(kwargs)
        return [("batch", "lengths", "labels")]

    monkeypatch.setattr(mlx_trainer, "create_batches", fake_create_batches)

    MLXTrainer, trainer = _make_mlx_text_trainer(
        max_steps=1,
        completion_only_loss=True,
        assistant_only_loss=True,
    )
    batches, _ = MLXTrainer._prepare_data(trainer, is_vlm=False)

    assert batches == [("batch", "lengths", "labels")]
    assert received["completion_only_loss"] is True
    assert received["assistant_only_loss"] is True


def test_text_prepare_data_ordered_batches_emit_completion_only_labels():
    MLXTrainer, trainer = _make_mlx_text_trainer(
        max_steps=1,
        completion_only_loss=True,
        dataset_order="sequential",
        per_device_train_batch_size=2,
    )
    trainer.tokenizer = types.SimpleNamespace(
        chat_template=None,
        eos_token_id=None,
        pad_token_id=7,
        encode=lambda text, add_special_tokens=True: [
            int(part) for part in str(text).split()
        ],
    )
    trainer.train_dataset = [
        {"prompt": "1", "completion": " 2"},
        {"prompt": "3", "completion": " 4 5"},
    ]
    batches, _ = MLXTrainer._prepare_data(trainer, is_vlm=False)

    batch, _, labels = batches[0]
    assert batch.tolist() == [[1, 2, 7], [3, 4, 5]]
    assert labels.tolist() == [[-100, 2, -100], [-100, 4, 5]]


def test_text_prepare_data_streaming_batches_emit_completion_only_labels():
    MLXTrainer, trainer = _make_mlx_text_trainer(
        max_steps=1,
        completion_only_loss=True,
        streaming=True,
        per_device_train_batch_size=2,
    )
    trainer.tokenizer = types.SimpleNamespace(
        chat_template=None,
        eos_token_id=None,
        encode=lambda text, add_special_tokens=True: [
            int(part) for part in str(text).split()
        ],
    )
    trainer.train_dataset = [
        {"prompt": "1", "completion": " 2"},
        {"prompt": "3", "completion": " 4 5"},
    ]

    batches, batch_iter = MLXTrainer._prepare_data(trainer, is_vlm=False)

    assert batches is None
    batch, _, labels = next(batch_iter)
    assert batch.tolist() == [[1, 2, 0], [3, 4, 5]]
    assert labels.tolist() == [[-100, 2, -100], [-100, 4, 5]]

    trainer.train_dataset = [{"text": "1 2"}, {"text": "3 4"}]
    with pytest.raises(ValueError, match="completion_only_loss=True"):
        next(MLXTrainer._prepare_data(trainer, is_vlm=False)[1])


def test_mlx_text_loss_masks_exclude_position_at_sequence_length():
    import inspect
    from unsloth_zoo.mlx import utils as mlx_utils

    source = inspect.getsource(mlx_utils.make_baseline_loss_fn)
    assert "steps < lengths[:, 1:]" in source


def test_train_on_responses_only_forwards_last_response_only(monkeypatch):
    import unsloth_zoo.dataset_utils as dataset_utils
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    class CallableTokenizer:
        chat_template = None
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    received = {}

    def fake_hf(trainer, *, instruction_part=None, response_part=None,
                force_match=True, tokenizer=None, return_function=False,
                num_proc=None, last_response_only=False):
        received["last_response_only"] = last_response_only
        return lambda batch: batch

    monkeypatch.setattr(dataset_utils, "train_on_responses_only", fake_hf)
    tokenizer = CallableTokenizer()
    train_on_responses_only(
        None,
        instruction_part="<user>",
        response_part="<assistant>",
        tokenizer=tokenizer,
        return_function=True,
        last_response_only=True,
    )

    assert received["last_response_only"] is True


def test_vlm_eval_batches_define_completion_only_loss_before_use():
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    source = inspect.getsource(MLXTrainer._train_inner)
    definition = source.index("text_completion_only_loss = _text_completion_only_loss_arg(args)")
    eval_use = source.index("text_completion_only_loss,")
    text_eval_block = inspect.getsource(MLXTrainer._create_text_eval_batches)
    assert definition < eval_use
    assert "completion_only_loss=completion_only_loss" in text_eval_block


def test_evaluate_dict_eval_datasets_records_split_metrics():
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer

    class Model:
        def __init__(self):
            self.modes = []

        def eval(self):
            self.modes.append("eval")

        def train(self):
            self.modes.append("train")

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.model = Model()
    trainer.stop_requested = False

    def loss_fn(_model, name, _lengths, _labels):
        if name == "small":
            return mx.array(1.0), mx.array(2)
        return mx.array(3.0), mx.array(6)

    loss, ppl = trainer._evaluate(
        {"small": [("small", None, None)], "large": [("large", None, None)]},
        loss_fn,
        is_vlm=False,
    )

    assert loss == pytest.approx(2.5)
    assert ppl == pytest.approx(__import__("math").exp(2.5))
    assert trainer._last_eval_metrics["eval_small_loss"] == pytest.approx(1.0)
    assert trainer._last_eval_metrics["eval_large_loss"] == pytest.approx(3.0)
    assert trainer._last_eval_metrics["eval_loss"] == pytest.approx(2.5)
    assert trainer.model.modes == ["eval", "train"]


def test_evaluate_batch_totals_uses_single_eval_status_collective():
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    source = inspect.getsource(MLXTrainer._evaluate_batch_totals)
    assert "_distributed_eval_status" in source
    assert "_distributed_should_stop" not in source
    assert "_raise_distributed_failure(" not in source


def test_check_all_masked_reduces_counts_across_ranks(monkeypatch):
    # In DDP each rank only sees its own shard. A rank whose shard happens to be
    # entirely masked must not raise alone (that would hang peers at the next
    # collective); the bad/good counts are all-summed first so the raise/warn
    # decision is global and identical on every rank.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import _check_all_masked

    def fake_all_sum(value, group=None, stream=None):
        # Simulate a peer rank that contributed trainable (good) rows.
        return value + mx.array([0, 5], dtype=mx.int32)

    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    all_bad = [("ids", None, mx.array([[-100, -100]]))]
    # Local shard is fully masked, but the global reduction sees good rows, so
    # no rank raises. (Would raise ZeroDivisionError without the reduction.)
    _check_all_masked(all_bad, comm_group=object(), world_size=2)


def test_check_all_masked_single_process_still_raises_when_all_masked():
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import _check_all_masked

    all_bad = [("ids", None, mx.array([[-100, -100]]))]
    with pytest.raises(ZeroDivisionError):
        _check_all_masked(all_bad)


def test_eval_callback_stop_request_synced_before_best_model_track():
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    cb_idx = src.index("for cb in self._eval_callbacks")
    track_idx = src.index("_track = not self.stop_requested")
    assert cb_idx < track_idx
    # A rank-wide stop sync must sit between the rank-0-only eval callbacks and
    # the divergent best-model / early-stopping branch, else a callback that
    # sets stop_requested on rank 0 alone makes _track diverge and hangs peers
    # at the rank-0-guarded best-model save collective.
    assert src.find("self._distributed_should_stop()", cb_idx, track_idx) != -1


def test_check_vlm_all_masked_reduces_counts_across_ranks(monkeypatch):
    # VLM mirror of the text-path mask check: a fully-masked local shard must
    # not raise alone in DDP; counts are all-summed before deciding.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import _check_vlm_all_masked

    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array([0, 5], dtype=mx.int32)

    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    all_bad = [{"labels": mx.array([[-100, -100]])}]
    _check_vlm_all_masked(all_bad, comm_group=object(), world_size=2)


def test_check_vlm_all_masked_single_process_still_raises():
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import _check_vlm_all_masked

    all_bad = [{"labels": mx.array([[-100, -100]])}]
    with pytest.raises(ZeroDivisionError):
        _check_vlm_all_masked(all_bad)


def test_reset_run_state_clears_last_eval_metrics():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    # A prior run's eval metrics must not leak into a reused trainer that then
    # runs without eval (eval_steps=0 or no eval dataset).
    trainer._last_eval_metrics = {"eval_loss": 1.23, "eval_perplexity": 4.5}
    trainer._reset_run_state()
    assert trainer._last_eval_metrics == {}


def test_distributed_diagnostics_per_rank_tokens_use_local_history():
    import inspect
    import re

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._distributed_training_diagnostics)
    # per_rank_tokens must be gathered from this rank's LOCAL token total, not
    # the all-reduced global trained_tokens (which would inflate by world_size).
    m = re.search(
        r"per_rank_tokens\s*=\s*self\._distributed_rank_vector\(\s*([A-Za-z_]+)",
        src,
    )
    assert m is not None and m.group(1) == "local_trained_tokens"
    assert "_local_token_count_history" in src


def test_reset_run_state_preserves_external_stop_request():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)

    # An externally-set cancel (e.g. a controller thread firing during train()
    # setup or batch prep) must survive the per-run reset.
    trainer.stop_requested = True
    trainer._reset_run_state()
    assert trainer.stop_requested is True
    assert trainer._early_stopped is False

    # A run-1 early stop must not block run 2 on a reused trainer.
    trainer.stop_requested = False
    trainer._early_stopped = True
    trainer._reset_run_state()
    assert trainer._early_stopped is False
    assert trainer.stop_requested is False


def test_reset_run_state_preserves_callbacks_and_batches():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)

    # Callbacks registered via add_step_callback / add_eval_callback before
    # train() (and the report_to callbacks set up inside train() before
    # _train_inner) must survive the per-run reset that _train_inner runs, else
    # user eval hooks never fire and W&B / TensorBoard logging is dropped.
    step_cb, eval_cb = object(), object()
    prebuilt = ["batch"]
    trainer._batches = prebuilt
    trainer._step_callbacks = [step_cb]
    trainer._eval_callbacks = [eval_cb]

    trainer._reset_run_state()

    assert trainer._batches is prebuilt
    assert trainer._step_callbacks == [step_cb]
    assert trainer._eval_callbacks == [eval_cb]


def test_resolved_best_metric_name_mirrors_hf_lookup():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)

    class Args:
        pass

    trainer.args = Args()
    for value, expected in [
        (None, "eval_loss"),
        ("loss", "eval_loss"),
        ("eval_loss", "eval_loss"),
        ("perplexity", "eval_perplexity"),
        ("eval_val_loss", "eval_val_loss"),
    ]:
        trainer.args.metric_for_best_model = value
        assert trainer._resolved_best_metric_name() == expected


def test_vlm_cce_prefers_collated_position_ids_for_cuda_parity():
    import inspect
    from unsloth_zoo.mlx import utils as mlx_utils

    forward_source = inspect.getsource(mlx_utils._vlm_cce_forward)
    unpack_source = inspect.getsource(mlx_utils._unpack_embed_result)
    prepare_source = inspect.getsource(mlx_utils._prepare_vlm_batch_for_compile)
    assert '"_unsloth_collated_position_ids"' in prepare_source
    assert 'not k.startswith("_unsloth_")' in forward_source
    assert 'use_collated_position_ids and "position_ids" in extra_kwargs' in forward_source
    assert 'lm is not None and "position_ids" not in backbone_kwargs' in unpack_source


def test_mlx_train_result_reports_base_quantization():
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    source = inspect.getsource(MLXTrainer._train_inner)
    assert '"base_quantization_config"' in source
    assert '"base_quantization_policy"' in source
    assert '"base_quantized_source"' in source


def test_mlx_loader_exposes_dense_nf4_diagnostic_mode():
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import (
        _MLX_QUANT_MODE_DEFAULTS,
        _nf4_dense_dequantize_weight,
    )

    assert _MLX_QUANT_MODE_DEFAULTS["nf4_dense"] == (64, 4)

    weight = mx.array([[-1.0, -0.6961928, 0.0, 0.72295684]], dtype=mx.float32)
    dequantized = _nf4_dense_dequantize_weight(weight, group_size=4)
    assert dequantized.shape == weight.shape
    assert dequantized.reshape((-1,)).tolist() == pytest.approx(
        weight.reshape((-1,)).tolist()
    )


def test_mlx_loader_keeps_norm_parameters_float32():
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import _keep_norm_parameters_float32

    class TinyModel:
        def __init__(self):
            self._parameters = {
                "vision_tower": {
                    "blocks": {
                        "0": {
                            "norm1": {
                                "weight": mx.array([1.0], dtype=mx.bfloat16),
                                "bias": mx.array([0.0], dtype=mx.bfloat16),
                            },
                            "attn": {
                                "qkv": {
                                    "weight": mx.array([[1.0]], dtype=mx.bfloat16),
                                },
                            },
                        },
                    },
                },
                "language_model": {
                    "model": {
                        "layers": {
                            "0": {
                                "input_layernorm": {
                                    "weight": mx.array([1.0], dtype=mx.bfloat16),
                                },
                            },
                        },
                    },
                },
            }

        def parameters(self):
            return self._parameters

        def update(self, parameters):
            self._parameters = parameters

    model = TinyModel()
    _keep_norm_parameters_float32(model)
    params = model.parameters()

    assert params["vision_tower"]["blocks"]["0"]["norm1"]["weight"].dtype == mx.float32
    assert params["vision_tower"]["blocks"]["0"]["norm1"]["bias"].dtype == mx.float32
    assert (
        params["language_model"]["model"]["layers"]["0"]["input_layernorm"]["weight"].dtype
        == mx.float32
    )
    assert (
        params["vision_tower"]["blocks"]["0"]["attn"]["qkv"]["weight"].dtype
        == mx.bfloat16
    )


def test_mlx_trainer_upcasts_norms_and_restores_prior_norm_output_cast_state(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import set_mlx_norm_output_cast_to_input_dtype

    class LoaderOnlyNorm(nn.Module):
        def __init__(self, dtype=mx.float32):
            super().__init__()
            self.weight = mx.ones((4,), dtype=dtype)

        def __call__(self, x):
            return x.astype(mx.float32) * self.weight

        def parameters(self):
            return {"weight": self.weight}

    class LoadedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = LoaderOnlyNorm()

    class TrainerModel(nn.Module):
        _config = {}

        def __init__(self):
            super().__init__()
            self.input_layernorm = LoaderOnlyNorm(mx.bfloat16)

    set_mlx_norm_output_cast_to_input_dtype(False)
    loaded_model = LoadedModel()
    x = mx.ones((2, 4), dtype=mx.bfloat16)
    try:
        set_mlx_norm_output_cast_to_input_dtype(True, loaded_model)
        assert loaded_model.input_layernorm(x).dtype == x.dtype
        patched_state = (
            LoaderOnlyNorm.__call__,
            getattr(LoaderOnlyNorm, "_unsloth_original_call"),
            getattr(LoaderOnlyNorm, "_unsloth_cast_output_to_input_dtype"),
        )

        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.model = TrainerModel()
        assert trainer.model.parameters()["input_layernorm.weight"].dtype == mx.bfloat16
        trainer.args = MLXTrainingConfig(
            cast_norm_output_to_input_dtype=False,
            gradient_checkpointing=False,
            compile=False,
            compile_auto_tune=False,
            compile_trace=False,
            disable_memory_limits=True,
        )
        trainer._is_vlm = False
        monkeypatch.setattr(MLXTrainer, "_configure_memory_limits", lambda self: {})
        monkeypatch.setattr(MLXTrainer, "_restore_memory_limits", lambda self: None)

        def train_inner(self):
            assert self.model.parameters()["input_layernorm.weight"].dtype == mx.float32
            assert loaded_model.input_layernorm(x).dtype == mx.float32
            return {"ok": True}

        monkeypatch.setattr(MLXTrainer, "_train_inner", train_inner)

        assert trainer.train() == {"ok": True}
        assert loaded_model.input_layernorm(x).dtype == x.dtype
        assert (
            LoaderOnlyNorm.__call__,
            getattr(LoaderOnlyNorm, "_unsloth_original_call"),
            getattr(LoaderOnlyNorm, "_unsloth_cast_output_to_input_dtype"),
        ) == patched_state

        class FailingNorm:
            weight = mx.ones((4,), dtype=mx.float32)

            def __call__(self, x):
                return x.astype(mx.float32)

            def parameters(self):
                return {"weight": self.weight}

        failing_norm = FailingNorm()

        class FailingModel:
            _config = {}

            def parameters(self):
                return {}

            def named_modules(self):
                return [("input_layernorm", failing_norm)]

        def raising_set_norm_output_cast(enabled, model=None):
            set_mlx_norm_output_cast_to_input_dtype(enabled, model)
            raise RuntimeError("setup failed")

        monkeypatch.setattr(
            "unsloth_zoo.mlx.trainer._set_norm_output_cast_to_input_dtype",
            raising_set_norm_output_cast,
        )

        failing_trainer = MLXTrainer.__new__(MLXTrainer)
        failing_trainer.model = FailingModel()
        failing_trainer.args = MLXTrainingConfig(cast_norm_output_to_input_dtype=True)
        with pytest.raises(RuntimeError, match="setup failed"):
            failing_trainer.train()
        assert not getattr(FailingNorm.__call__, "_unsloth_norm_output_cast_wrapper", False)
    finally:
        set_mlx_norm_output_cast_to_input_dtype(False)


def test_mlx_loader_fixes_gemma3_vision_post_layernorm_eps():
    from types import SimpleNamespace

    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_post_layernorm_eps

    post_layernorm = SimpleNamespace(eps=1e-5)
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(layer_norm_eps=1e-6),
        ),
        vision_tower=SimpleNamespace(
            vision_model=SimpleNamespace(post_layernorm=post_layernorm),
        ),
    )

    assert _fix_gemma3_vision_post_layernorm_eps(model) is True
    assert post_layernorm.eps == 1e-6
    assert model._unsloth_gemma3_vision_post_layernorm_eps == 1e-6


def test_mlx_loader_patches_gemma3_vision_attention_fp32_sdpa():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_attention_fp32_sdpa

    patched = _fix_gemma3_vision_attention_fp32_sdpa()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_attention_fp32_sdpa)
    assert "scaled_dot_product_attention" in source
    assert "astype(mx.float32)" in source
    assert "output.astype(orig_dtype)" in source


def test_mlx_loader_patches_gemma3_text_rmsnorm_fp32(monkeypatch):
    import inspect
    from types import SimpleNamespace

    import mlx.core as mx
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_text_rmsnorm_fp32
    from unsloth_zoo.mlx.utils import set_mlx_norm_output_cast_to_input_dtype

    patched = _fix_gemma3_text_rmsnorm_fp32()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_text_rmsnorm_fp32)
    assert "x.astype(mx.float32)" in source
    assert "mx.rsqrt(mx.mean(x_f * x_f" in source
    assert "return y.astype(orig_dtype)" in source
    assert "_unsloth_fp32_rmsnorm_patched" in source

    class FakeRMSNorm:
        def __init__(self):
            self.weight = mx.ones((4,), dtype=mx.float32)

        def __call__(self, x):
            return x.astype(mx.float32)

        def parameters(self):
            return {"weight": self.weight}

    class TinyModel:
        def __init__(self):
            self.norm = FakeRMSNorm()

        def named_modules(self):
            return [("language_model.input_layernorm", self.norm)]

    real_import_module = loader.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "mlx_vlm.models.gemma3.language":
            return SimpleNamespace(RMSNorm=FakeRMSNorm)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(loader.importlib, "import_module", fake_import_module)
    model = TinyModel()
    set_mlx_norm_output_cast_to_input_dtype(False)
    try:
        set_mlx_norm_output_cast_to_input_dtype(True, model)
        assert _fix_gemma3_text_rmsnorm_fp32(model) is True
        gemma_call = FakeRMSNorm.__call__

        set_mlx_norm_output_cast_to_input_dtype(False, model)
        assert FakeRMSNorm.__call__ is gemma_call

        set_mlx_norm_output_cast_to_input_dtype(True, model)
        assert getattr(FakeRMSNorm, "_unsloth_original_call") is gemma_call

        set_mlx_norm_output_cast_to_input_dtype(False, model)
        assert FakeRMSNorm.__call__ is gemma_call
    finally:
        set_mlx_norm_output_cast_to_input_dtype(False)


def test_vlm_hidden_stack_preserves_inputs_embed_dtype():
    import inspect

    import unsloth_zoo.mlx.utils as utils

    source = inspect.getsource(utils._run_hidden_stack)
    assert "h = inputs_embeds" in source
    assert "inputs_embeds.astype(norm_weight.dtype)" not in source


def test_mlx_loader_patches_gemma3_vision_mlp_fp32_activation():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_mlp_fp32_activation

    patched = _fix_gemma3_vision_mlp_fp32_activation()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_mlp_fp32_activation)
    assert "activation_fn(x.astype(mx.float32)).astype(orig_dtype)" in source
    assert "_unsloth_fp32_activation_patched" in source


def test_mlx_loader_patches_gemma3_vision_encoder_fp32_layernorm():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_encoder_fp32_layernorm

    patched = _fix_gemma3_vision_encoder_fp32_layernorm()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_encoder_fp32_layernorm)
    assert "x.astype(mx.float32)" in source
    assert "return y.astype(orig_dtype)" in source
    assert "_unsloth_fp32_layernorm_patched" in source


def test_mlx_loader_patches_gemma3_vision_post_layernorm_fp32():
    import inspect

    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_vision_post_layernorm_fp32

    patched = _fix_gemma3_vision_post_layernorm_fp32()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_vision_post_layernorm_fp32)
    assert "pooler_output = torch_like_layer_norm" in source
    assert "return y.astype(orig_dtype)" in source
    assert "_unsloth_fp32_post_layernorm_patched" in source


def test_mlx_loader_patches_gemma3_image_feature_scale():
    import inspect

    import mlx.core as mx
    import unsloth_zoo.mlx.loader as loader
    from unsloth_zoo.mlx.loader import _fix_gemma3_multimodal_image_feature_scale

    patched = _fix_gemma3_multimodal_image_feature_scale()
    assert patched in {True, False}

    source = inspect.getsource(loader._fix_gemma3_multimodal_image_feature_scale)
    assert "embed_dim = image_features.shape[-1]" in source
    assert "image_features / (embed_dim**0.5)" in source
    assert "del hidden_size" in source

    if patched:
        from mlx_vlm.models.gemma3.gemma3 import Model

        image_token_id = 99
        input_ids = mx.array([[1, image_token_id, image_token_id]])
        inputs_embeds = mx.ones((1, 3, 4))
        image_features = mx.ones((1, 2, 4))
        attention_mask = mx.ones((1, 3))

        embeds, _ = Model.prepare_inputs_for_multimodal(
            9,
            0,
            image_token_id,
            image_features,
            inputs_embeds,
            input_ids,
            attention_mask,
        )

        assert mx.allclose(embeds[0, 1:], mx.full((2, 4), 0.5))


def test_qwen3_vl_vision_rotary_uses_transformers_fp32_math():
    import inspect
    import unsloth_zoo.mlx.compile as mc

    source = inspect.getsource(mc._install_qwen3_family_compile_patches)

    assert "def _qwen3_vision_rotary_fp32" in source
    assert "tensor_f = tensor.astype(mx.float32)" in source
    assert "freqs_f = freqs.astype(mx.float32)" in source
    assert "return rotated.astype(orig_dtype)" in source
    assert "q = _qwen3_vision_rotary_fp32(q, rotary_pos_emb)" in source
    assert "k = _qwen3_vision_rotary_fp32(k, rotary_pos_emb)" in source


def test_qwen3_vl_vision_block_mlp_fp32_guard_for_fp16():
    """Pin the fp16 MLP overflow guard in patched_qwen3_vision_block_call.

    On M1/M2 Macs (no native bf16), MLX defaults to float16 for the vision
    tower. The vision block's MLP linear_fc1 (up-projection) produces output
    magnitudes that exceed fp16's 65504 ceiling for some inputs; downcasting
    to fp16 saturates to inf and cascades to NaN in the backward.

    Fix: when activation dtype is fp16, upcast the MLP input to fp32 so the
    entire MLP (fc1, GELU, fc2) runs in fp32. The output is cast back to
    source dtype at the residual add. bf16/fp32 keep the original path.
    """
    import inspect
    import unsloth_zoo.mlx.compile as mc

    source = inspect.getsource(mc._install_qwen3_family_compile_patches)

    # Guard is present
    assert "linear_fc1 (up-projection) overflows fp16" in source, (
        "Missing comment documenting the fp16 overflow rationale"
    )
    # Dtype-conditional branch keys on residual_dtype (the activation dtype)
    assert "if residual_dtype == mx.float16:" in source, (
        "MLP fp32 guard must be gated on residual_dtype == mx.float16"
    )
    # fp16 path: upcast input to fp32 before calling self.mlp
    assert "self.mlp(mlp_norm_out.astype(mx.float32))" in source, (
        "fp16 branch must upcast mlp input to fp32"
    )
    # non-fp16 path: original (cheaper) cast-only flow preserved
    assert "self.mlp(mlp_norm_out)" in source, (
        "bf16/fp32 path must keep the original self.mlp(...) call"
    )


def test_qwen3_vl_training_compile_verified():
    import unsloth_zoo.mlx.compile as mc

    assert "qwen3_vl" in mc._VERIFIED_TRAINING_ARCHES
    assert "qwen3_vl_moe" in mc._VERIFIED_TRAINING_ARCHES


def test_quantized_cce_uses_layer_mode_and_affine_bias_guard():
    import inspect
    import unsloth_zoo.mlx.utils as mlx_utils

    source = inspect.getsource(mlx_utils.make_vlm_cce_loss_fn)
    assert 'quant_mode = getattr(lm_layer, "mode", "affine")' in source
    assert "mode=quant_mode" in source
    assert 'if bi is None and quant_mode == "affine":' in source
    assert "bi = mx.zeros_like(sc)" in source


def test_gemma3_training_compile_verified():
    import unsloth_zoo.mlx.compile as mc

    assert "gemma3" in mc._VERIFIED_TRAINING_ARCHES


# ---------------------------------------------------------------------------
# 2. compile module-level discovery functions return sensible defaults
#    on a host with no real MLX architectures.
# ---------------------------------------------------------------------------

def test_compile_discovers_no_archs_under_shim():
    """No real mlx_vlm.models.* installed -> empty discovery, not crash."""
    import unsloth_zoo.mlx.compile as mc
    archs = mc.discover_architectures()
    assert isinstance(archs, tuple)


def test_compile_patch_primitives_exist():
    import unsloth_zoo.mlx.compile as mc
    primitives = mc.list_compile_patch_primitives()
    assert len(primitives) > 0


def test_compile_protocol_requirements_exist():
    import unsloth_zoo.mlx.compile as mc
    reqs = mc.list_protocol_requirements()
    assert len(reqs) > 0


def test_compile_summarize_qualifications_returns_dict():
    import unsloth_zoo.mlx.compile as mc
    s = mc.summarize_compile_qualifications()
    assert isinstance(s, dict)
    assert "architectures" in s


# ---------------------------------------------------------------------------
# 3. CCE backward via the pure-Python fallback.
# ---------------------------------------------------------------------------

def test_cce_backward_via_torch_autograd():
    """Build a tiny CCE forward and verify torch.autograd traverses it."""
    from unsloth_zoo.mlx.cce.runtime_cce import _forward_chunked_fused_finalize

    torch.manual_seed(0)
    n, hd, vocab = 4, 8, 32
    hidden = torch.randn(n, hd, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(vocab, hd, dtype=torch.float32) * 0.1
    weight.requires_grad_(True)
    targets = torch.tensor([3, 17, 5, 29], dtype=torch.int32)

    loss, _ = _forward_chunked_fused_finalize(
        hidden, weight, targets,
        scales=None, biases=None, group_size=None, bits=None, mode="affine",
        ignore_index=-100, logit_softcap=0.0, chunk_size=16,
        forward_update_kernel=None, forward_update_finalize_kernel=None,
    )
    loss.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


# ---------------------------------------------------------------------------
# 4. mx.dequantize cross-validation against the helper's output.
# ---------------------------------------------------------------------------

def test_mx_dequantize_with_nonzero_bias_and_scale():
    import mlx.core as mx

    bits, group_size = 4, 8
    elements_per_word = 32 // bits
    packed_value = 0
    for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        packed_value |= v << (i * bits)
    packed = torch.tensor([[packed_value]], dtype=torch.int32)
    scale = 0.5
    bias = -1.0
    scales = torch.tensor([[scale]])
    biases = torch.tensor([[bias]])

    out = mx.dequantize(packed, scales, biases, group_size=group_size,
                       bits=bits, mode="affine")
    expected = torch.tensor([[v * scale + bias for v in range(8)]],
                            dtype=scales.dtype)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 5. mx.fast.scaled_dot_product_attention works for a small attention.
# ---------------------------------------------------------------------------

def test_mx_fast_sdpa_works():
    import mlx.core as mx
    B, H, T, D = 1, 2, 4, 8
    q = torch.randn(B, H, T, D, dtype=torch.float32)
    k = torch.randn(B, H, T, D, dtype=torch.float32)
    v = torch.randn(B, H, T, D, dtype=torch.float32)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5))
    assert out.shape == (B, H, T, D)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. Tree utilities round-trip.
# ---------------------------------------------------------------------------

def test_tree_flatten_unflatten_roundtrip():
    from mlx.utils import tree_flatten, tree_unflatten

    tree = {"a": {"b": torch.tensor([1.0]), "c": torch.tensor([2.0])},
            "d": torch.tensor([3.0])}
    flat = tree_flatten(tree)
    keys = sorted(k for k, _ in flat)
    assert keys == ["a.b", "a.c", "d"]

    rebuilt = tree_unflatten(flat)
    assert set(rebuilt.keys()) == {"a", "d"}
    torch.testing.assert_close(rebuilt["d"], torch.tensor([3.0]))


# ---------------------------------------------------------------------------
# 7. Quantized layer __call__ works (forward through nn.QuantizedLinear).
# ---------------------------------------------------------------------------

def test_quantized_linear_forward():
    import mlx.nn as nn
    bits, group_size = 4, 8

    # 4-bit, in_features=8, out_features=2.
    elements_per_word = 32 // bits
    packed_value = 0
    for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7]):
        packed_value |= v << (i * bits)
    packed_row = torch.tensor([[packed_value]], dtype=torch.int32)
    packed = torch.cat([packed_row, packed_row], dim=0)  # (2, 1)
    scales = torch.ones((2, 1), dtype=torch.float32)
    biases = torch.zeros((2, 1), dtype=torch.float32)

    layer = nn.QuantizedLinear(8, 2, bias=False, group_size=group_size,
                                bits=bits, mode="affine")
    layer.weight = packed
    layer.scales = scales
    layer.biases = biases

    x = torch.ones((1, 8), dtype=torch.float32)
    # x @ W.T  with W = [[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7]] = [28, 28]
    out = layer(x)
    torch.testing.assert_close(out, torch.tensor([[28.0, 28.0]]))


def _first_tokens(stream, count):
    return [[row[0] for row in batch.tolist()]
            for batch, _l, _lab in (next(stream) for _ in range(count))]


def _window_rows(lengths):
    # Row index rides in the first token so batch order is observable.
    return [" ".join([str(100 + idx)] + ["7"] * (n - 1)) for idx, n in enumerate(lengths)]


def _window_stream(rows, *, window, order="default", repeat=False, seed=3407,
                   source=None, **kwargs):
    from unsloth_zoo.mlx.utils import iterate_training_batches
    return iterate_training_batches(
        dataset=_CountingTextRows(rows) if source is None else source,
        tokenizer=_streaming_text_tokenizer(), batch_size=2, max_seq_length=64,
        seed=seed, dataset_order=order, repeat=repeat,
        length_window_batches=window, **kwargs,
    )


def _window_batches(rows, *, max_batches=None, **kwargs):
    out = []
    for batch, _l, _lab in _window_stream(rows, **kwargs):
        out.append(([row[0] for row in batch.tolist()], batch.shape[1]))
        if max_batches is not None and len(out) >= max_batches:
            break
    return out


_WINDOW_LENGTHS = (40, 3, 44, 4, 2, 46, 3, 41)


def test_streaming_window_identity_grouping_padding_gate_and_ddp_slice():
    rows, arrival = _window_rows(_WINDOW_LENGTHS), [[100, 101], [102, 103], [104, 105], [106, 107]]
    assert [ids for ids, _ in _window_batches(rows, window=1)] == arrival
    assert [ids for ids, _ in _window_batches(rows, window=8, order="sequential")] == arrival

    class FakeWorld:
        def rank(self): return 1
        def size(self): return 2
    from unsloth_zoo.mlx.utils import _iterate_lazy_text_training_batches
    ddp = _iterate_lazy_text_training_batches(
        _CountingTextRows(rows), _streaming_text_tokenizer(), 1, 64,
        comm_group=FakeWorld(), repeat=False, length_window_batches=1)
    assert _first_tokens(ddp, 4) == [[101], [103], [105], [107]]

    grouped, baseline = _window_batches(rows, window=4), _window_batches(rows, window=1)
    assert sorted(t for ids, _ in grouped for t in ids) == [100 + i for i in range(8)]
    assert sum(w for _, w in grouped) < sum(w for _, w in baseline)
    assert grouped == _window_batches(rows, window=4) != baseline
    assert [ids for ids, _ in _window_batches(rows, window=4, seed=1)] != [ids for ids, _ in grouped]

    # Single-process windowed slicing must see an EMPTY pad_source (cycle
    # padding is multi-rank-only) and leave the partial tail short.
    from unsloth_zoo.mlx import utils as mlx_utils
    odd, seen_pads = _window_rows(_WINDOW_LENGTHS[:7]), []
    original = mlx_utils._rank_slice_distributed_batch
    def spy(items, n, comm_group=None, pad_source=None, pad_mode="cycle"):
        seen_pads.append([] if not pad_source else list(pad_source))
        return original(items, n, comm_group=comm_group, pad_source=pad_source, pad_mode=pad_mode)
    mlx_utils._rank_slice_distributed_batch = spy
    try:
        tail = _window_batches(odd, window=3)[-1]
    finally:
        mlx_utils._rank_slice_distributed_batch = original
    assert len(tail[0]) == 1 and seen_pads and all(p == [] for p in seen_pads)


def test_streaming_window_epochs_cardinality_oneshot_and_knob():
    odd_rows = _window_rows(_WINDOW_LENGTHS[:7])  # 7 rows -> 4 batches/pass
    source = _DeclaredTextRows(odd_rows)
    crossing = _window_batches(odd_rows, window=3, repeat=True, source=source, max_batches=6)
    assert source.epochs[:2] == [0, 1]
    assert sorted(t for ids, _ in crossing[:4] for t in ids) == [100 + i for i in range(7)]
    assert _window_batches(odd_rows, window=3, repeat=True, max_batches=6) == crossing
    assert [ids for ids, _ in crossing[4:6]] != [ids for ids, _ in crossing[:2]]  # epoch reaches seed

    skipped = _window_stream(odd_rows, window=3, repeat=True, require_replayable=True)
    for _ in range(5):
        next(skipped)
    assert _first_tokens(skipped, 1) == [crossing[5][0]]  # resume fast-forward

    from unsloth_zoo.mlx.utils import _iterate_lazy_text_training_batches
    def exact_stream(expected):
        return _iterate_lazy_text_training_batches(
            _DeclaredTextRows(odd_rows), _streaming_text_tokenizer(), 2, 64,
            repeat=True, length_window_batches=3, window_seed=3407,
            expected_rows_per_pass=expected)
    two_passes = _first_tokens(exact_stream(7), 8)
    assert sorted(t for ids in two_passes[:4] for t in ids) == [100 + i for i in range(7)]
    emitted = []
    with pytest.raises(ValueError, match="declared length"):
        for batch in exact_stream(6):
            emitted.append(batch)
    assert len(emitted) < 4  # buffered final window withheld on overrun

    one_shot = _window_stream(odd_rows, window=3, repeat=True,
                              source=iter(list(_CountingTextRows(odd_rows))))
    for _ in range(4):
        next(one_shot)
    with pytest.raises(RuntimeError, match="one-shot"):
        next(one_shot)

    assert next(iter(_window_stream(odd_rows, window=4, seed=None)))[0].shape[0] == 2

    for bad in (True, False, 0, -2, 2.0, "4"):
        probe = _CountingTextRows(_window_rows((3, 2)))
        with pytest.raises(ValueError, match="streaming_text_length_window"):
            next(iter(_window_stream([], window=bad, source=probe)))
        assert probe.pulls == 0


def test_streaming_window_trainer_routing_and_config_copy():
    rows, arrival = _window_rows(_WINDOW_LENGTHS), [[100, 101], [102, 103], [104, 105], [106, 107]]

    def prepared(**config_kwargs):
        _T, trainer = _streaming_text_trainer(
            per_device_train_batch_size=2, max_seq_length=64, **config_kwargs)
        trainer.train_dataset = _CountingTextRows(rows)
        _batches, stream = trainer._prepare_data(is_vlm=False)
        return _first_tokens(stream, 4)

    windowed = prepared(streaming_text_length_window_batches=4)
    assert sorted(t for ids in windowed for t in ids) == [100 + i for i in range(8)]
    assert windowed != arrival
    assert prepared(streaming_text_length_window_batches=4,
                    preserve_dataset_order=True) == arrival

    _T, bad_trainer = _streaming_text_trainer(
        per_device_train_batch_size=2, max_seq_length=64,
        streaming_text_length_window_batches=0)
    probe = _DeclaredTextRows(rows)
    bad_trainer.train_dataset = probe
    with pytest.raises(ValueError, match="streaming_text_length_window"):
        bad_trainer._prepare_data(is_vlm=False)
    assert probe.pulls == 0 and probe.epochs == []

    from unsloth_zoo.mlx.trainer import MLXTrainingConfig
    base = MLXTrainingConfig(warmup_ratio=0.25)
    for omitted in (("streaming_text_length_window_batches",),
                    ("streaming_text_length_window_batches", "max_eval_batches")):
        copied = {f.name: getattr(base, f.name)
                  for f in dataclasses.fields(MLXTrainingConfig)
                  if f.init and f.name not in omitted}
        clone = MLXTrainingConfig(**copied)
        assert clone.streaming_text_length_window_batches == 8
        assert clone._unsloth_mlx_warmup_steps_explicit is False


def test_host_staging_seam_parity_and_host_valued_flag():
    import numpy as np
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import (
        _HostStagedTextBatch, _finalize_text_batch, _stage_tokenized_text_batch,
    )
    rows = _window_rows((5, 3, 4, 2))

    from unsloth_zoo.mlx.utils import _iterate_lazy_text_training_batches
    def _direct(**kwargs):
        return _iterate_lazy_text_training_batches(
            _CountingTextRows(rows), _streaming_text_tokenizer(), 2, 64,
            repeat=False, length_window_batches=2, window_seed=3407, **kwargs)
    staged_stream = _direct(yield_host_staged=True)
    finalized_stream = _direct()
    for _ in range(2):
        staged = next(staged_stream)
        assert isinstance(staged, _HostStagedTextBatch)
        assert isinstance(staged.ids, np.ndarray) and staged.host_valued
        batch, lengths, labels = next(finalized_stream)
        f_batch, f_lengths, f_labels = _finalize_text_batch(staged)
        assert f_batch.tolist() == batch.tolist()
        assert f_lengths.tolist() == lengths.tolist()
        assert f_labels is None and labels is None

    # MLX-valued rows via the REAL pipeline: origin recorded pre-.tolist().
    class MxRows:
        def __iter__(self):
            return iter([
                {"input_ids": mx.array([7, 8, 9])},
                {"input_ids": [1, 2, 3]},
            ])
    mx_staged = next(_iterate_lazy_text_training_batches(
        MxRows(), _streaming_text_tokenizer(), 2, 8,
        repeat=False, yield_host_staged=True))
    assert mx_staged.host_valued is False       # flagged for the prefetch producer
    ids, _lengths, _labels = _finalize_text_batch(mx_staged)
    assert ids.tolist()[0][:3] == [7, 8, 9]     # sync mode still accepts it

    assert not _stage_tokenized_text_batch(
        [(mx.array([7, 8]), None), ([1, 2], None)], 8).host_valued

    # Raw-text pipeline end-to-end: an mx-returning tokenizer must surface as a
    # host_valued=False STAGED batch (the raw stager forwards the stream flag).
    class MxRawTok(_StreamingTextTokenizer):
        def encode(self, text, add_special_tokens=True):
            return mx.array(super().encode(text, add_special_tokens))
    raw_staged = next(_iterate_lazy_text_training_batches(
        _CountingTextRows(["5 6 7", "8 9 10"]), MxRawTok(), 2, 8,
        repeat=False, yield_host_staged=True))
    assert raw_staged.host_valued is False

    # User chat-template variables named 'state' must keep working.
    from unsloth_zoo.mlx.utils import _tokenize_mlx_prompt_completion_row
    seen_kwargs = {}
    class TemplateTok(_StreamingTextTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen_kwargs.update(kwargs)
            return [10, 11, 12]
    row = {"prompt": [{"role": "user", "content": "1 2"}],
           "completion": [{"role": "assistant", "content": "3 4"}],
           "chat_template_kwargs": {"state": "CA", "_unsloth_state": "USER"}}
    assert _tokenize_mlx_prompt_completion_row(TemplateTok(), row) is not None
    assert seen_kwargs.get("state") == "CA" and seen_kwargs.get("_unsloth_state") == "USER"


def test_streaming_prefetch_identity_laziness_and_knob():
    rows = _window_rows(_WINDOW_LENGTHS)
    sync = _window_batches(rows, window=4)
    prefetched = _window_batches(rows, window=4, prefetch_batches=2)
    assert prefetched == sync  # bit-for-bit consumer-visible sequence

    probe = _CountingTextRows(rows)
    from unsloth_zoo.mlx.utils import iterate_training_batches
    stream = iterate_training_batches(
        dataset=probe, tokenizer=_streaming_text_tokenizer(), batch_size=2,
        max_seq_length=64, seed=3407, dataset_order="default", repeat=False,
        length_window_batches=4, prefetch_batches=2)
    assert probe.pulls == 0  # construction-lazy at P>0
    first = next(iter(stream))
    assert first[0].shape[0] == 2 and probe.pulls >= 2

    for bad in (True, -1, 1.5):
        with pytest.raises(ValueError, match="streaming_prefetch_batches"):
            next(iter(_window_stream(rows, window=1, prefetch_batches=bad)))

    from unsloth_zoo.mlx.trainer import MLXTrainingConfig, _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    assert "streaming_prefetch_batches" in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    assert MLXTrainingConfig().streaming_prefetch_batches == 0  # default OFF

    # Producer-side resume skip parity: first batch equals the sync sequence
    # at the skip offset (single skip authority — no double fast-forward).
    sync_all = _window_batches(rows, window=4)
    skipped = _window_stream(rows, window=4, prefetch_batches=2,
                             prefetch_skip_batches=2)
    first_after_skip = [row[0] for row in next(iter(skipped))[0].tolist()]
    assert first_after_skip == sync_all[2][0]

    # Trainer records prefetch eligibility synchronously (guards its own
    # legacy fast-forward) and the orphan gate survives exceptional teardown.
    _T, trainer = _streaming_text_trainer(
        per_device_train_batch_size=2, max_seq_length=64,
        streaming_prefetch_batches=2)
    trainer.train_dataset = _CountingTextRows(rows)
    _b, _stream = trainer._prepare_data(is_vlm=False)
    assert trainer._mlx_prefetch_control.get("eligible") is True

    class FakeOrphan:
        def close(self): pass
        def orphan_alive(self): return True
    trainer._mlx_prefetch_control = {"prefetcher": FakeOrphan()}
    trainer._active_batch_iter = None
    trainer._close_active_batch_iterator()  # exceptional-teardown path
    assert trainer._mlx_prefetch_orphan.orphan_alive()


def test_prefetcher_lifecycle_quiescence_orphan_and_positioned_error():
    import threading
    import time
    from unsloth_zoo.mlx import utils as mlx_utils
    from unsloth_zoo.mlx.utils import _LazyTextPrefetcher

    tok = _streaming_text_tokenizer()

    def staged(rows, **kwargs):
        return lambda: mlx_utils._iterate_lazy_text_training_batches(
            _CountingTextRows(rows), tok, 1, 64, repeat=False,
            yield_host_staged=True, **kwargs)

    # Quiesce/resume mid-stream, then clean terminal close (no orphan).
    pf = _LazyTextPrefetcher(staged(["1 2", "3 4", "5 6"]), depth=1)
    next(pf)
    pf.quiesce()
    pf.resume()
    next(pf)
    assert pf.close() and not pf.orphan_alive()

    # A blocked source orphans within the bounded join and gates trainer reuse.
    gate, entered = threading.Event(), threading.Event()
    class Blocked:
        def __iter__(self):
            def _gen():
                entered.set()
                gate.wait()
                yield {"text": "1 2"}
            return _gen()
    stuck = _LazyTextPrefetcher(
        lambda: mlx_utils._iterate_lazy_text_training_batches(
            Blocked(), tok, 1, 64, repeat=False, yield_host_staged=True),
        depth=1)
    stuck._JOIN_TIMEOUT = 0.2
    try:
        stuck._ensure_started()
        assert entered.wait(timeout=2.0)  # deterministically inside the pull
        assert not stuck.close() and stuck.orphan_alive()
        from unsloth_zoo.mlx.trainer import MLXTrainer
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer._mlx_prefetch_orphan = stuck
        trainer._mlx_prefetch_control = {}
        with pytest.raises(RuntimeError, match="refusing to serialize"):
            trainer._quiesce_prefetcher_for_save()
    finally:
        gate.set()
        assert stuck._done.wait(timeout=2.0)
        stuck._thread.join(timeout=2.0)
        assert not stuck._thread.is_alive()  # no daemon leak beyond the test

    # Producer exceptions arrive positioned after prior good batches.
    class Exploding(_StreamingTextTokenizer):
        def __init__(self): super().__init__(); self.count = 0
        def encode(self, text, add_special_tokens=True):
            self.count += 1
            if self.count > 2:
                raise RuntimeError("late tokenizer failure")
            return super().encode(text, add_special_tokens)
    boom = _LazyTextPrefetcher(
        lambda: mlx_utils._iterate_lazy_text_training_batches(
            _CountingTextRows(["1 2", "3 4", "5 6"]), Exploding(), 1, 64,
            repeat=False, yield_host_staged=True),
        depth=2)
    assert next(boom) is not None and next(boom) is not None
    with pytest.raises(RuntimeError, match="late tokenizer failure"):
        next(boom)
    boom.close()


def test_lazy_text_producer_rejects_mlx_valued_rows_before_parsing():
    """Raw rows / formatter results carrying MLX values reject actionably
    before any truthiness probe or parsing can evaluate them off-thread."""
    import mlx.core as mx

    from unsloth_zoo.mlx.utils import _iterate_lazy_text_training_batches

    def _reject_probe(row=None, formatting_func=None):
        def _src():
            yield row if row is not None else {"text": "hello"}
        with pytest.raises(ValueError, match="streaming_prefetch_batches=0"):
            next(_iterate_lazy_text_training_batches(
                _src(), None, 1, 32, formatting_func=formatting_func,
                yield_host_staged=True, reject_mlx_valued=True))

    _reject_probe({"text": mx.array([1, 2])})
    _reject_probe({"messages": mx.array([1, 2])})
    _reject_probe(formatting_func=lambda item: {"text": mx.array([3, 4])})
