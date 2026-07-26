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
import os
import tempfile
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


def test_finite_text_batch_plan_materializes_cpu_rows_on_demand():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, _FiniteTextRow

    plan = FiniteTextBatchPlan(
        (
            _FiniteTextRow((1, 2, 3), offset=1),
            _FiniteTextRow((4, 5), offset=0),
        ),
        ((0, 1),),
        max_seq_length=8,
        pad_id=9,
    )

    assert all(not isinstance(value, mx.array) for value in plan.rows)
    batch, lengths, labels = plan[0]
    assert batch.tolist() == [[1, 2, 3], [4, 5, 9]]
    assert lengths.tolist() == [[1, 3], [0, 2]]
    assert labels is None


def test_finite_text_batch_plan_preserves_label_padding():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, _FiniteTextRow

    plan = FiniteTextBatchPlan(
        (
            _FiniteTextRow((1, 2, 3), labels=(-100, 2, 3)),
            _FiniteTextRow((4, 5), labels=(-100, 5)),
        ),
        ((0, 1),),
        max_seq_length=8,
        pad_id=7,
    )

    batch, lengths, labels = plan[0]
    assert batch.tolist() == [[1, 2, 3], [4, 5, 7]]
    assert lengths.tolist() == [[0, 3], [0, 2]]
    assert labels.tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert labels.dtype == mx.int64


def test_finite_text_training_plan_keeps_long_schedule_cpu_only():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import (
        FiniteTextBatchPlan,
        _create_text_batch_plan,
        create_batches,
    )

    dataset = [
        {"input_ids": [1, 2]},
        {"input_ids": [3, 4, 5]},
        {"input_ids": [6, 7, 8, 9]},
        {"input_ids": [10, 11, 12, 13, 14]},
    ]
    tokenizer = types.SimpleNamespace(pad_token_id=7)
    plan = _create_text_batch_plan(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=2,
        max_seq_length=16,
        num_batches=120,
        seed=11,
        completion_only_loss=False,
    )

    assert isinstance(plan, FiniteTextBatchPlan)
    assert len(plan) == 120
    assert all(not isinstance(row.input_ids, mx.array) for row in plan.rows)
    assert all(
        isinstance(row_index, int)
        for batch_indices in plan.schedule
        for row_index in batch_indices
    )

    expected = create_batches(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=2,
        max_seq_length=16,
        num_batches=1,
        seed=11,
        completion_only_loss=False,
    )[0]
    actual = plan[0]
    assert [value.tolist() if value is not None else None for value in actual] == [
        value.tolist() if value is not None else None for value in expected
    ]


def test_default_text_plan_uses_mlx_lm_padding():
    # Regression: the default (non-pretokenized) finite text plan must pad to
    # mlx-lm's 1 + 32*ceil(len/32) width, not the raw row length, so it keeps
    # the causal-shift contract and the bounded compile signatures.
    from unsloth_zoo.mlx.utils import _create_default_text_plan

    class _DS:
        def __init__(self, lengths):
            self._rows = [(list(range(1, n + 1)), 1) for n in lengths]

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, index):
            return self._rows[index]

        def itemlen(self, index):
            return len(self._rows[index][0])

    plan = _create_default_text_plan(
        _DS([14]), batch_size=1, max_seq_length=2048, num_batches=1, seed=0,
    )
    assert plan.batch_width(0) == 33  # 1 + 32*ceil(14/32)


def _make_shape_guard_text_plan(widths, *, schedules=None, labeled=True):
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, _FiniteTextRow

    rows = tuple(
        _FiniteTextRow(
            tuple(range(1, width + 1)),
            offset=1,
            labels=(tuple(range(1, width + 1)) if labeled else None),
        )
        for width in widths
    )
    return FiniteTextBatchPlan(
        rows,
        schedules or tuple((index,) for index in range(len(rows))),
        max_seq_length=64,
        pad_id=99,
    )


def test_single_process_text_shape_guard_buckets_and_validates_before_materializing():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )

    batches = _make_shape_guard_text_plan((10, 11, 30))
    args = MLXTrainingConfig(
        max_steps=6,
        gradient_accumulation_steps=1,
        compile_max_variants=2,
    )
    shape_plan, report, compile_allowed, _ = _plan_single_process_text_shapes(
        batches,
        None,
        args=args,
        total_steps=6,
        is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=args),
    )

    assert compile_allowed is True
    assert report.action == "bucket"
    assert report.raw_signatures == 3
    assert report.planned_signatures == 2
    assert len(shape_plan.planned_catalog) == 2
    batch, lengths, labels = batches.materialize(0, phase="single")
    assert batch.shape == (1, 11)
    assert batch[0, -1].item() == 99
    assert labels[0, -1].item() == -100
    assert lengths.tolist() == [[1, 10]]
    assert batches[0][0].shape == (1, 10)
    with pytest.raises(RuntimeError, match="was not admitted"):
        batches.materialize(0, phase="unknown")


def test_automatic_text_shape_guard_installs_deterministic_budgeted_plan():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )

    args = MLXTrainingConfig(max_steps=40, gradient_accumulation_steps=1)
    batches = _make_shape_guard_text_plan(tuple(range(10, 50)))
    shape_plan, report, allowed, frontier = _plan_single_process_text_shapes(
        batches,
        None,
        args=args,
        total_steps=40,
        is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=args),
    )

    assert allowed is True and frontier is not None
    assert report.cap_selection == "padding_budget"
    assert report.configured_cap == 128
    assert report.effective_cap == report.cap == 15
    assert report.planned_signatures == 15
    assert report.padding_work_fraction <= 0.05
    assert report.max_width_stretch <= 1.5
    assert report.budget_satisfied is True
    batch, lengths, labels = batches.materialize(0, phase="single")
    assert batch.shape == labels.shape == (1, 13)
    assert batch[0, :10].tolist() == list(range(1, 11))
    assert labels[0, 10:].tolist() == [-100, -100, -100]
    assert lengths.tolist() == [[1, 10]]
    assert shape_plan.report == report


def test_ddp_automatic_shape_guard_reuses_frontier_at_shared_maximum_cap():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.shape_guard import (
        DDP_LOCAL_GRAD_SCOPE,
        TextShapeEvent,
        build_text_shape_frontier,
        select_text_shape_padding_budget,
    )
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    events = [
        TextShapeEvent(("text",), width, "none")
        for width in range(10, 50)
    ]
    frontier = build_text_shape_frontier(
        events, compile_scope=DDP_LOCAL_GRAD_SCOPE,
    )
    local_plan = select_text_shape_padding_budget(frontier)
    shared_cap = local_plan.report.effective_cap + 4
    trainer = object.__new__(MLXTrainer)
    trainer._distributed_initialized = True
    trainer._distributed_world_size = 2
    trainer._distributed_any_flag = lambda _failed: False
    trainer._distributed_max_int = lambda cap: shared_cap

    plan, report, allowed = trainer._coordinate_text_shape_guard(
        local_plan,
        frontier,
        local_plan.report,
        True,
        build_compile_policy(args=MLXTrainingConfig()),
        automatic=True,
    )

    assert allowed is True
    assert report.effective_cap == report.cap == shared_cap <= 128
    assert report.planned_signatures <= shared_cap
    assert report.budget_satisfied is True
    assert plan.report == report

    trainer._distributed_max_int = lambda _cap: 129
    failure_consensus = iter((False, True))
    trainer._distributed_any_flag = lambda _failed: next(failure_consensus)
    plan, failed_report, allowed = trainer._coordinate_text_shape_guard(
        local_plan,
        frontier,
        local_plan.report,
        True,
        build_compile_policy(args=MLXTrainingConfig()),
        automatic=True,
    )

    assert plan is None and allowed is False
    assert failed_report.action == "eager"
    assert failed_report.cap_selection == "not_applicable"
    assert failed_report.effective_cap == failed_report.cap == 128


def test_ddp_not_applicable_auto_shape_guard_skips_cap_synchronization():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _shape_guard_report,
    )

    trainer = object.__new__(MLXTrainer)
    trainer._distributed_initialized = True
    trainer._distributed_world_size = 2
    trainer._distributed_any_flag = lambda _failed: False
    trainer._distributed_max_int = lambda _cap: (_ for _ in ()).throw(
        AssertionError("not-applicable paths must not synchronize a cap")
    )
    report = _shape_guard_report(
        "not_applicable", "streaming", 128, lazy_batches=False,
    )

    plan, coordinated, allowed = trainer._coordinate_text_shape_guard(
        None,
        None,
        report,
        True,
        build_compile_policy(args=MLXTrainingConfig(streaming=True)),
        automatic=True,
    )

    assert plan is None and allowed is True
    assert coordinated == report


def test_ddp_synchronizes_bounded_padding_budget_cap():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )

    args = MLXTrainingConfig(max_steps=40)
    policy = build_compile_policy(args=args)
    local_plan, report, allowed, frontier = _plan_single_process_text_shapes(
        _make_shape_guard_text_plan(tuple(range(10, 50))),
        None,
        args=args,
        total_steps=40,
        is_vlm=False,
        distributed_world_size=2,
        compile_policy=policy,
        install_plan=False,
    )
    shared_cap = min(128, report.effective_cap + 3)
    trainer = object.__new__(MLXTrainer)
    trainer._distributed_initialized = True
    trainer._distributed_world_size = 2
    trainer._distributed_any_flag = lambda _failed: False
    trainer._distributed_max_int = lambda _cap: shared_cap

    final_plan, final_report, final_allowed = trainer._coordinate_text_shape_guard(
        local_plan, frontier, report, allowed, policy, automatic=True,
    )

    assert final_allowed is True
    assert final_report.cap_selection == "padding_budget"
    assert final_report.effective_cap == shared_cap
    assert final_report.planned_signatures <= shared_cap
    assert final_plan.report == final_report


def test_text_shape_guard_exact_and_compile_disabled_paths_add_no_padding():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )

    exact_batches = _make_shape_guard_text_plan((10, 11, 30), labeled=False)
    exact_args = MLXTrainingConfig(
        max_steps=3,
        gradient_accumulation_steps=1,
        compile_max_variants=3,
    )
    _, exact_report, exact_allowed, _ = _plan_single_process_text_shapes(
        exact_batches,
        None,
        args=exact_args,
        total_steps=3,
        is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=exact_args),
    )
    assert exact_allowed is True
    assert exact_report.action == "exact"
    assert exact_batches.materialize(0, phase="single")[0].shape == (1, 10)

    eager_batches = _make_shape_guard_text_plan((10, 11, 30), labeled=False)
    eager_args = MLXTrainingConfig(compile=False, compile_max_variants=1)
    shape_plan, report, allowed, _ = _plan_single_process_text_shapes(
        eager_batches,
        None,
        args=eager_args,
        total_steps=3,
        is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=eager_args),
    )
    assert shape_plan is None and allowed is True
    assert (report.action, report.reason) == ("not_applicable", "compile_disabled")
    assert eager_batches[0][0].shape == (1, 10)


def test_text_shape_guard_failure_obeys_best_effort_and_strict_modes():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )

    schedules = ((0,), (1, 0))
    best_effort = MLXTrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=1,
        compile_max_variants=1,
    )
    _, report, allowed, _ = _plan_single_process_text_shapes(
        _make_shape_guard_text_plan((10, 11), schedules=schedules),
        None,
        args=best_effort,
        total_steps=2,
        is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=best_effort),
    )
    assert allowed is False
    assert (report.action, report.reason) == ("eager", "irreducible_signatures")

    strict = MLXTrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=1,
        compile_mode="strict",
        compile_max_variants=1,
    )
    with pytest.raises(RuntimeError, match="shape planning failed"):
        _plan_single_process_text_shapes(
            _make_shape_guard_text_plan((10, 11), schedules=schedules),
            None,
            args=strict,
            total_steps=2,
            is_vlm=False,
            distributed_world_size=1,
            compile_policy=build_compile_policy(args=strict),
        )


def test_strict_text_shape_rejection_precedes_model_setup():
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _shape_guard_report,
    )

    class Model:
        _config = {}

        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer(
        Model(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [],
        args=MLXTrainingConfig(
            max_steps=1,
            gradient_accumulation_steps=2,
            compile_mode="strict",
            compile_max_variants=1,
        ),
    )
    trainer._batches = _make_shape_guard_text_plan((10, 11), labeled=False)
    setup_calls = []
    trainer._install_neftune = lambda: setup_calls.append("neftune")

    with pytest.raises(RuntimeError, match="shape planning failed"):
        trainer.train()

    assert setup_calls == []


def test_ddp_text_shape_preparation_failure_is_coordinated_before_setup():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class Model:
        _config = {}

        def trainable_parameters(self):
            return {}

    trainer = MLXTrainer(
        Model(),
        types.SimpleNamespace(pad_token_id=0, eos_token_id=2),
        [],
        args=MLXTrainingConfig(max_steps=1),
    )
    trainer._distributed_initialized = True
    trainer._distributed_world_size = 2
    trainer._distributed_rank = 0
    trainer._distributed_is_main_process = True
    trainer._prepare_data = lambda _is_vlm: (_ for _ in ()).throw(
        KeyError("rank-local preparation failure")
    )
    calls = []

    def coordinated_abort(failed, context, exc):
        calls.append((failed, context, type(exc)))
        raise RuntimeError("coordinated preparation failure")

    trainer._raise_distributed_failure = coordinated_abort
    trainer._install_neftune = lambda: calls.append("model setup")

    with pytest.raises(RuntimeError, match="coordinated preparation failure"):
        trainer.train()

    assert calls == [
        (True, "preparing finite text shape guard", KeyError),
    ]


def test_text_shape_guard_dispositions_for_vlm_streaming_and_clipped_accum():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )

    cases = (
        (True, None, MLXTrainingConfig(), "vlm"),
        (False, iter(()), MLXTrainingConfig(), "streaming"),
        (False, None, MLXTrainingConfig(gradient_accumulation_steps=2,
                                        max_grad_norm=1.0), None),
    )
    for is_vlm, batch_iter, args, reason in cases:
        batches = _make_shape_guard_text_plan((10, 30), labeled=False)
        shape_plan, report, compile_allowed, _ = _plan_single_process_text_shapes(
            batches,
            batch_iter,
            args=args,
            total_steps=2,
            is_vlm=is_vlm,
            distributed_world_size=1,
            compile_policy=build_compile_policy(args=args),
        )
        assert compile_allowed is True
        if reason is None:
            assert shape_plan is not None
        else:
            assert shape_plan is None
            assert (report.action, report.reason) == ("not_applicable", reason)
            assert batches[0][0].shape == (1, 10)


def test_ddp_text_shape_guard_uses_local_phases_and_rank_local_endpoints():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.shape_guard import DDP_LOCAL_GRAD_SCOPE
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )

    args = MLXTrainingConfig(
        max_steps=3,
        gradient_accumulation_steps=2,
        compile_max_variants=2,
    )
    endpoints = []
    for widths in ((10, 11, 30), (10, 11)):
        schedules = (
            ((0,), (1,), (1,), (0,)) if len(widths) == 2 else None
        )
        batches = _make_shape_guard_text_plan(
            widths, schedules=schedules, labeled=False,
        )
        shape_plan, report, allowed, _ = _plan_single_process_text_shapes(
            batches,
            None,
            args=args,
            total_steps=3 if len(widths) == 3 else 2,
            is_vlm=False,
            distributed_world_size=2,
            compile_policy=build_compile_policy(args=args),
        )
        assert allowed is True
        assert report.compile_scope == DDP_LOCAL_GRAD_SCOPE
        assert (report.raw_signatures, report.planned_signatures) == (
            2 * len(widths), 2,
        )
        endpoints.append(shape_plan.endpoint_for(batches.batch_family(0), 10))

    assert endpoints == [30, 11]


def test_ddp_text_shape_guard_coordinates_peer_failure_and_strict_mode():
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _shape_guard_report,
    )

    trainer = object.__new__(MLXTrainer)
    trainer._distributed_initialized = True
    trainer._distributed_world_size = 2
    trainer._distributed_any_flag = lambda _failed: True
    local_report = _shape_guard_report(
        "exact", "schedule_within_cap", 32, "ddp_local_grad",
    )
    policy = build_compile_policy(args=MLXTrainingConfig())
    _, report, allowed = trainer._coordinate_text_shape_guard(
        None, None, local_report, True, policy,
    )
    assert allowed is False
    assert (report.action, report.reason) == ("eager", "peer_planner_failure")
    assert report.planned_signatures is None

    strict_policy = build_compile_policy(
        args=MLXTrainingConfig(compile_mode="strict"),
    )
    with pytest.raises(RuntimeError, match="at least one DDP rank"):
        trainer._coordinate_text_shape_guard(
            None, None, local_report, True, strict_policy,
        )


@pytest.mark.parametrize("compile_failure", [None, "setup", "runtime"])
def test_text_trainer_bounds_compiled_signatures_and_unpads_fallback(
    monkeypatch, tmp_path, compile_failure,
):
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(128, 4)
            self.proj = nn.Linear(4, 128, bias=False)
            self._config = {"model_type": "tiny"}

        def __call__(self, input_ids):
            return self.proj(self.embed(input_ids))

        def train(self):
            return self

        @property
        def state(self):
            return []

    seen = set()
    executed_widths = set()
    failed_runtime = False

    def compile_spy(fn, **_kwargs):
        if compile_failure == "setup":
            raise RuntimeError("compile setup failure")

        def compiled(*args):
            nonlocal failed_runtime
            batch, prev_state, do_update = args
            seen.add((int(batch[0].shape[-1]), prev_state is None, bool(do_update)))
            if compile_failure == "runtime" and not failed_runtime:
                failed_runtime = True
                raise RuntimeError("compile runtime failure")
            return fn(*args)
        return compiled

    def value_and_grad_with_aux(model, fn):
        from mlx.utils import tree_map

        def wrapped(*args):
            executed_widths.add(int(args[1].shape[-1]))
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())

        return wrapped

    monkeypatch.setattr(mx, "compile", compile_spy)
    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)
    args = MLXTrainingConfig(
        max_steps=6,
        gradient_accumulation_steps=1,
        compile=True,
        compile_max_variants=2,
        use_cce=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        logging_steps=6,
        output_dir=str(tmp_path),
    )
    trainer = MLXTrainer(
        TinyLM(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [],
        args=args,
    )
    trainer._batches = _make_shape_guard_text_plan(
        (10, 11, 30), labeled=False,
    )
    trainer._build_optimizer = lambda _total_steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5),
        state={},
        update=lambda _model, _grad: None,
    )
    trainer.save_model = lambda *_args, **_kwargs: None

    result = trainer.train()

    assert result["compile_shape_guard"]["action"] == "bucket"
    assert result["compile_shape_guard"]["planned_signatures"] == 2
    if compile_failure is None:
        assert result["compile_enabled"] is True
        assert {signature[0] for signature in seen} == {11, 30}
        assert executed_widths == {11, 30}
        assert len(seen) == 2
    else:
        assert result["compile_enabled"] is False
        assert result["compile_scope"] == "fallback_eager"
        assert executed_widths == {10, 11, 30}


def test_response_masked_text_batches_can_remain_a_lazy_plan():
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _create_labeled_batches
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan

    tokenizer = types.SimpleNamespace(
        chat_template=None,
        eos_token_id=None,
        pad_token_id=7,
        encode=lambda text, add_special_tokens=True: [
            int(part) for part in str(text).split()
        ],
    )

    def mask_fn(batch):
        ids = batch["input_ids"][0]
        return {"labels": [[-100] + ids[1:]]}

    kwargs = dict(
        dataset=[{"text": "1 2"}, {"text": "3 4 5"}],
        tokenizer=tokenizer,
        mask_fn=mask_fn,
        batch_size=2,
        max_seq_length=64,
        dataset_order="sequential",
    )
    plan = _create_labeled_batches(**kwargs, return_plan=True)
    eager = _create_labeled_batches(**kwargs)[0]

    assert isinstance(plan, FiniteTextBatchPlan)
    assert plan.widths == (33,)
    actual = plan[0]
    assert [value.tolist() for value in actual] == [
        value.tolist() for value in eager
    ]
    assert actual[2].dtype == mx.int32
    assert eager[2].dtype == mx.int32


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


@pytest.mark.parametrize("value", [True, False, 0, 257, 1.5, "32"])
def test_mlx_training_config_validates_compile_max_variants(value):
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    with pytest.raises(ValueError, match="compile_max_variants"):
        MLXTrainingConfig(compile_max_variants=value)


def test_mlx_trainer_distributed_defaults_world_size_one():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self): return {}

    trainer = MLXTrainer(DummyModel(), None, [], args=MLXTrainingConfig())

    # Seeding the on_init_end process-zero flags resolves the rank at
    # construction (mirroring HF, which knows its rank via the accelerator at
    # Trainer.__init__), so distributed metadata is initialized here. The shim's
    # mx.distributed.init() returns None, so this stays rank 0 / world size 1.
    assert trainer._distributed_initialized is True
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
        max_seq_length=64,
        seed=7,
        comm_group=FakeWorld(),
        tokenizer=Tokenizer(),
    )

    assert np.random.random(3) == pytest.approx(expected)
    assert batches[0][0].shape == (2, 33)
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

    from unsloth_zoo.mlx.trainer import _clip_grad_norm_fp32, _global_grad_norm_fp32

    norm_source = inspect.getsource(_global_grad_norm_fp32)
    assert "g.astype(mx.float32)" in norm_source
    assert "tree_reduce" in norm_source
    assert "scale.astype(g.dtype)" in inspect.getsource(_clip_grad_norm_fp32)


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


def test_text_prepare_data_passes_completion_only_loss_to_batch_plan(monkeypatch):
    from unsloth_zoo.mlx import trainer as mlx_trainer

    received = {}

    def fake_create_plan(**kwargs):
        received.update(kwargs)
        return [("batch", "lengths", "labels")]

    monkeypatch.setattr(mlx_trainer, "_create_text_batch_plan", fake_create_plan)

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
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    received = {}

    def fake_hf(trainer, *, instruction_part=None, response_part=None,
                force_match=True, tokenizer=None, return_function=False,
                num_proc=None, last_response_only=False):
        received["last_response_only"] = last_response_only
        return lambda batch: batch

    monkeypatch.setattr(dataset_utils, "train_on_responses_only", fake_hf)
    train_on_responses_only(
        None,
        instruction_part="<user>",
        response_part="<assistant>",
        tokenizer=CallableTokenizer(),
        return_function=True,
        last_response_only=True,
    )

    assert received["last_response_only"] is True


def test_response_mask_tokenizer_rejects_encode_only_tokenizer():
    from unsloth_zoo.mlx.trainer import _resolve_response_mask_tokenizer

    class EncodeOnlyTokenizer:
        def encode(self, text):
            return [1, 2, 3]

        def convert_tokens_to_ids(self, token):
            return 1

    with pytest.raises(TypeError, match="requires a callable"):
        _resolve_response_mask_tokenizer(EncodeOnlyTokenizer())


def test_vlm_eval_batches_define_completion_only_loss_before_use():
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    source = inspect.getsource(MLXTrainer._train_inner)
    definition = source.index("text_completion_only_loss = _text_completion_only_loss_arg(args)")
    eval_use = source.index("completion_only_loss=text_completion_only_loss")
    text_eval_start = source.index("return create_batches(")
    text_eval_end = source.index("if isinstance(self.eval_dataset, dict)")
    text_eval_block = source[text_eval_start:text_eval_end]
    assert definition < eval_use
    assert "completion_only_loss=text_completion_only_loss" in text_eval_block


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


def test_callback_state_num_input_tokens_seen_uses_reduced_global_count():
    # HF's TrainerState.num_input_tokens_seen is a GLOBAL (all-rank gathered)
    # count of INPUT tokens that callbacks read directly to report progress or
    # stop on a token budget. The training loop must increment it from the
    # all-reduced global_input_toks (the batch input numel summed across ranks),
    # NOT global_toks (the supervised/label-token count from the loss mask) and
    # NOT the rank-local value (which would undercount by ~world_size under DDP).
    import inspect
    import re

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    m = re.search(
        r"self\.state\.num_input_tokens_seen\s*\+=\s*int\(\s*([A-Za-z_]+)\.item\(\)\s*\)",
        src,
    )
    assert m is not None, "num_input_tokens_seen increment not found"
    assert m.group(1) == "global_input_toks", (
        "callback-visible num_input_tokens_seen must use the all-reduced "
        f"INPUT-token count global_input_toks, not {m.group(1)}"
    )
    # global_input_toks must be the all-reduced batch input-token count, reduced
    # before it is consumed for the callback state.
    reduce_at = src.index("global_input_toks = self._distributed_all_sum(")
    assert reduce_at < m.start()
    assert "_mlx_batch_input_token_count(batch_data)" in src


def test_num_input_tokens_seen_incremented_before_on_optimizer_step():
    # HF advances TrainerState.num_input_tokens_seen right after the forward pass,
    # BEFORE firing on_optimizer_step (transformers Trainer._inner_training_loop),
    # so a callback that reports or stops on a token budget observes this step's
    # tokens at the step it fires on. The MLX loop must increment ahead of the
    # on_optimizer_step fire; incrementing after would make the callback lag one
    # optimizer step behind the true token count.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    incr_at = src.index("self.state.num_input_tokens_seen +=")
    fire_at = src.index('_fire("on_optimizer_step")')
    assert incr_at < fire_at, (
        "num_input_tokens_seen must be incremented before on_optimizer_step fires"
    )
    # And still after the all-reduce that produces the global count it consumes.
    reduce_at = src.index("global_input_toks = self._distributed_all_sum(")
    assert reduce_at < incr_at


def test_mlx_batch_input_token_count_counts_all_positions():
    # The helper feeding num_input_tokens_seen must count every input position
    # (prompt + response + padding), matching HF's input_ids.numel(), for both
    # the text/preference/GRPO tuple batch and the VLM dict batch, and degrade to
    # 0 (rather than raise) when no input-id tensor is present.
    import mlx.core as mx
    from unsloth_zoo.mlx.trainer import _mlx_batch_input_token_count

    # tuple batch: (input_ids[B, L], lengths/aux, labels) -> B*L
    tup = (mx.zeros((3, 5), dtype=mx.int32), mx.zeros((3, 2)), mx.zeros((3, 5)))
    assert _mlx_batch_input_token_count(tup) == 15
    # dict (VLM) batch keyed by input_ids -> numel
    assert _mlx_batch_input_token_count({"input_ids": mx.zeros((2, 7))}) == 14
    # no input ids present -> 0, never raises
    assert _mlx_batch_input_token_count({"pixel_values": mx.zeros((2, 3))}) == 0
    assert _mlx_batch_input_token_count(None) == 0


def test_on_init_end_receives_seeded_process_zero_flags():
    # on_init_end must see the real per-rank process-zero flags (not the default
    # True-on-every-rank), so a DDP callback gating file I/O on
    # is_world_process_zero during on_init_end runs once, not once per rank.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class DummyModel:
        def trainable_parameters(self):
            return {}

    seen = {}

    class RecCb:
        def on_init_end(self, args, state, control, **kw):
            seen["wpz"] = state.is_world_process_zero
            seen["lpz"] = state.is_local_process_zero

    trainer = MLXTrainer(
        DummyModel(), None, None,
        callbacks=[RecCb()], args=MLXTrainingConfig(),
    )
    # The state passed to on_init_end must carry the resolved rank flags, equal
    # to is_main_process (rank 0 -> True in the single-process shim).
    assert seen["wpz"] == trainer.is_main_process
    assert seen["lpz"] == trainer.is_main_process


def test_on_init_end_dispatch_uses_distributed_failure_consensus():
    # Because on_init_end now runs with rank-specific process-zero flags, a
    # rank-0-only callback failure there must abort every rank via the same DDP
    # failure consensus as _fire; otherwise rank 0 unwinds __init__ while peers
    # proceed into train() and hang at the next collective.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer.__init__)
    assert '"on_init_end"' in src
    oi = src.index('"on_init_end"')
    tail = src[oi:]
    # The dispatch is caught and OR-reduced across ranks before continuing.
    assert "_init_error" in tail
    assert "_raise_distributed_failure(" in tail


def test_num_input_tokens_seen_persisted_and_restored_across_resume():
    # HF saves num_input_tokens_seen in trainer_state.json and restores it on
    # resume; the MLX loop increments it every step, so it must be checkpointed
    # and restored into the callback-visible state (else a token-budget callback
    # restarts at 0 after resume and overruns).
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    ti = inspect.getsource(MLXTrainer._train_inner)
    # Saved in the checkpoint state dict ...
    assert '"num_input_tokens_seen": int(' in ti
    # ... and restored from it on resume into the resume attr.
    assert 'ts.get("num_input_tokens_seen"' in ti
    assert "_resume_num_input_tokens_seen = int(" in ti
    # ... then seeded into the callback-visible TrainerState.
    ics = inspect.getsource(MLXTrainer._init_callback_state)
    assert "num_input_tokens_seen=int(" in ics
    assert "_resume_num_input_tokens_seen" in ics


def test_callback_batches_per_epoch_uses_single_pass_for_max_steps():
    # With max_steps>0, `batches` is the whole cycled run (max_steps*grad_accum
    # micro-batches), so the per-epoch count must be the single-pass
    # approximation ceil(dataset_len / (per_device_batch * world)), NOT
    # len(batches); otherwise state.epoch climbs to 1.0 across the run and epoch
    # callbacks fire once instead of per dataset pass.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    t = MLXTrainer.__new__(MLXTrainer)
    t.args = MLXTrainingConfig(max_steps=50, per_device_train_batch_size=2)
    t._distributed_world_size = 1
    t._mlx_train_dataset_for_batches = list(range(8))  # 8 examples
    t.train_dataset = t._mlx_train_dataset_for_batches
    t._prepared_batches_include_epochs = False
    # One pass = ceil(8 / (2*1)) = 4 micro-batches, not the 100-batch run length.
    assert t._callback_batches_per_epoch(list(range(100))) == 4
    # No upper clamp to the run length: even when the whole max_steps run is
    # shorter than one dataset pass, the per-epoch denominator stays one_pass so
    # state.epoch = it / one_pass reports the true fraction of a pass. Clamping to
    # the run length (min(one_pass, total)) would make a sub-one-pass run report a
    # full 1.0 epoch instead of HF's fractional value.
    assert t._callback_batches_per_epoch(list(range(3))) == 4

    # The epoch-based path (max_steps<=0, num_train_epochs>0) is unchanged.
    t2 = MLXTrainer.__new__(MLXTrainer)
    t2.args = MLXTrainingConfig(num_train_epochs=3, max_steps=-1)
    t2._prepared_batches_include_epochs = True
    assert t2._callback_batches_per_epoch(list(range(12))) == 4  # 12 // 3


def test_callback_batches_per_epoch_sub_one_pass_reports_fractional_epoch():
    # A max_steps run that stops before completing even one dataset pass must
    # report a FRACTIONAL state.epoch (HF: global_step / (updates per pass)), not a
    # spurious full 1.0. Bug: min(one_pass, total) clamped the per-epoch denominator
    # down to the (short) run length, so state.epoch = it / batches_per_epoch hit
    # 1.0 by the end of the run. Fix: max(1, one_pass) keeps the true pass length.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    t = MLXTrainer.__new__(MLXTrainer)
    # 100 examples, per_device 1 -> one_pass = 100 micro-batches, but max_steps=10
    # means the whole run is only 10 micro-batches (a tenth of a pass).
    t.args = MLXTrainingConfig(max_steps=10, per_device_train_batch_size=1)
    t._distributed_world_size = 1
    t._mlx_train_dataset_for_batches = list(range(100))
    t.train_dataset = t._mlx_train_dataset_for_batches
    t._prepared_batches_include_epochs = False
    bpe = t._callback_batches_per_epoch(list(range(10)))  # run length = 10
    assert bpe == 100, "denominator must be one_pass (100), not the run length (10)"
    # After the whole 10-step run, state.epoch = it / bpe = 10/100 = 0.1 (HF value),
    # whereas the old clamp gave 10/10 = 1.0 (a full phantom epoch).
    assert 10 / bpe == 0.1


def test_should_epoch_stop_field_reset_and_honored():
    # HF's TrainerControl.should_epoch_stop must (1) exist so a callback reading
    # it does not AttributeError, (2) be reset at on_epoch_begin, and (3) be
    # honored by ending the current epoch early -- skipping its remaining
    # micro-batches to the next epoch boundary, rank-synced for DDP lockstep.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer, _MLXTrainerControl

    # (1) Field exists and defaults False.
    assert _MLXTrainerControl().should_epoch_stop is False

    src = inspect.getsource(MLXTrainer._train_inner)
    # (2) Reset at epoch begin.
    assert "self.control.should_epoch_stop = False" in src
    # (3) Rank-synced honoring: an all-reduced flag drives an epoch-boundary skip.
    assert "def _sync_epoch_stop" in src
    assert "_distributed_any_flag(self.control.should_epoch_stop)" in src
    assert "_sync_epoch_stop()" in src
    # The honor fast-forwards the batch cursor to the next epoch boundary (shared
    # skip helper), and only for materialized batches (a streaming iterator can't
    # be index-skipped).
    assert "def _honor_epoch_stop_skip" in src
    assert "batch_idx += next_boundary - it_val" in src
    assert "batch_iter is None" in src
    # On an epoch-count-driven path the shortened epoch also shrinks the
    # optimizer-step budget so the run does not overtrain past num_train_epochs.
    # The budget is recomputed from the micro-batches that remain after the skip
    # (conceptual total minus the advanced cursor). Using
    # _epoch_stop_total_microbatches covers BOTH epoch layouts (the default cycled
    # single-pass and the torch_randperm materialized-all-epochs path); the old
    # flag-gated len(batches) form skipped the default path.
    assert "(_epoch_stop_total_microbatches - batch_idx) // grad_accum" in src
    assert '_epoch_stop_total_microbatches' in src


def test_train_entry_clears_stale_stop_before_setup():
    # Regression for "Keep callback stops from poisoning trainer reuse" +
    # "Preserve external stop requests during setup".
    # stop_requested is a persistent instance flag; a run whose callback latched
    # should_training_stop leaves it True. The clear that unblocks a reused
    # trainer must happen at train() ENTRY, before any long setup (data prep /
    # optimizer build), so a stale PRIOR-run stop is gone before setup begins
    # while an external cancel raised DURING this run's setup still survives.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer.train)
    reset_idx = src.index("self.stop_requested = False")
    # The clear must precede the _train_inner() call (and the data prep it
    # drives), else a stale stop would only be cleared after setup (or not).
    inner_idx = src.index("self._train_inner()")
    assert reset_idx < inner_idx
    # It must be the FIRST executable statement (only the docstring + comments
    # precede it), so no long work runs before the stale stop is cleared. The
    # statement is the generation guard, which clears only a PRIOR run's stop.
    body = src[src.index('"""', src.index('"""') + 3) + 3:]
    first_stmt = next(
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert first_stmt.startswith("if self._stop_request_generation() <")


def test_reset_run_state_preserves_in_setup_stop_request():
    # Regression for "Preserve external stop requests during setup".
    # _train_inner calls _reset_run_state AFTER _prepare_data / _build_optimizer.
    # An external controller (e.g. a Studio cancel button) may set stop_requested
    # while those long setup steps run; the post-setup _reset_run_state must NOT
    # clear that in-flight cancel, or training proceeds despite the cancel. So
    # _reset_run_state no longer owns stop_requested at all.
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)

    # An external cancel set after the train()-entry clear but during setup must
    # survive _reset_run_state so the loop's _distributed_should_stop() sees it.
    trainer.stop_requested = True
    trainer._reset_run_state()
    assert trainer.stop_requested is True

    # _reset_run_state still clears _early_stopped so a run-1 early stop doesn't
    # block run 2 on a reused trainer.
    trainer._early_stopped = True
    trainer._reset_run_state()
    assert trainer._early_stopped is False


def test_reset_run_state_does_not_own_stop_requested_attr():
    # _reset_run_state must not touch stop_requested at all: with the attribute
    # absent, the reset leaves it absent (train() entry is the sole owner).
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer._reset_run_state()
    assert not hasattr(trainer, "stop_requested")


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


def test_on_step_end_defers_callback_stop_until_after_same_step_eval():
    # Regression for the "defer stop-control until after same-step eval" bug.
    # HF runs this step's log/evaluate/save before the loop breaks, so a stop
    # requested by a callback on on_step_end must NOT be copied into
    # stop_requested before the same-step eval: _evaluate_batch_totals skips
    # every eval batch while stop_requested is set, which reports 0.0 loss and
    # corrupts best-model / early-stopping state. Only an external cancel may be
    # OR-reduced here; the callback stop is applied by the tail _sync_stop().
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    step_end = src.index('_fire("on_step_end")')
    eval_block = src.index("should_eval = (", step_end)
    tail_stop = src.index("if _sync_stop():", eval_block)
    # Strip comment lines so the prose describing the deferral is not mistaken
    # for the code that performs it.
    between = "\n".join(
        line for line in src[step_end:eval_block].splitlines()
        if not line.strip().startswith("#")
    )

    # The callback stop must not be latched into stop_requested before the eval.
    assert "_sync_stop()" not in between
    assert "_sync_callback_stop()" not in between
    # Only the external-cancel OR-reduce is allowed ahead of the same-step eval.
    assert "self._distributed_should_stop()" in between
    # The deferred callback stop is applied after log/eval/save (loop tail).
    assert tail_stop > eval_block


def test_same_step_eval_not_preempted_by_callback_stop_ddp(monkeypatch):
    # world_size == 2 regression proving DDP lockstep is preserved: a callback
    # stop deferred past the same-step eval still stops every rank, the same-step
    # eval reports the real loss (not 0.0), and a stop on rank 0 stops the peer.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, _MLXTrainerControl

    # all_sum == identity plus a configurable peer contribution, so rank 0 (which
    # owns the callback stop) and its peer can be modelled independently here.
    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    class Model:
        def __init__(self): self.modes = []
        def eval(self): self.modes.append("eval")
        def train(self): self.modes.append("train")

    def make_trainer(rank, local_stop):
        t = MLXTrainer.__new__(MLXTrainer)
        t.model = Model()
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        t.stop_requested = local_stop
        t.control = _MLXTrainerControl()
        t._last_eval_metrics = {}
        return t

    def loss_fn(_model, _batch, _lengths, _labels):
        return mx.array(2.0), mx.array(4)

    eval_batches = [("a", None, None), ("b", None, None)]

    # Rank 0: a callback sets should_training_stop during on_step_end.
    rank0 = make_trainer(rank=0, local_stop=False)
    rank0.control.should_training_stop = True
    peer["value"] = 0  # the peer requested nothing this step
    # Re-implanted on_step_end sync: log/eval/save flags then only an external
    # OR-reduce. The callback stop must stay deferred.
    rank0._distributed_sync_control_actions()
    rank0._distributed_should_stop()
    assert rank0.stop_requested is False

    # The same-step eval therefore consumes every batch and reports real loss.
    loss, _ = rank0._evaluate(eval_batches, loss_fn, is_vlm=False)
    assert loss == pytest.approx(2.0)
    assert loss != 0.0

    # The deferred callback stop, applied by the tail _sync_stop(), stops rank 0.
    rank0._sync_callback_stop()
    assert rank0._distributed_should_stop() is True
    assert rank0.stop_requested is True

    # Lockstep: the stop on rank 0 must OR-reduce onto the peer (rank 1), which
    # requested nothing locally, so no rank is left spinning at the next
    # collective.
    rank1 = make_trainer(rank=1, local_stop=False)
    rank1.control.should_training_stop = False
    peer["value"] = 1  # rank 0 contributes its stop into the reduction
    rank1._sync_callback_stop()
    assert rank1.stop_requested is False
    assert rank1._distributed_should_stop() is True
    assert rank1.stop_requested is True

    # Contrast: the pre-fix ordering (stop latched before eval) skips every eval
    # batch and reports 0.0, corrupting best-model tracking.
    buggy = make_trainer(rank=0, local_stop=True)
    peer["value"] = 0
    buggy_loss, _ = buggy._evaluate(eval_batches, loss_fn, is_vlm=False)
    assert buggy_loss == 0.0


def test_on_optimizer_step_defers_callback_stop_until_after_same_step_eval():
    # Regression for the on_optimizer_step twin of the on_step_end deferral bug.
    # HF fires on_optimizer_step, then on_step_end, then _maybe_log_save_evaluate
    # (log+eval+save) for this step, and only breaks on should_training_stop
    # AFTER that block. A stop requested by an on_optimizer_step callback must
    # therefore NOT be copied into stop_requested before the same-step eval:
    # _evaluate_batch_totals skips every eval batch while stop_requested is set,
    # which reports 0.0 loss and corrupts best-model / early-stopping state. Only
    # an external cancel may be OR-reduced here; the callback stop is applied by
    # the tail _sync_stop().
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    opt_step = src.index('_fire("on_optimizer_step")')
    # The shared post-update bookkeeping begins once the do_update branch closes
    # (the `if batches_per_epoch:` epoch update that runs every microstep); bound
    # the inspected region there so we only look at what runs between the
    # on_optimizer_step event and the rest of the same step. (num_input_tokens_seen
    # is now incremented ahead of on_optimizer_step, so it no longer marks the
    # branch close.)
    region_end = src.index("if batches_per_epoch:", opt_step)
    tail_stop = src.index("if _sync_stop():", region_end)
    # Strip comment lines so the prose describing the deferral is not mistaken
    # for the code that performs it.
    between = "\n".join(
        line for line in src[opt_step:region_end].splitlines()
        if not line.strip().startswith("#")
    )

    # The callback stop must not be latched into stop_requested before the eval.
    assert "_sync_stop()" not in between
    assert "_sync_callback_stop()" not in between
    # Only the external-cancel OR-reduce is allowed ahead of the same-step eval.
    assert "self._distributed_should_stop()" in between
    # The deferred callback stop is applied after log/eval/save (loop tail).
    assert tail_stop > region_end


def test_on_optimizer_step_stop_not_preempted_same_step_eval_ddp(monkeypatch):
    # world_size == 2 regression proving DDP lockstep is preserved for the
    # on_optimizer_step deferral: a callback stop deferred past the same-step
    # eval still stops every rank, the same-step eval reports the real loss (not
    # 0.0), and a stop on rank 0 OR-reduces onto the peer.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, _MLXTrainerControl

    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    class Model:
        def __init__(self): self.modes = []
        def eval(self): self.modes.append("eval")
        def train(self): self.modes.append("train")

    def make_trainer(rank, local_stop):
        t = MLXTrainer.__new__(MLXTrainer)
        t.model = Model()
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        t.stop_requested = local_stop
        t.control = _MLXTrainerControl()
        t._last_eval_metrics = {}
        return t

    def loss_fn(_model, _batch, _lengths, _labels):
        return mx.array(2.0), mx.array(4)

    eval_batches = [("a", None, None), ("b", None, None)]

    # Rank 0: an on_optimizer_step callback sets should_training_stop. The fix
    # runs only the external-cancel OR-reduce at that point (no latch), so the
    # callback stop stays deferred and stop_requested is still False.
    rank0 = make_trainer(rank=0, local_stop=False)
    rank0.control.should_training_stop = True
    peer["value"] = 0  # the peer requested nothing this step
    rank0._distributed_should_stop()
    assert rank0.stop_requested is False

    # The same-step eval therefore consumes every batch and reports real loss.
    loss, _ = rank0._evaluate(eval_batches, loss_fn, is_vlm=False)
    assert loss == pytest.approx(2.0)
    assert loss != 0.0

    # The deferred callback stop, applied by the tail _sync_stop(), stops rank 0.
    rank0._sync_callback_stop()
    assert rank0._distributed_should_stop() is True
    assert rank0.stop_requested is True

    # Lockstep: the stop on rank 0 OR-reduces onto the peer (rank 1), which
    # requested nothing locally, so no rank is left spinning at the next
    # collective.
    rank1 = make_trainer(rank=1, local_stop=False)
    rank1.control.should_training_stop = False
    peer["value"] = 1  # rank 0 contributes its stop into the reduction
    rank1._sync_callback_stop()
    assert rank1.stop_requested is False
    assert rank1._distributed_should_stop() is True
    assert rank1.stop_requested is True


def test_on_log_control_actions_synced_before_eval_save():
    # Regression for "Sync callback actions raised from on_log in DDP". on_log
    # fires on rank 0 only and HF checks should_evaluate/should_save after the
    # log in the same step, so a callback that requests an eval/save inside
    # on_log sets the flag on rank 0 alone. _run_training_log must OR-sync those
    # flags across ranks (_distributed_sync_control_actions) right after the
    # on_log dispatch, before the caller's collective eval/save branches, or
    # rank 0 enters _run_eval/_run_checkpoint while peers skip them and hang.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    log_fire = src.index('_fire("on_log", logs=logs)')
    # The sync must follow the on_log dispatch inside _run_training_log.
    sync_after = src.find("self._distributed_sync_control_actions()", log_fire)
    assert sync_after != -1
    # ...and land before the loss counter reset that ends _run_training_log, so
    # the synced flags are the ones the caller's should_eval/should_save read.
    reset_after = src.index("losses = 0", log_fire)
    assert sync_after < reset_after


def test_on_log_eval_request_or_syncs_onto_peer_ddp(monkeypatch):
    # world_size == 2: a callback sets should_evaluate during on_log on rank 0
    # only; _distributed_sync_control_actions must OR it onto the peer (rank 1)
    # so both ranks agree to enter the collective eval, none left spinning.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, _MLXTrainerControl

    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    def make_trainer(rank):
        t = MLXTrainer.__new__(MLXTrainer)
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        t.control = _MLXTrainerControl()
        return t

    base = 2 + 1  # _distributed_sync_control_actions packs flags base-(world+1)

    # Rank 0's on_log requested an eval; the peer requested nothing this step.
    rank0 = make_trainer(rank=0)
    rank0.control.should_evaluate = True
    peer["value"] = base  # rank 1 contributes 0 to the should_evaluate digit
    rank0._distributed_sync_control_actions()
    assert rank0.control.should_evaluate is True

    # Rank 1 saw no local request but must adopt rank 0's eval after the sync,
    # so it enters the same collective eval instead of hanging.
    rank1 = make_trainer(rank=1)
    rank1.control.should_evaluate = False
    peer["value"] = base  # rank 0 contributes its should_evaluate into the OR
    rank1._distributed_sync_control_actions()
    assert rank1.control.should_evaluate is True


def test_run_eval_syncs_control_actions_after_on_evaluate():
    # Regression for "Synchronize save requests raised by eval callbacks". Inside
    # _run_eval, on_log and on_evaluate fire on rank 0 only and HF checks
    # should_save after on_evaluate in the same step, so a callback that requests
    # a save inside either event sets the flag on rank 0 alone. _run_eval must
    # OR-sync the control action flags (_distributed_sync_control_actions) after
    # the on_evaluate dispatch, before returning to the caller's should_save
    # branch, or rank 0 enters the collective _run_checkpoint while peers skip it
    # and hang at the next collective.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    eval_def = src.index("def _run_eval(current_step):")
    eval_body = src[eval_def:src.index("def _run_best_tracking(", eval_def)]
    evaluate_fire = eval_body.index('_fire("on_evaluate"')
    sync_after = eval_body.find(
        "self._distributed_sync_control_actions()", evaluate_fire
    )
    # The sync must follow the on_evaluate dispatch...
    assert sync_after != -1
    # ...and land before _run_eval returns, so the synced flags are the ones the
    # caller's should_log / should_save branches read.
    assert sync_after < eval_body.index("return True", evaluate_fire)


def test_on_evaluate_save_request_or_syncs_onto_peer_ddp(monkeypatch):
    # world_size == 2: a callback sets should_save during on_evaluate on rank 0
    # only; _distributed_sync_control_actions must OR it onto the peer (rank 1)
    # so both ranks agree to enter the collective checkpoint save, none left
    # spinning at the on_save / _raise_distributed_failure collective.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, _MLXTrainerControl

    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    def make_trainer(rank):
        t = MLXTrainer.__new__(MLXTrainer)
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        t.control = _MLXTrainerControl()
        return t

    base = 2 + 1  # flags pack base-(world+1); should_save is the base*base digit

    # Rank 0's on_evaluate requested a save; the peer requested nothing.
    rank0 = make_trainer(rank=0)
    rank0.control.should_save = True
    peer["value"] = 0  # rank 1's code contributes 0 to the should_save digit
    rank0._distributed_sync_control_actions()
    assert rank0.control.should_save is True

    # Rank 1 saw no local request but must adopt rank 0's save after the sync,
    # so it enters the same collective checkpoint instead of hanging.
    rank1 = make_trainer(rank=1)
    rank1.control.should_save = False
    peer["value"] = base * base  # rank 0 contributes its should_save into the OR
    rank1._distributed_sync_control_actions()
    assert rank1.control.should_save is True


def test_fire_rank_zero_callback_failure_syncs_across_ranks(monkeypatch):
    # Regression for "Synchronize rank-zero callback failures". A callback that
    # raises on rank 0 must not unwind rank 0 alone: the peers never enter the
    # rank-0-only dispatch, so they would return and hang at the next collective
    # while rank 0 aborts. _fire routes the rank-0 failure through the
    # distributed consensus (_raise_distributed_failure), which every rank calls
    # in lockstep, so all ranks raise together and the original error surfaces.
    import inspect

    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer

    # Source-level: _fire wraps the rank-0 call_event and routes failures through
    # the distributed consensus path rather than propagating on rank 0 alone.
    src = inspect.getsource(MLXTrainer._train_inner)
    fire_def = src.index("def _fire(event, **kwargs):")
    fire_body = src[fire_def:src.index("def _sync_stop():", fire_def)]
    assert "call_event" in fire_body
    assert "except Exception" in fire_body
    assert "self._raise_distributed_failure(" in fire_body

    # Behavioral world_size == 2 consensus: rank 0 failed, peer succeeded, both
    # must raise (no peer left waiting at the collective).
    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    def make_trainer(rank):
        t = MLXTrainer.__new__(MLXTrainer)
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        t.stop_requested = False
        return t

    # Rank 0's callback raised; its failure flag is 1, the peer contributes 0.
    rank0 = make_trainer(rank=0)
    peer["value"] = 0
    with pytest.raises(RuntimeError, match="callback"):
        rank0._raise_distributed_failure(True, "on_log callback", ValueError("boom"))

    # Rank 1 saw no local failure but the reduced consensus is non-zero, so it
    # aborts too instead of hanging at the next all-reduce.
    rank1 = make_trainer(rank=1)
    peer["value"] = 1  # rank 0 contributes its failure into the reduction
    with pytest.raises(RuntimeError, match="peer rank failed"):
        rank1._raise_distributed_failure(False, "on_log callback")


def test_init_callback_state_seeds_best_from_restored_resume_state():
    # Regression for "Seed callback best state when resuming". On resume the
    # native best fields (self._best_metric/_best_step) are restored before
    # _init_callback_state, but the fresh TrainerState leaves best_metric=None.
    # HF callbacks (EarlyStoppingCallback) and _update_callback_best_metric would
    # then treat the first post-resume eval as the new best and overwrite the
    # real best with a worse metric. _init_callback_state must seed the visible
    # best fields from the restored native best state.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    def make_shell(best_metric, best_step):
        t = MLXTrainer.__new__(MLXTrainer)
        t.args = MLXTrainingConfig(output_dir="out_dir")
        t._distributed_initialized = True
        t._distributed_is_main_process = True
        t._distributed_world_size = 1
        t._distributed_rank = 0
        t.callback_handler = types.SimpleNamespace(
            call_event=lambda *a, **k: k.get("control", a[3] if len(a) > 3 else None)
        )
        t._best_metric = best_metric
        t._best_step = best_step
        return t

    # Resume: restored best is seeded into the callback-visible state.
    resumed = make_shell(best_metric=0.5, best_step=7)
    resumed._init_callback_state(total_steps=100, resume_step=7)
    assert resumed.state.best_metric == 0.5
    assert resumed.state.best_global_step == 7
    assert resumed.state.best_model_checkpoint == "out_dir/best"

    # A fresh run has no prior best; the fields stay None (no phantom best).
    fresh = make_shell(best_metric=None, best_step=None)
    fresh._init_callback_state(total_steps=100, resume_step=0)
    assert fresh.state.best_metric is None
    assert fresh.state.best_global_step is None
    assert fresh.state.best_model_checkpoint is None

    # With the seed in place, a worse post-resume eval must NOT overwrite the
    # restored best (greater_is_better=False: lower eval_loss is better).
    resumed.args.metric_for_best_model = "eval_loss"
    resumed.args.greater_is_better = False
    resumed._update_callback_best_metric({"eval_loss": 0.9})
    assert resumed.state.best_metric == 0.5  # unchanged: 0.9 is worse than 0.5
    # A genuine improvement still updates it.
    resumed._update_callback_best_metric({"eval_loss": 0.3})
    assert resumed.state.best_metric == 0.3


def test_run_best_tracking_not_skipped_by_callback_stop():
    # Regression for "Defer callback stop until after best tracking". HF's
    # _maybe_log_save_evaluate runs _determine_best_metric and writes the
    # checkpoint for the current step BEFORE the loop honors should_training_stop,
    # so a callback that requests a stop on on_step_end / on_evaluate must not
    # skip the best-model save for this step's (valid, possibly improving) eval.
    # _run_eval must therefore NOT copy the callback stop into stop_requested
    # before _run_best_tracking (whose _track guard is `not self.stop_requested`);
    # only an external cancel is OR-reduced there, and the callback stop is
    # applied by the caller's tail _sync_stop().
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    eval_def = src.index("def _run_eval(current_step):")
    eval_body = src[eval_def:src.index("def _run_best_tracking(", eval_def)]

    # Strip comments so the prose describing the deferral is not mistaken for the
    # code that performs it.
    code = "\n".join(
        line for line in eval_body.splitlines()
        if not line.strip().startswith("#")
    )
    # The callback stop must NOT be latched inside _run_eval (it would make
    # _run_best_tracking skip a valid eval's best-model save).
    assert "_sync_callback_stop()" not in code
    # An external cancel is still OR-reduced before the divergent best-model
    # branch so peers do not hang at the rank-0-guarded best save collective.
    assert "self._distributed_should_stop()" in code

    # The main loop applies the deferred callback stop only after best tracking
    # and the same-step save (the loop-tail _sync_stop()).
    best_call = src.index("_run_best_tracking(current_step)", src.index("if should_eval:"))
    tail_stop = src.index("if _sync_stop():", best_call)
    assert tail_stop > best_call


def test_run_best_tracking_runs_after_callback_stop_ddp(monkeypatch):
    # world_size == 2 lockstep: a callback stop deferred past best tracking still
    # leaves _track rank-consistent (every rank reads the same stop_requested,
    # which reflects only external cancels), so the rank-0-guarded best-model save
    # in _run_best_tracking never diverges. Model the two ranks independently.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, _MLXTrainerControl

    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    def make_trainer(rank, external_stop=False):
        t = MLXTrainer.__new__(MLXTrainer)
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        t.stop_requested = external_stop
        t.control = _MLXTrainerControl()
        return t

    # Rank 0's on_evaluate requested a stop but the eval was valid; no external
    # cancel is pending on either rank. The end-of-_run_eval OR-reduce sees only
    # external stops (none), so stop_requested stays False and _track proceeds.
    rank0 = make_trainer(rank=0)
    rank0.control.should_training_stop = True
    peer["value"] = 0
    rank0._distributed_should_stop()
    assert rank0.stop_requested is False  # callback stop NOT latched -> best runs

    rank1 = make_trainer(rank=1)
    peer["value"] = 0
    rank1._distributed_should_stop()
    assert rank1.stop_requested is False  # peer agrees: best tracking runs on both

    # A genuine external cancel on rank 0 mid-eval still OR-reduces onto the peer
    # so _track is skipped in lockstep (a garbage aborted eval is never "best").
    rank0c = make_trainer(rank=0, external_stop=True)
    peer["value"] = 0
    assert rank0c._distributed_should_stop() is True
    rank1c = make_trainer(rank=1)
    peer["value"] = 1  # rank 0 contributes its external cancel into the OR
    assert rank1c._distributed_should_stop() is True
    assert rank1c.stop_requested is True


def test_on_save_fires_only_after_checkpoint_written():
    # Regression for "Fire on_save only after writing a checkpoint". HF calls
    # callback_handler.on_save only after _save_checkpoint writes to disk. Here
    # save_trainable_adapters raises ValueError for a fully frozen / no-adapter
    # model and the write is skipped; firing on_save anyway makes integrations
    # (hub uploaders, checkpoint trackers) record a checkpoint-N that never
    # existed. on_save must be gated on the actual write, broadcast from rank 0.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    ckpt_def = src.index("def _run_checkpoint(current_step):")
    ckpt_body = src[ckpt_def:src.index(
        "def _run_callback_control_actions(", ckpt_def)]

    # The write outcome is tracked and only set after save_trainable_adapters
    # succeeds (inside the try/except-ValueError else branch).
    assert "checkpoint_written = False" in ckpt_body
    assert "checkpoint_written = True" in ckpt_body
    # on_save is dispatched only under the written guard, not unconditionally...
    guard = ckpt_body.index("if checkpoint_written_any:")
    assert guard < ckpt_body.index('_fire("on_save")')
    # ...and the rank-0-only write outcome is broadcast so every rank fires (or
    # skips) on_save together, or the skipping rank strands peers at the _fire
    # consensus collective.
    assert "_distributed_status_mask" in ckpt_body


def test_on_save_skip_broadcasts_to_peer_ddp(monkeypatch):
    # world_size == 2: when rank 0 skipped the checkpoint write (no trainable
    # adapter), the broadcast written flag is 0 on every rank, so no rank fires
    # on_save alone; when rank 0 wrote, the flag reaches the peer so both fire.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer

    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    def make_trainer(rank):
        t = MLXTrainer.__new__(MLXTrainer)
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        return t

    # Rank 0 skipped the write; both ranks contribute 0 -> neither fires on_save.
    rank0 = make_trainer(0)
    peer["value"] = 0
    assert (rank0._distributed_status_mask(0) > 0) is False
    rank1 = make_trainer(1)
    peer["value"] = 0  # rank 0 also contributed 0
    assert (rank1._distributed_status_mask(0) > 0) is False

    # Rank 0 wrote a checkpoint; the flag broadcasts to the peer so both fire.
    rank0w = make_trainer(0)
    peer["value"] = 0
    assert (rank0w._distributed_status_mask(1) > 0) is True
    rank1w = make_trainer(1)
    peer["value"] = 1  # rank 0 contributes its written flag into the reduction
    assert (rank1w._distributed_status_mask(0) > 0) is True


def test_epoch_end_fires_on_substep_boundary():
    # Regression for "Fire epoch-end callbacks before substep continue". When
    # batches-per-epoch is not a multiple of grad_accum, the epoch-boundary
    # microstep is a non-update (substep). The substep branch must fire
    # on_epoch_end (via _maybe_callback_epoch_end) before `continue`, or the
    # event and any log/eval/save/stop it requests are dropped for that epoch.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    substep = src.index("if not do_update:")
    branch = src[substep:src.index("continue", substep)]
    assert '_fire("on_substep_end")' in branch
    assert "_maybe_callback_epoch_end(" in branch


def test_epoch_boundary_can_land_on_substep():
    # Prove the boundary is reachable on a non-update microstep: replicate the
    # loop's accumulation cadence for grad_accum=2, batches_per_epoch=3. The
    # microstep at the epoch boundary (it % batches_per_epoch == 0) is a substep,
    # so on_epoch_end would be dropped without the substep-branch dispatch.
    grad_accum = 2
    batches_per_epoch = 3
    accum_progress = 0
    boundary_is_update = None
    for it in range(1, batches_per_epoch + 1):
        do_update = (accum_progress + 1 >= grad_accum)
        if it % batches_per_epoch == 0:
            boundary_is_update = do_update
        accum_progress = 0 if do_update else accum_progress + 1
    assert boundary_is_update is False


def test_substep_defers_callback_stop_until_after_epoch_end_eval():
    # Regression for the epoch-end twin of the on_step_end / on_optimizer_step
    # deferral bug. When batches-per-epoch is not a multiple of grad_accum, the
    # epoch-boundary microstep is a non-update (substep); on_substep_end can set
    # should_training_stop while _maybe_callback_epoch_end then fires on_epoch_end
    # and may run a same-epoch eval. Latching the stop before that eval makes
    # _evaluate_batch_totals skip every batch (it is gated on not stop_requested),
    # reporting 0.0 loss and corrupting best-model / early-stopping state. Only an
    # external cancel may be OR-reduced ahead of the epoch-end eval; the callback
    # stop is applied by the tail _sync_stop() after it.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    substep = src.index("if not do_update:")
    # Work on a comment-stripped slice of the substep branch (up to its continue)
    # so prose mentioning continue / _sync_stop / _maybe_callback_epoch_end is not
    # mistaken for the code that performs it.
    branch_lines = []
    for line in src[substep:].splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        branch_lines.append(line)
        if stripped == "continue":
            break
    branch = "\n".join(branch_lines)

    substep_end = branch.index('_fire("on_substep_end")')
    epoch_end_call = branch.index("_maybe_callback_epoch_end(")
    continue_idx = branch.index("continue", epoch_end_call)

    between = branch[substep_end:epoch_end_call]
    # The callback stop must not be latched into stop_requested before the
    # epoch-end eval.
    assert "_sync_stop()" not in between
    assert "_sync_callback_stop()" not in between
    # Only the external-cancel OR-reduce is allowed ahead of the epoch-end eval.
    assert "self._distributed_should_stop()" in between

    # The deferred callback stop is applied by a tail _sync_stop() after the
    # epoch-end log/eval/save and before the substep continue.
    after_epoch = branch[epoch_end_call:continue_idx]
    assert "_sync_stop()" in after_epoch


def test_substep_stop_not_preempted_by_epoch_end_eval_ddp(monkeypatch):
    # world_size == 2 regression proving DDP lockstep is preserved for the
    # substep epoch-end deferral: a callback stop set on on_substep_end at an
    # epoch-boundary microstep is deferred past the epoch-end eval (which reports
    # the real loss, not 0.0), the tail _sync_stop() then applies it, and a stop
    # on rank 0 OR-reduces onto the peer so no rank is left spinning.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, _MLXTrainerControl

    peer = {"value": 0}
    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array(peer["value"], dtype=value.dtype)
    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    class Model:
        def __init__(self): self.modes = []
        def eval(self): self.modes.append("eval")
        def train(self): self.modes.append("train")

    def make_trainer(rank, local_stop):
        t = MLXTrainer.__new__(MLXTrainer)
        t.model = Model()
        t._distributed_initialized = True
        t._distributed_world = object()
        t._distributed_world_size = 2
        t._distributed_rank = rank
        t._distributed_is_main_process = (rank == 0)
        t.stop_requested = local_stop
        t.control = _MLXTrainerControl()
        t._last_eval_metrics = {}
        return t

    def loss_fn(_model, _batch, _lengths, _labels):
        return mx.array(2.0), mx.array(4)

    eval_batches = [("a", None, None), ("b", None, None)]

    # Rank 0: an on_substep_end callback sets should_training_stop at an epoch
    # boundary. The substep branch runs only the external-cancel OR-reduce there
    # (no latch), so the callback stop stays deferred and stop_requested is False.
    rank0 = make_trainer(rank=0, local_stop=False)
    rank0.control.should_training_stop = True
    peer["value"] = 0  # the peer requested nothing this step
    rank0._distributed_should_stop()
    assert rank0.stop_requested is False

    # The epoch-end eval therefore consumes every batch and reports the real loss.
    loss, _ = rank0._evaluate(eval_batches, loss_fn, is_vlm=False)
    assert loss == pytest.approx(2.0)
    assert loss != 0.0

    # The deferred callback stop, applied by the tail _sync_stop(), stops rank 0.
    rank0._sync_callback_stop()
    assert rank0._distributed_should_stop() is True
    assert rank0.stop_requested is True

    # Lockstep: the stop on rank 0 OR-reduces onto the peer (rank 1), which
    # requested nothing locally, so no rank is left spinning at the next
    # collective.
    rank1 = make_trainer(rank=1, local_stop=False)
    rank1.control.should_training_stop = False
    peer["value"] = 1  # rank 0 contributes its stop into the reduction
    rank1._sync_callback_stop()
    assert rank1.stop_requested is False
    assert rank1._distributed_should_stop() is True
    assert rank1.stop_requested is True


def _simulate_epoch_stop_loop(bpe, num_epochs, grad_accum, budget_rule,
                              include_epochs=True, max_iters=100000):
    """Replicate the flattened micro-batch loop's consume / skip / budget cadence.

    Models a callback that sets should_epoch_stop on every optimizer step (i.e.
    ends each epoch at its first update), which is the scenario Codex flagged: a
    skipped tail smaller than grad_accum. Returns the terminal loop state and
    whether the modulo fetch ever wrapped past the real materialized data.
    """
    n = bpe * num_epochs                    # len(batches): all epochs materialized
    total_steps = max(1, n // grad_accum)
    global_step = 0
    batch_idx = 0
    accum = 0
    microstep = 0
    wrapped = False
    guard = 0
    while global_step < total_steps:
        guard += 1
        if guard > max_iters:
            raise RuntimeError("loop did not terminate")
        it = microstep + 1
        if batch_idx >= n:                  # batches[batch_idx % n] re-uses data
            wrapped = True
        batch_idx += 1
        do_update = (accum + 1 >= grad_accum)
        if not do_update:
            accum += 1
            microstep = it
            continue
        global_step += 1
        accum = 0
        # Callback keeps should_epoch_stop set: honor it at every mid-epoch update.
        if it % bpe != 0:
            next_boundary = ((it // bpe) + 1) * bpe
            skipped = next_boundary - it
            batch_idx += skipped
            it = next_boundary
            if include_epochs:
                total_steps = budget_rule(total_steps, global_step, skipped,
                                          grad_accum, n, batch_idx)
        microstep = it
    return dict(global_step=global_step, batch_idx=batch_idx,
                wrapped=wrapped, total_steps=total_steps, n=n)


def test_epoch_stop_budget_recompute_no_data_reuse():
    # Regression for "Recompute remaining steps when skipping partial epochs".
    # bpe=7, grad_accum=4, 3 epochs (total_steps = 21 // 4 = 5). A callback ends
    # each epoch at its first optimizer step, so the skipped tail is 3 < grad_accum.
    # The old budget rule reduced total_steps by `skipped // grad_accum == 0`, so
    # the loop kept its original budget and wrapped batches[idx % 21] back into
    # already-seen data (overtraining + phantom epoch events). The new rule
    # recomputes the budget from the micro-batches that remain (len(batches) -
    # batch_idx), stopping cleanly when the materialized data is exhausted.
    def new_rule(total_steps, global_step, skipped, grad_accum, n, batch_idx):
        return global_step + (n - batch_idx) // grad_accum

    def old_rule(total_steps, global_step, skipped, grad_accum, n, batch_idx):
        return max(global_step, total_steps - skipped // grad_accum)

    new = _simulate_epoch_stop_loop(7, 3, 4, new_rule)
    old = _simulate_epoch_stop_loop(7, 3, 4, old_rule)

    # New rule: never re-reads data, one optimizer step per epoch, budget shrinks.
    assert new["wrapped"] is False
    assert new["global_step"] == 3
    assert new["batch_idx"] <= new["n"]     # cursor never passes the real data end
    assert new["total_steps"] == 3
    # Old rule: the sub-grad_accum tail left the budget unchanged, so the loop
    # cycled past the materialized data (the bug being fixed).
    assert old["wrapped"] is True
    assert old["global_step"] > new["global_step"]

    # A skipped tail that IS a whole grad_accum window was already handled by the
    # old rule; the new rule matches it (no regression on the aligned case).
    aligned_new = _simulate_epoch_stop_loop(8, 3, 4, new_rule)
    assert aligned_new["wrapped"] is False


def test_epoch_stop_budget_recompute_present_in_source():
    # Guard the exact recompute expression so the fix is not silently reverted to
    # the floor-division form that under-counts sub-grad_accum tails, nor re-gated
    # on _prepared_batches_include_epochs (which skipped the default path).
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "(_epoch_stop_total_microbatches - batch_idx) // grad_accum" in src
    assert "total_steps - _skipped // grad_accum" not in src
    # The recompute is driven by the conceptual total micro-batches, which is set
    # for both epoch layouts (default cycled pass and torch_randperm), so it is not
    # gated behind _prepared_batches_include_epochs. The default (flag=False) path
    # multiplies the single materialized pass by num_train_epochs.
    assert "n_batches * int(args.num_train_epochs)" in src


def test_epoch_stop_skip_keeps_fractional_epoch():
    # Codex NEW-d: "Keep early-stopped epoch values fractional." When a callback
    # raises should_epoch_stop mid-epoch, _honor_epoch_stop_skip fires the truncated
    # epoch's on_epoch_end. HF sets state.epoch = epoch + (step+1)/steps_in_epoch at
    # the last optimizer step and does NOT snap it to the next integer when the epoch
    # is cut short (transformers _inner_training_loop fires on_epoch_end with that
    # fractional value). Snapping to ceil(it_val / batches_per_epoch) reported a full
    # epoch (e.g. 1.0) for a truncated one, so an epoch-based integration treated a
    # partial epoch as completed. The skip must set the fractional it_val /
    # batches_per_epoch instead.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    skip_def = src.index("def _honor_epoch_stop_skip(")
    skip_body = src[skip_def:src.index('_fire("on_epoch_end")', skip_def) + 40]
    # The fractional value is assigned; the integer-snapping ceil form is gone.
    assert "self.state.epoch = it_val / batches_per_epoch" in skip_body
    assert "math.ceil(it_val / batches_per_epoch)" not in skip_body

    # Pure-logic parity with HF: a callback stopping after the first optimizer step
    # of a long epoch reports the true partial fraction, not 1.0. batches_per_epoch
    # = 10, stop at the second micro-batch (it_val = 2, the first optimizer step for
    # grad_accum=2): HF epoch = 0 + 2/10 = 0.2, not ceil(0.2) = 1.0.
    batches_per_epoch = 10
    it_val = 2
    assert it_val / batches_per_epoch == pytest.approx(0.2)
    assert float(__import__("math").ceil(it_val / batches_per_epoch)) == 1.0  # old bug value

    # A stop deeper into a later epoch stays fractional and monotonic: epoch 2, three
    # micro-batches in -> it_val = 23, fraction = 2.3 (HF: 2 + 3/10), never 3.0.
    assert 23 / batches_per_epoch == pytest.approx(2.3)


def test_pending_metrics_flushed_on_early_callback_stop():
    # Regression for "Flush metrics before honoring callback stops". A
    # should_training_stop callback (or external cancel) can break the loop at a
    # step that is neither a logging_steps multiple nor the last step, so
    # _run_training_log never ran for that window. trained_tokens and
    # _train_loss_history are updated only inside _run_training_log, so without a
    # post-loop flush the returned train_loss/trained_tokens would be 0 despite
    # completed training. HF folds the trailing tr_loss into _total_loss_scalar
    # before computing the returned train_loss; assert the equivalent flush exists,
    # runs after the loop, is guarded by an unlogged-window check, and precedes the
    # returned avg_loss.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    loop_pos = src.index("while self._global_step < total_steps:")
    avg_pos = src.index("avg_loss = (")
    # The flush is gated on the COMMITTED window (steps > 0). Committed excludes any
    # not-yet-applied PENDING partial window (those micro-batches live in
    # pending_losses/pending_n_tokens/pending_steps), so the flush reports only
    # applied optimizer steps -- no accum_progress==0 clause is needed anymore.
    assert "if steps > 0:" in src
    flush_pos = src.index("if steps > 0:")
    assert loop_pos < flush_pos < avg_pos
    assert "_run_training_log(self._global_step, None)" in src[flush_pos:avg_pos]


def test_max_steps_can_end_off_epoch_boundary():
    # Reachability for the truncated-epoch on_epoch_end fix: max_steps rarely
    # aligns to a dataset boundary. With batches_per_epoch=5, grad_accum=2,
    # max_steps=3 the run ends at microstep 6, and 6 % 5 == 1 != 0, so the in-loop
    # boundary dispatch (gated on microstep % batches_per_epoch == 0) never fires
    # on_epoch_end for that final epoch.
    batches_per_epoch = 5
    grad_accum = 2
    max_steps = 3
    end_microstep = max_steps * grad_accum
    assert end_microstep % batches_per_epoch != 0


def test_on_epoch_end_fires_for_truncated_final_epoch():
    # Regression for "Close open epochs before leaving the loop". HF fires
    # on_epoch_end for a truncated final epoch after its inner step loop breaks
    # (max_steps ending mid-dataset, or a should_training_stop mid-epoch). The MLX
    # loop's only in-loop on_epoch_end dispatch is gated on
    # microstep % batches_per_epoch == 0, so a mid-epoch exit would drop the event.
    # Assert a post-loop dispatch closes the open epoch, guarded against a double
    # fire at a natural boundary, before on_train_end.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    tail = src[src.index("Close a truncated final epoch"):src.index("avg_loss = (")]
    assert "microstep % batches_per_epoch != 0" in tail
    assert '_fire("on_epoch_end")' in tail
    # The counter is advanced before the stop-break so a callback-stop exit still
    # leaves microstep pointing at the finished step for the guard above (the
    # assignment precedes the break comment + `if _sync_stop(): break`).
    assert (
        "microstep = it\n"
        "            # Propagate any stop set by the tail callbacks"
    ) in src


def test_epoch_stop_budget_default_path_no_data_reuse():
    # Regression for "Shrink epoch-stop budgets for default epoch runs". On the
    # default batching path (_prepared_batches_include_epochs=False, the documented
    # dataset_order="default") `batches` is ONE dataset pass cycled
    # num_train_epochs times via batches[idx % len], and the round-14 recompute was
    # gated behind that flag, so a should_epoch_stop callback advanced batch_idx but
    # never shrank total_steps -- the loop wrapped batches[idx % len] into extra
    # passes (phantom epochs, overtraining). The unified recompute uses the
    # conceptual total micro-batches (len(batches)*num_train_epochs on this path)
    # regardless of the flag. The simulation's n = bpe*num_epochs is that
    # conceptual total.
    def unified_rule(total_steps, global_step, skipped, grad_accum, n, batch_idx):
        return global_step + (n - batch_idx) // grad_accum

    def flag_gated_default_rule(total_steps, global_step, skipped, grad_accum,
                                n, batch_idx):
        # Old default-path behavior: recompute gated out (flag False) -> no change.
        return total_steps

    fixed = _simulate_epoch_stop_loop(7, 3, 4, unified_rule)
    buggy = _simulate_epoch_stop_loop(7, 3, 4, flag_gated_default_rule)

    assert fixed["wrapped"] is False
    assert fixed["global_step"] == 3               # 3 epochs, one opt-step each
    assert fixed["batch_idx"] <= fixed["n"]
    # The un-shrunk default-path budget cycles past the real data (the G bug).
    assert buggy["wrapped"] is True
    assert buggy["global_step"] > fixed["global_step"]


def test_substep_honors_epoch_stop_abandons_window():
    # Regression for "Honor epoch-stop requests from substep callbacks". With
    # grad_accum>1 a callback can set should_epoch_stop from on_substep_end on a
    # non-update microstep. HF checks should_epoch_stop after every inner-loop
    # iteration (substeps included) and breaks mid-accumulation-window, abandoning
    # the partial gradient. The substep branch must therefore also check
    # _sync_epoch_stop, discard the partial window (grad_accum_state = None) and
    # skip to the next boundary via the shared helper -- not finish the window and
    # apply an extra optimizer update on the ended epoch's (or wrapped next-epoch's)
    # data.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    substep = src.index("if not do_update:")
    branch = src[substep:src.index("continue", substep)]
    assert "_sync_epoch_stop()" in branch
    assert "grad_accum_state = None" in branch          # abandon the partial window
    assert "_honor_epoch_stop_skip(" in branch
    # The epoch-stop check comes after on_substep_end / _maybe_callback_epoch_end so
    # a boundary substep still fires its natural on_epoch_end exactly once.
    assert branch.index('_fire("on_substep_end")') < branch.index("_sync_epoch_stop()")
    assert branch.index("_maybe_callback_epoch_end(") < branch.index("_sync_epoch_stop()")


def test_substep_can_be_mid_epoch_with_grad_accum():
    # Reachability for the substep epoch-stop fix: a non-update microstep mid-epoch
    # exists. grad_accum=3, batches_per_epoch=6: microstep it=1 is a substep
    # (accum 0->1, not an optimizer step) and 1 % 6 != 0 (mid-epoch), so a callback
    # setting should_epoch_stop from on_substep_end lands on a mid-epoch substep the
    # branch must honor.
    grad_accum = 3
    batches_per_epoch = 6
    accum_progress = 0
    it = 1
    do_update = (accum_progress + 1 >= grad_accum)   # False -> substep
    assert do_update is False
    assert it % batches_per_epoch != 0               # mid-epoch


def test_boundary_substep_epoch_stop_keeps_gradient():
    # Regression for "Preserve boundary substep gradients on epoch-stop". The
    # substep should_epoch_stop branch must only abandon the accumulation window
    # for an ACTUAL mid-epoch skip (it % batches_per_epoch != 0). At a boundary
    # substep the normal loop carries the micro-batch's gradient into the next
    # window, so `grad_accum_state = None` must sit INSIDE the mid-epoch guard, not
    # before it -- otherwise the epoch's final batch gradient is dropped while its
    # loss/tokens were already counted (losses += lvalue * toks).
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    substep = src.index("if not do_update:")
    branch = src[substep:src.index("continue", substep)]
    guard_pos = branch.index("if it % batches_per_epoch != 0:")
    discard_pos = branch.index("grad_accum_state = None")
    reset_pos = branch.index("self.control.should_epoch_stop = False")
    # The window discard is guarded by (follows) the mid-epoch check, and both
    # precede the should_epoch_stop reset.
    assert guard_pos < discard_pos < reset_pos
    # The skip helper is also inside the mid-epoch guard.
    assert guard_pos < branch.index("_honor_epoch_stop_skip(")


def test_boundary_can_be_substep_with_grad_accum():
    # Reachability for the boundary-substep case: a non-update microstep that is
    # ALSO an epoch boundary exists. batches_per_epoch=3, grad_accum=2: it=1 is a
    # substep, it=2 an optimizer step, it=3 a substep AND 3 % 3 == 0 (boundary). A
    # callback setting should_epoch_stop at it=3 must NOT discard the carried
    # gradient.
    grad_accum = 2
    batches_per_epoch = 3
    accum_progress = 0
    states = []
    for it in range(1, batches_per_epoch + 1):
        do_update = (accum_progress + 1 >= grad_accum)
        states.append((it, do_update, it % batches_per_epoch == 0))
        accum_progress = 0 if do_update else accum_progress + 1
    # it=3: a substep (not an optimizer update) that is also an epoch boundary.
    assert (3, False, True) in states


def test_post_loop_epoch_end_runs_eval_for_callback_stop():
    # Regression for "Defer callback stops through final epoch-end eval". The
    # post-loop truncated on_epoch_end runs after the tail _sync_stop() latched
    # stop_requested, so a callback-requested epoch-end eval would hit
    # _evaluate_batch_totals' `not stop_requested` gate, skip every batch, and
    # dispatch a phantom 0.0 eval to on_log/on_evaluate -- corrupting best/early-
    # stop state. HF runs a real epoch-end eval for a callback stop, so lift the
    # callback stop around the final actions and restore it; a hard external cancel
    # (stop_requested without should_training_stop) keeps its suppression.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    tail = src[src.index("Close a truncated final epoch"):src.index("avg_loss = (")]
    # Callback-stop vs external-cancel distinguished rank-consistently.
    assert "_callback_stop = self._distributed_any_flag(" in tail
    assert 'getattr(self.control, "should_training_stop", False)' in tail
    # Run the final actions on a normal exit OR a callback stop; suppress a hard
    # external cancel. Lift stop_requested around the actions, then restore it.
    assert "if not self.stop_requested or _callback_stop:" in tail
    assert "self.stop_requested = False" in tail
    assert "self.stop_requested = _restore_stop" in tail
    # The lift/restore wraps the control actions (try/finally) so the stop is
    # always restored even if the eval raises.
    assert "finally:" in tail


def test_run_training_log_guards_empty_window():
    # Regression for "Guard epoch-end forced logs with pending tokens". A callback
    # forcing should_log again on a step that already logged (e.g. logging_steps=1
    # at a dataset boundary via on_epoch_end) re-enters _run_training_log with the
    # accumulators reset to plain int 0; without a guard metric_tokens.item() raises
    # (single process) or a phantom 0.0 log is emitted (DDP). The helper must early-
    # return when nothing is pending, mirroring HF's global_step >
    # _globalstep_last_logged guard, and before the first collective all-sum so DDP
    # stays in lockstep.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    log_start = src.index("def _run_training_log(")
    guard = src.index("if steps == 0:", log_start)
    allsum = src.index("_distributed_all_sum(losses", log_start)
    # The empty-window guard precedes the first all-reduce in the helper body.
    assert log_start < guard < allsum
    body = src[guard:allsum]
    assert "return" in body


def _simulate_forced_epoch_logs(grad_accum, bpe, n_microsteps):
    """Simulate committed/pending metrics with a forced log at every epoch boundary.

    Models Codex NEW-b: an epoch boundary that lands on a non-update substep fires
    on_epoch_end, and a logging callback there forces should_log. _run_training_log
    flushes the COMMITTED window only. Returns the list of logged windows (each the
    number of micro-batches reported) so a caller can assert no not-yet-applied
    micro-batch is logged and none is lost from a later window.
    """
    committed = 0
    pending = 0
    accum_progress = 0
    logged_windows = []
    logged_total = 0

    def run_training_log():
        nonlocal committed, logged_total
        if committed == 0:            # HF guard: global_step > _globalstep_last_logged
            return
        logged_windows.append(committed)
        logged_total += committed
        committed = 0

    for microstep in range(1, n_microsteps + 1):
        do_update = (accum_progress + 1 >= grad_accum)
        pending += 1
        if do_update:
            committed += pending
            pending = 0
            accum_progress = 0
        else:
            accum_progress += 1
        # Forced log at an epoch boundary (fired from on_epoch_end).
        if microstep % bpe == 0:
            run_training_log()
    # Post-loop committed flush (E-flush) emits any remaining applied window.
    run_training_log()
    return dict(
        logged_windows=logged_windows,
        logged_total=logged_total,
        pending_leftover=pending,
    )


def test_forced_epoch_log_flushes_only_committed_window():
    # Codex NEW-b: an epoch boundary on a non-update substep (grad_accum=2, bpe=3)
    # that forces should_log must log only the COMMITTED (applied) window, never the
    # pending partial window, and the pending micro-batch must still appear in a
    # later window's log rather than being reset away.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    # Source: _run_training_log flushes/reset the COMMITTED counters (losses/
    # n_tokens/steps); the fold into committed happens only under do_update, so a
    # forced log entered from the substep on_epoch_end path never flushes pending.
    src = inspect.getsource(MLXTrainer._train_inner)
    log_body = src[src.index("def _run_training_log("):src.index("def _run_eval(")]
    assert "pending_losses" not in log_body        # helper never reads/resets pending
    assert "_distributed_all_sum(losses" in log_body
    assert "losses = 0" in log_body                # resets committed only
    # The fold adds the pending window into committed and then resets pending.
    fold = src.index("losses += pending_losses")
    assert "pending_losses = 0" in src[fold:fold + 300]

    # Behaviour: over a 6-microstep run (two 3-batch epochs, grad_accum=2), the
    # boundary microsteps 3 and 6 are substeps. The forced epoch-end logs report the
    # applied windows only, and every micro-batch is accounted for exactly once
    # across the logs (none logged before it was applied, none dropped afterwards).
    sim = _simulate_forced_epoch_logs(grad_accum=2, bpe=3, n_microsteps=6)
    # microsteps 1..6: updates at 2 (window 1+2), 4 (window 3+4), 6 (window 5+6).
    # boundary log at microstep 3 flushes committed window {1,2}; at microstep 6 the
    # update {5,6} commits first, so committed = {3,4}+{5,6} = 4 micro-batches.
    assert sim["logged_windows"] == [2, 4]
    assert sim["logged_total"] == 6                # every micro-batch logged once
    assert sim["pending_leftover"] == 0            # nothing left un-applied


def test_early_stop_flush_emits_committed_not_pending_partial_window():
    # Regression for Codex NEW-a: "Preserve completed-step metrics across partial
    # stops." With gradient_accumulation_steps>1, a stop from on_substep_end (or an
    # external cancel) can break on a non-update substep AFTER an earlier optimizer
    # step that has not hit logging_steps, so losses/n_tokens would contain both the
    # APPLIED update and a new PENDING partial window. The old E-flush guard
    # (accum_progress == 0) skipped the flush entirely in that case, dropping the
    # completed update from the returned train_loss/trained_tokens. The fix splits
    # committed (applied) from pending (not-yet-applied) metrics so the post-loop
    # flush emits only committed -- fired on `if steps > 0:` regardless of a pending
    # window -- matching HF, which folds every applied step's tr_loss into
    # _total_loss_scalar but never logs a not-yet-applied partial window.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    # Per-microstep accumulation targets the PENDING window; the fold into the
    # COMMITTED window happens only on an applied optimizer step (do_update).
    assert "pending_losses += lvalue * toks" in src
    assert "pending_n_tokens += toks" in src
    assert "pending_steps += 1" in src
    # The fold folds pending into committed and then resets pending to zero.
    fold = src.index("losses += pending_losses")
    fold_block = src[fold:fold + 300]
    assert "n_tokens += pending_n_tokens" in fold_block
    assert "steps += pending_steps" in fold_block
    assert "pending_losses = 0" in fold_block
    # The post-loop flush is gated only on the committed window (no accum_progress
    # clause), so a completed update that shares a stop with a pending partial
    # window is still emitted rather than dropped.
    assert "if steps > 0:" in src
    assert "if steps > 0 and accum_progress == 0:" not in src


def _simulate_committed_pending(grad_accum, n_microsteps):
    """Replicate the loop's committed/pending accumulation over n_microsteps.

    Uses a unit loss/token contribution of 1 per micro-batch. Returns the terminal
    committed/pending counters (mirroring losses/steps and pending_losses/
    pending_steps) at a break after the last micro-batch, plus whether the post-loop
    E-flush (gated on committed steps > 0) would fire.
    """
    committed_steps = 0
    committed_toks = 0
    pending_steps = 0
    pending_toks = 0
    accum_progress = 0
    for _ in range(n_microsteps):
        do_update = (accum_progress + 1 >= grad_accum)
        pending_steps += 1
        pending_toks += 1
        if do_update:
            committed_steps += pending_steps
            committed_toks += pending_toks
            pending_steps = 0
            pending_toks = 0
            accum_progress = 0
        else:
            accum_progress += 1
    return dict(
        committed_steps=committed_steps,
        committed_toks=committed_toks,
        pending_steps=pending_steps,
        pending_toks=pending_toks,
        eflush_fires=(committed_steps > 0),
    )


def test_substep_stop_flushes_committed_drops_pending_partial_window():
    # Pure-logic proof of the Codex NEW-a fix. grad_accum=2, a stop lands on the
    # first substep of a NEW window (microstep 3) that follows an already-APPLIED
    # optimizer step (microstep 2). committed holds the applied window (micro-batches
    # 1+2); pending holds the un-applied micro-batch 3. The E-flush fires on
    # committed steps > 0 and reports ONLY the committed window (2 micro-batches),
    # so the completed update is not dropped, and the pending micro-batch is not
    # reported as trained.
    state = _simulate_committed_pending(grad_accum=2, n_microsteps=3)
    assert state["committed_steps"] == 2      # micro-batches 1+2 (applied window)
    assert state["committed_toks"] == 2
    assert state["pending_steps"] == 1        # micro-batch 3 (not applied)
    assert state["eflush_fires"] is True      # committed window is flushed

    # Contrast: a stop at the very first substep of the run (before any optimizer
    # step) leaves committed empty and pending holding the un-applied micro-batch,
    # so the E-flush is skipped (no phantom trained_tokens at global_step 0).
    first = _simulate_committed_pending(grad_accum=4, n_microsteps=1)
    assert first["committed_steps"] == 0
    assert first["pending_steps"] == 1
    assert first["eflush_fires"] is False

    # A run that ends exactly on an applied optimizer step has no pending remainder,
    # and the committed window flushes fully (parity with the pre-split E case).
    aligned = _simulate_committed_pending(grad_accum=2, n_microsteps=4)
    assert aligned["pending_steps"] == 0
    assert aligned["committed_steps"] == 4
    assert aligned["eflush_fires"] is True


def test_substep_epoch_stop_discard_clears_only_pending_window():
    # Codex NEW-b, discard side: when a mid-epoch substep honors should_epoch_stop it
    # abandons the partial accumulation window (grad_accum_state = None) -- those
    # micro-batches never updated the model. Their loss/tokens must NOT surface in a
    # forced epoch-end log or inflate trained_tokens, so the discard clears ONLY the
    # PENDING accumulators. The COMMITTED window (already-applied optimizer steps not
    # yet logged) must survive, so a truncated epoch-end forced log still reports the
    # completed update instead of dropping it (HF logs applied steps at on_epoch_end
    # and never the abandoned partial window).
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    substep = src.index("if not do_update:")
    branch = src[substep:src.index("continue", substep)]
    discard = branch.index("grad_accum_state = None")
    skip = branch.index("_honor_epoch_stop_skip(")
    # ONLY the pending accumulators are zeroed in the discard block, before the skip.
    for name in ("pending_losses = 0", "pending_n_tokens = 0", "pending_steps = 0"):
        pos = branch.index(name, discard)
        assert discard < pos < skip, f"{name} must be cleared at the discard"
    # The committed accumulators must NOT be reset in the discard block (a bare
    # "losses = 0" would drop a completed update); only the pending ones are.
    for committed in ("losses = 0", "n_tokens = 0", "steps = 0"):
        assert f"pending_{committed}" in branch
        # No committed reset (a "losses = 0" not preceded by "pending_") in the discard.
        assert f"\n                        {committed}" not in branch
    # The steps==0 committed-window guard in _run_training_log still suppresses a
    # forced log when nothing has been committed since the last log.
    assert "if steps == 0:" in src


def test_epoch_permuted_visits_are_deterministic_and_guard_enumerated():
    # Golden pure epoch permutations; guard raw catalog equals resolved visits.
    import numpy as np
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.shape_guard import phase_for_microstep
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _plan_single_process_text_shapes,
    )
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, _FiniteTextRow

    np.random.seed(999)  # ambient state must not influence visits
    rows = tuple(
        _FiniteTextRow(tuple(range(1, w + 1)), 1, tuple(range(1, w + 1)))
        for w in (10, 11, 30, 50)
    )
    plan = FiniteTextBatchPlan(
        rows, tuple((i,) for i in range(4)), max_seq_length=64, pad_id=99,
        visit_policy="epoch_permute", visit_seed=1,
    )
    assert [plan.batch_index_for_visit(v) for v in range(12)] == [
        0, 1, 2, 3, 2, 3, 1, 0, 3, 1, 0, 2,
    ]
    args = MLXTrainingConfig(max_steps=4, gradient_accumulation_steps=2)
    shape_plan, report, _ok, _ = _plan_single_process_text_shapes(
        plan, None, args=args, total_steps=4, is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=args),
    )
    assert shape_plan.raw_catalog == frozenset(
        (
            report.compile_scope,
            phase_for_microstep(report.compile_scope, 2, m),
            plan.batch_family(plan.batch_index_for_visit(m)),
            plan.batch_width(plan.batch_index_for_visit(m)),
        )
        for m in range(8)
    )


def test_callback_metadata_config_fields_are_appended_last():
    # logging_dir / run_name are new callback-visible run metadata. MLXTrainingConfig
    # binds positional arguments by field order, so a new field declared anywhere but
    # last silently shifts the positional slot of every field after it. It must also be
    # registered as an appended field, or a full-field config dict produced by an older
    # release stops being recognised as a wholesale copy and its default warmup_steps
    # then overrides an explicitly set warmup_ratio.
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    field_names = [f.name for f in dataclasses.fields(MLXTrainingConfig)]
    assert field_names[-2:] == ["logging_dir", "run_name"], field_names[-4:]
    # report_to keeps the positional slot it had before the metadata fields existed.
    assert field_names.index("report_to") == field_names.index("output_dir") + 1

    # A full-field dump from a release predating logging_dir / run_name still resolves
    # warmup from the ratio, exactly as a same-version dump does.
    legacy = {
        f.name: getattr(MLXTrainingConfig(warmup_ratio = 0.1), f.name)
        for f in dataclasses.fields(MLXTrainingConfig)
        if f.name not in ("logging_dir", "run_name")
    }
    restored = MLXTrainingConfig(**legacy)
    assert restored.warmup_ratio == 0.1
    assert restored._unsloth_mlx_warmup_steps_explicit is False
    full = MLXTrainingConfig(warmup_ratio = 0.1)
    same_version = MLXTrainingConfig(
        **{f.name: getattr(full, f.name) for f in dataclasses.fields(MLXTrainingConfig)}
    )
    assert same_version._unsloth_mlx_warmup_steps_explicit is False


def test_wandb_artifact_mode_suppressed_for_on_train_end():
    # transformers' WandbCallback.on_train_end logs its final-model artifact by
    # constructing a Torch Trainer around args/model and calling its Torch
    # save_model (integrations/integration_utils.py, 4.x and 5.x alike). MLX
    # models are not torch.nn.Module and MLXTrainingConfig is not a
    # TrainingArguments, so that constructor raises AttributeError
    # (full_determinism on 5.x, batch_eval_metrics on 4.57.x). The adapters are
    # already on disk by then, so the crash costs the caller the whole
    # MLXTrainOutput of a run that actually finished. The bridge must clear the
    # artifact mode for that one dispatch and restore it afterwards.
    import enum
    import inspect

    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _MLXCallbackHandler,
    )

    class _LogModel(str, enum.Enum):
        # Mirrors transformers' WandbLogModel (a str Enum with .is_enabled).
        CHECKPOINT = "checkpoint"
        END = "end"
        FALSE = "false"

        @property
        def is_enabled(self):
            return self in (_LogModel.CHECKPOINT, _LogModel.END)

    class WandbCallback:  # the class NAME is what the bridge matches on
        def __init__(self, mode):
            self._log_model = _LogModel(mode)
            self._initialized = True
            self.saw_enabled = None

        def on_train_end(self, args, state, control, **kwargs):
            self.saw_enabled = self._log_model.is_enabled
            if self._log_model.is_enabled:
                # Stands in for Trainer(args=MLXTrainingConfig, model=<mlx>).
                raise AttributeError(
                    "'MLXTrainingConfig' object has no attribute 'full_determinism'"
                )

    class CustomWandbCallback(WandbCallback):
        """Subclassing WandbCallback to customise logging is a common recipe,
        and it inherits the same on_train_end."""

    class OtherCallback:
        def __init__(self):
            self._log_model = _LogModel("end")

    artifact_cb = WandbCallback("end")
    checkpoint_cb = WandbCallback("checkpoint")
    subclass_cb = CustomWandbCallback("end")
    off_cb = WandbCallback("false")
    other_cb = OtherCallback()

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(max_steps=1)
    trainer.callback_handler = _MLXCallbackHandler(
        [artifact_cb, checkpoint_cb, subclass_cb, off_cb, other_cb],
        model=object(),
        processing_class=None,
        optimizer=None,
        lr_scheduler=None,
    )

    suppressed = trainer._suppress_torch_only_wandb_artifacts()
    assert [cb for cb, _ in suppressed] == [
        artifact_cb, checkpoint_cb, subclass_cb,
    ]
    # Both artifact modes are cleared for the dispatch...
    assert artifact_cb._log_model.is_enabled is False
    assert checkpoint_cb._log_model.is_enabled is False
    # ... a WandbCallback that never asked for artifacts is untouched, and a
    # same-shaped non-Wandb callback is never rewritten.
    assert off_cb._log_model is _LogModel.FALSE
    assert other_cb._log_model is _LogModel.END

    # The on_train_end dispatch now completes instead of raising.
    trainer.callback_handler.call_event(
        "on_train_end", trainer.args, object(), object(),
    )
    assert artifact_cb.saw_enabled is False
    assert checkpoint_cb.saw_enabled is False

    # The user's callbacks get their requested mode back afterwards.
    trainer._restore_wandb_artifact_modes(suppressed)
    assert artifact_cb._log_model is _LogModel.END
    assert checkpoint_cb._log_model is _LogModel.CHECKPOINT

    # ...and the training loop actually wires it around the real dispatch,
    # with a restore that survives a callback raising.
    source = inspect.getsource(MLXTrainer._train_inner)
    assert "_suppress_torch_only_wandb_artifacts()" in source
    assert 'finally:\n            self._restore_wandb_artifact_modes(' in source


def _tiny_lm_for_loop_tests():
    """Minimal MLX module the training loop can run end to end."""
    import mlx.nn as nn

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(128, 4)
            self.proj = nn.Linear(4, 128, bias=False)
            self._config = {"model_type": "tiny"}

        def __call__(self, input_ids):
            return self.proj(self.embed(input_ids))

        def train(self):
            return self

        @property
        def state(self):
            return []

    return TinyLM()


def _frozen_optimizer():
    """Optimizer stub that never changes the weights, so two runs see the
    identical per-batch losses and only the log cadence differs."""
    import mlx.core as mx

    return lambda _total_steps: types.SimpleNamespace(
        learning_rate=mx.array(1e-5),
        state={},
        update=lambda _model, _grad: None,
    )


def _patch_value_and_grad_with_aux(monkeypatch):
    """The MLX-on-torch shim's nn.value_and_grad has no aux support, so the
    trainer's (loss, tokens) return unpacks wrong. Return the real loss with
    zero gradients; the frozen optimizer keeps the weights fixed anyway."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map

    def value_and_grad_with_aux(model, fn):
        def wrapped(*args):
            return fn(*args), tree_map(mx.zeros_like, model.trainable_parameters())
        return wrapped

    monkeypatch.setattr(nn, "value_and_grad", value_and_grad_with_aux)


def test_returned_train_loss_is_independent_of_callback_log_cadence(monkeypatch):
    # A callback that sets control.should_log splits the accumulated loss into
    # different windows. _train_loss_history entries are per-window token-weighted
    # means, so averaging them UNWEIGHTED made MLXTrainOutput["train_loss"] move
    # when a logging callback was added to an otherwise identical run - the caller
    # got a different experiment metric for the same training. HF has no such
    # dependency (train_loss = _total_loss_scalar / effective_global_step,
    # transformers trainer.py). Aggregate loss*tokens / tokens instead.
    import tempfile

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class ForceLogAtStepOne:
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_log = True
            return control

    # One model instance for both runs: the frozen optimizer never updates it, so
    # every step sees the same weights and any difference in the reported loss can
    # only come from the aggregation.
    model = _tiny_lm_for_loop_tests()

    def run(callbacks):
        args = MLXTrainingConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            logging_steps=4,
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            output_dir=tempfile.mkdtemp(),
        )
        trainer = MLXTrainer(
            model,
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [],
            args=args,
            callbacks=callbacks,
        )
        # Uneven token counts per step: an unweighted mean over windows can only
        # match the weighted one when every window carries the same tokens.
        trainer._batches = _make_shape_guard_text_plan((30, 10, 10, 10))
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer, trainer.train()

    plain, plain_result = run(None)
    forced, forced_result = run([ForceLogAtStepOne()])

    # The callback really did change the cadence.
    assert len(plain._train_loss_history) == 1
    assert len(forced._train_loss_history) == 2
    # ...but not the number handed back to the caller.
    assert forced_result["train_loss"] == pytest.approx(
        plain_result["train_loss"], rel=1e-5,
    )
    # Guard the guard: the old unweighted mean would have reported something else.
    unweighted = (
        sum(forced._train_loss_history) / len(forced._train_loss_history)
    )
    assert unweighted != pytest.approx(plain_result["train_loss"], rel=1e-5)


def test_streaming_runs_report_a_numeric_epoch_to_callbacks(monkeypatch):
    # Streaming has no finite dataset length, so batches_per_epoch is None. Leaving
    # state.epoch at None diverges from HF, which falls back to steps_in_epoch =
    # max_steps * gradient_accumulation_steps for a length-less dataloader, and it
    # breaks stock integrations: WandbCallback.on_save builds its checkpoint alias
    # as f"epoch_{round(state.epoch, 2)}" (transformers integrations/
    # integration_utils.py), which raises TypeError on None and takes down a run
    # that was training fine.
    import tempfile

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class EpochAliasCallback:
        def __init__(self):
            self.step_epochs = []
            self.save_aliases = []

        def on_step_end(self, args, state, control, **kwargs):
            self.step_epochs.append(state.epoch)
            return control

        def on_save(self, args, state, control, **kwargs):
            # Verbatim shape of WandbCallback.on_save's alias.
            self.save_aliases.append(f"epoch_{round(state.epoch, 2)}")
            return control

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    spy = EpochAliasCallback()
    args = MLXTrainingConfig(
        max_steps=4,
        gradient_accumulation_steps=1,
        logging_steps=4,
        save_steps=2,
        streaming=True,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=tempfile.mkdtemp(),
    )
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [],
        args=args,
        callbacks=[spy],
    )
    stream = iter([make_batch(10) for _ in range(4)])
    trainer._prepare_data = lambda _is_vlm: (None, stream)
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None

    trainer.train()

    # HF's (step + 1) / (max_steps * grad_accum) progress, not None.
    assert spy.step_epochs == [0.25, 0.5, 0.75, 1.0]
    assert spy.save_aliases == ["epoch_0.5", "epoch_1.0"]


def test_resume_mid_epoch_fires_epoch_begin(monkeypatch):
    # Codex NEW-d: "Fire epoch-begin when resuming inside an epoch." HF dispatches
    # on_epoch_begin at the top of every epoch of `range(epochs_trained, ...)`,
    # including the resumed partial one, and only afterwards skips its
    # already-trained batches. The MLX loop only fired begin at exact boundaries,
    # so a run resumed mid-epoch delivered that epoch's on_epoch_end with no
    # preceding begin and a freshly constructed callback tore down per-epoch state
    # it never set up.
    import tempfile

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class EpochOrderCallback:
        def __init__(self):
            self.events = []

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.events.append("begin")
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            self.events.append("end")
            return control

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    out_dir = tempfile.mkdtemp()
    # 4 micro-batches per epoch, 6 steps, checkpoint at 3: mid-epoch-2 resume.
    batches = [make_batch(10) for _ in range(6)]

    def build(spy):
        args = MLXTrainingConfig(
            max_steps=6,
            gradient_accumulation_steps=1,
            logging_steps=6,
            save_steps=3,
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            output_dir=out_dir,
        )
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [{"text": f"row {i}"} for i in range(4)],
            args=args,
            callbacks=[spy],
        )
        trainer._prepare_data = lambda _is_vlm: (list(batches), None)
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer

    build(EpochOrderCallback()).train()
    ckpt = os.path.join(out_dir, "checkpoint-3")
    assert os.path.isdir(ckpt), sorted(os.listdir(out_dir))

    resumed = EpochOrderCallback()
    build(resumed).train(resume_from_checkpoint=ckpt)

    # Every end is paired with a preceding begin, and the resumed partial epoch
    # opens with one rather than closing an epoch nobody opened.
    assert resumed.events, resumed.events
    assert resumed.events[0] == "begin", resumed.events
    depth = 0
    for event in resumed.events:
        depth += 1 if event == "begin" else -1
        assert depth in (0, 1), resumed.events


def test_train_entry_preserves_pre_start_cancel(monkeypatch):
    # An external controller owns stop_requested and may raise it at any moment,
    # including between construction and train(): Unsloth Studio's cancel poller
    # does exactly that (studio worker sets trainer.stop_requested = True as soon
    # as the trainer is registered, well before train() is reached, then reads the
    # flag back after train() returns to report "cancelled"). An unconditional
    # clear at train() entry silently discarded that cancel and ran the whole job.
    # The generation stamp clears only a stop left latched by an EARLIER run.
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    steps = []

    class StepSpy:
        def on_step_end(self, args, state, control, **kwargs):
            steps.append(state.global_step)
            return control

    def build():
        args = MLXTrainingConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            logging_steps=4,
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            output_dir=tempfile.mkdtemp(),
        )
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [{"text": f"row {i}"} for i in range(4)],
            args=args,
            callbacks=[StepSpy()],
        )
        trainer._prepare_data = lambda _is_vlm: ([make_batch(10) for _ in range(4)], None)
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer

    # Cancelled after construction, before train(): nothing must run.
    trainer = build()
    trainer.stop_requested = True
    trainer.train()
    assert steps == [], steps
    # Studio reads the flag after train() returns to distinguish cancelled runs.
    assert trainer.stop_requested is True

    # A stop still latched from a finished run is stale and must NOT block the
    # next run of a reused trainer.
    steps.clear()
    trainer.train()
    assert steps == [1, 2, 3, 4], steps


def test_train_bumps_run_generation_in_finally():
    # The stamp only distinguishes runs if every train() closes its generation,
    # including one that raised, else a stop latched by a failed run would be
    # treated as belonging to the next run and block it forever.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer.train)
    bump = "self._run_generation = getattr(self, \"_run_generation\", 0) + 1"
    assert bump in src
    assert src.rindex("finally:") < src.index(bump)


def test_stateful_callbacks_exported_into_checkpoints(monkeypatch):
    # TrainerState declared stateful_callbacks and nothing ever wrote it, so the
    # field was permanently {} and checkpoints carried no callback bookkeeping at
    # all. HF populates it in _save_checkpoint unconditionally (the opt-in flag
    # gates only the RESTORE side), so without this a checkpoint written today
    # can never have that state recovered, by this release or a later one.
    import json

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class Patience:
        """Shaped like transformers' ExportableState callbacks."""
        def __init__(self):
            self.counter = 0

        def on_step_end(self, args, state, control, **kwargs):
            self.counter += 1
            return control

        def state(self):
            return {"args": {}, "attributes": {"counter": self.counter}}

    class NotExportable:
        def state(self):
            raise NotImplementedError

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    out_dir = tempfile.mkdtemp()
    args = MLXTrainingConfig(
        max_steps=4,
        gradient_accumulation_steps=1,
        logging_steps=4,
        save_steps=2,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=out_dir,
    )

    def build(callbacks):
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [{"text": f"row {i}"} for i in range(4)],
            args=args,
            callbacks=callbacks,
        )
        trainer._prepare_data = lambda _is_vlm: ([make_batch(10) for _ in range(4)], None)
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer

    build([Patience(), NotExportable()]).train()

    with open(os.path.join(out_dir, "checkpoint-4", "trainer_state.json")) as fh:
        saved = json.load(fh)
    # The exporting callback is recorded by class name; the one whose state()
    # raises NotImplementedError is skipped rather than aborting the save.
    assert saved["stateful_callbacks"] == {
        "Patience": {"args": {}, "attributes": {"counter": 4}}
    }
    assert "NotExportable" not in saved["stateful_callbacks"]

    # Resuming mirrors the checkpoint into the callback-visible state instead of
    # leaving the declared field empty.
    resumed = build([Patience()])
    resumed.train(resume_from_checkpoint=os.path.join(out_dir, "checkpoint-2"))
    assert resumed.state.stateful_callbacks["Patience"]["attributes"]["counter"] == 2


def test_pre_optimizer_step_callback_fires_before_each_update(monkeypatch):
    # HF dispatches on_pre_optimizer_step immediately before optimizer.step()
    # (transformers trainer.py: clip -> on_pre_optimizer_step -> optimizer.step()
    # -> on_optimizer_step). The MLX loop only fired on_optimizer_step, so any
    # supplied TrainerCallback relying on the pre-update hook was silently inert
    # for the whole run even though the callback was otherwise accepted.
    import tempfile

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class OptimizerHookSpy:
        def __init__(self):
            self.events = []

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self.events.append(("pre", state.global_step))
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            self.events.append(("post", state.global_step))
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            self.events.append(("substep", state.global_step))
            return control

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    spy = OptimizerHookSpy()
    args = MLXTrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=2,
        logging_steps=100,
        save_steps=100,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=tempfile.mkdtemp(),
    )
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [{"text": f"row {i}"} for i in range(4)],
        args=args,
        callbacks=[spy],
    )
    batches = [make_batch(10) for _ in range(4)]
    trainer._prepare_data = lambda _is_vlm: (list(batches), None)
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None

    trainer.train()

    kinds = [kind for kind, _step in spy.events]
    # One pre-update dispatch per optimizer step, never on accumulation substeps.
    assert kinds.count("pre") == 2, spy.events
    # ... and each one immediately precedes that step's on_optimizer_step.
    assert [k for k in kinds if k in ("pre", "post")] == [
        "pre", "post", "pre", "post",
    ], spy.events
    for index, (kind, _step) in enumerate(spy.events):
        if kind == "pre":
            assert spy.events[index + 1][0] == "post", spy.events


def test_callback_batches_per_epoch_uses_prepared_plan_cycle():
    # For max_steps runs the callback epoch length must come from the rows
    # batching actually retained, not from len(dataset). Pretokenized rows under
    # two tokens are dropped and the tail partial batch is not emitted, so the
    # raw-dataset approximation ceil(len(ds) / batch) overshoots the real cycle
    # and on_epoch_begin/on_epoch_end land on micro-batches that are not dataset
    # boundaries.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import _create_text_batch_plan

    tokenizer = types.SimpleNamespace(pad_token_id=0, eos_token_id=2)
    # 12 source rows, every third is a single token -> filtered by the >=2 guard,
    # leaving 8 prepared rows == 4 micro-batches at batch size 2.
    dataset = [
        {"input_ids": [1]} if index % 3 == 0 else {"input_ids": [1, 2, 3, 4, 5]}
        for index in range(12)
    ]
    plan = _create_text_batch_plan(
        dataset, tokenizer, 2, 64, num_batches=8, seed=42,
    )
    assert len(plan) == 8, "the max_steps horizon cycles the plan"
    assert plan.cycle_length == 4, "one dataset pass is 4 micro-batches"

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(max_steps=8, per_device_train_batch_size=2)
    trainer._distributed_world_size = 1
    trainer._mlx_train_dataset_for_batches = dataset
    trainer.train_dataset = dataset
    trainer._prepared_batches_include_epochs = False
    # ceil(12 / 2) = 6 is the raw-dataset approximation; the truth is 4.
    assert trainer._callback_batches_per_epoch(plan) == 4

    # Plain materialized batch lists (no plan metadata) keep the old fallback.
    assert trainer._callback_batches_per_epoch(list(range(8))) == 6
