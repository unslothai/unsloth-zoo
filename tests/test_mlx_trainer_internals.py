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

    from unsloth_zoo.mlx.trainer import _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    legacy_fields = [
        field for field in dataclasses.fields(MLXTrainingConfig)
        if field.init and field.name not in _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    ]
    legacy_values = [getattr(MLXTrainingConfig(), field.name) for field in legacy_fields]
    _legacy_names = [field.name for field in legacy_fields]
    legacy_values[_legacy_names.index("warmup_ratio")] = 0.1
    # image_size is no longer the last legacy field, so set it by name.
    legacy_values[_legacy_names.index("image_size")] = (128, 256)
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


def _compat_args(**overrides):
    """Minimal MLXTrainingConfig-like args for _ensure_callback_args_compat."""
    import types

    fields = dict(
        logging_steps=1, eval_steps=0, save_steps=0,
        output_dir="out", logging_dir=None, run_name=None,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_synthesized_eval_strategy_refreshes_when_eval_is_enabled_later():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = _compat_args()
    trainer.eval_dataset = None
    trainer._ensure_callback_args_compat()
    assert trainer.args.eval_strategy == "no"

    # eval_dataset/eval_steps stay writable, and _ensure_callback_args_compat
    # re-runs per train(): a stale "no" makes EarlyStoppingCallback assert.
    trainer.eval_dataset = [{}]
    trainer.args.eval_steps = 1
    trainer._ensure_callback_args_compat()
    assert trainer.args.eval_strategy == "steps"

    # ... and back off again when eval is disabled for a later run.
    trainer.args.eval_steps = 0
    trainer._ensure_callback_args_compat()
    assert trainer.args.eval_strategy == "no"

    # Same derivation for the other synthesized strategies.
    trainer.args.save_steps = 5
    trainer.args.logging_steps = 0
    trainer._ensure_callback_args_compat()
    assert trainer.args.save_strategy == "steps"
    assert trainer.args.logging_strategy == "no"


def test_caller_supplied_strategies_are_never_overwritten():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    # A real TrainingArguments/SFTConfig already carries these fields.
    trainer.args = _compat_args(
        eval_strategy="epoch", logging_strategy="epoch", save_strategy="epoch",
    )
    trainer.eval_dataset = None
    trainer._ensure_callback_args_compat()
    trainer.args.eval_steps = 1
    trainer.eval_dataset = [{}]
    trainer._ensure_callback_args_compat()
    assert trainer.args.eval_strategy == "epoch"
    assert trainer.args.logging_strategy == "epoch"
    assert trainer.args.save_strategy == "epoch"


def test_user_override_of_synthesized_eval_strategy_survives():
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = _compat_args()
    trainer.eval_dataset = None
    trainer._ensure_callback_args_compat()
    assert trainer.args.eval_strategy == "no"
    # An explicit override of our synthesized value wins over the derivation.
    trainer.args.eval_strategy = "epoch"
    trainer._ensure_callback_args_compat()
    assert trainer.args.eval_strategy == "epoch"
    trainer.eval_dataset = [{}]
    trainer.args.eval_steps = 1
    trainer._ensure_callback_args_compat()
    assert trainer.args.eval_strategy == "epoch"


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
    assert re.search(
        r"_mlx_batch_input_token_count\(\s*batch_data\b", src,
    ) is not None


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
    # The honor fast-forwards to the next epoch boundary (shared skip helper) for
    # every run with a known epoch length: materialized batches advance the index,
    # declared-length streams drain the producer to the same boundary. Gating the
    # honor itself on `batch_iter is None` dropped the request for streams.
    assert "def _honor_epoch_stop_skip" in src
    assert "batch_idx += next_boundary - it_val" in src
    assert "for _ in range(next_boundary - it_val):" in src
    assert "batch_idx = next_boundary" in src
    assert "if batches_per_epoch and _sync_epoch_stop():" in src
    assert "batches_per_epoch and batch_iter is None" not in src
    # On an epoch-count-driven path the shortened epoch also shrinks the
    # optimizer-step budget so the run does not overtrain past num_train_epochs.
    # The budget is recomputed from the micro-batches that remain after the skip
    # (conceptual total minus the advanced cursor). Using
    # _epoch_stop_total_microbatches covers BOTH epoch layouts (the default cycled
    # single-pass and the torch_randperm materialized-all-epochs path); the old
    # flag-gated len(batches) form skipped the default path.
    assert "_remaining = _epoch_stop_total_microbatches - batch_idx" in src
    # The skip lands on an epoch boundary, so what remains is whole epochs and
    # each costs a ceil'd step count; flooring the micro-batches shortens the
    # epochs that were never truncated.
    assert "(_remaining // _epoch_flush_microbatches)" in src
    assert "_mlx_steps_per_epoch(" in src
    assert "_remaining // grad_accum" in src
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
    # It must be the FIRST executable statement, so no long work runs before the
    # stale stop is cleared. That statement is the generation guard.
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


def test_vlm_compile_patches_preserve_current_upstream_contracts(monkeypatch):
    import mlx.core as mx
    import unsloth_zoo.mlx.compile as mc
    upstream = lambda self, input_ids=None, pixel_values=None, **kwargs: "upstream"
    replacement = lambda self, input_ids=None, pixel_values=None, **kwargs: "replacement"
    adapted = mc._explicit_position_embedding_adapter(upstream, replacement)
    assert adapted(types.SimpleNamespace(training=True)) == "upstream"
    assert adapted(types.SimpleNamespace(training=False), position_ids=object()) == "upstream"
    assert adapted(types.SimpleNamespace(training=True), position_ids=object()) == "replacement"
    batched = types.SimpleNamespace(VisionModel=type("V", (), {"_forward_same_grid_batch": lambda self: None}))
    assert mc._paddleocr_vl_has_batched_vision(batched)
    assert mc._gemma3n_language_contract(len) is None
    assert mc._gemma3n_cache_offset([types.SimpleNamespace(offset=700, _idx=188)]) == 700
    assert mc._gemma3n_cache_offset([types.SimpleNamespace(offset=mx.array([650, 700]), _idx=188)]) == 700


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


def test_shared_family_installers_import_only_allowlisted_models(monkeypatch):
    import unsloth_zoo.mlx.compile as mc
    native = lambda *_args, **_kwargs: None
    qwen_arches = frozenset({"qwen2_vl", "qwen2_5_vl", "glm_ocr", "paddleocr_vl"})
    masked_arches = frozenset({"gemma3", "gemma4", "idefics2", "idefics3"})
    idefics_arches = frozenset({"idefics2", "idefics3"})
    assert (mc._QWEN_LIKE_MERGE_ARCHES, mc._MASKED_SCATTER_PATCH_ARCHES, mc._IDEFICS_SHARED_PATCH_ARCHES) == (qwen_arches, masked_arches, idefics_arches)
    methods = {
        "merge_input_ids_with_image_features": staticmethod(native),
        "_prepare_inputs_for_multimodal": native,
        "get_input_embeddings": native,
    }
    modules = {
        arch: types.SimpleNamespace(
            masked_scatter=native,
            Model=type(f"{arch}Model", (), methods),
        )
        for arch in qwen_arches | masked_arches
    }
    paddle_vision = types.SimpleNamespace(
        VisionModel=type("PaddleVision", (), {"_forward_same_grid_batch": native})
    )
    imported = []
    def import_module(name):
        imported.append(name)
        if name == "mlx_vlm.models.paddleocr_vl.vision":
            return paddle_vision
        parts = name.split(".")
        return modules.get(parts[-1]) if parts[-2:] == [parts[-1], parts[-1]] else None
    monkeypatch.setattr(mc, "_try_import_module", import_module)
    monkeypatch.setattr(mc, "build_compile_trait_reports", lambda: pytest.fail("runtime traits"))
    monkeypatch.setattr(mc, "_PATCHED_ARCHES", set())
    monkeypatch.setattr(mc, "_PATCH_BINDINGS", set())
    monkeypatch.setattr(mc, "_VERIFIED_TRAINING_ARCHES", set(mc._VERIFIED_TRAINING_ARCHES))
    mc._install_qwen_like_image_merge_patches()
    mc._install_masked_scatter_multimodal_patches()
    mc._install_idefics_family_compile_patches()
    assert not any(name.split(".")[-1] in {"lfm2_vl", "minicpmo", "phi4mm"} for name in imported)
    assert all(modules[arch].masked_scatter is mc._masked_scatter_no_numpy for arch in masked_arches)
    assert modules["paddleocr_vl"].Model.merge_input_ids_with_image_features is native
    assert all(modules[arch].Model.merge_input_ids_with_image_features is mc._merge_special_token_features_only for arch in qwen_arches - {"paddleocr_vl"})
    assert all(modules[arch].Model.get_input_embeddings is not native for arch in idefics_arches)


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
    # (the pending-window timing that runs every microstep); bound the inspected
    # region there so we only look at what runs between the on_optimizer_step
    # event and the rest of the same step. (num_input_tokens_seen is now
    # incremented ahead of on_optimizer_step, and the epoch update moved into the
    # optimizer-step path, so neither marks the branch close any more.)
    region_end = src.index("pending_time += time.perf_counter() - tic", opt_step)
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
    # BaseException, not Exception: an interrupt raised inside a callback
    # (KeyboardInterrupt from a Ctrl-C landing there, SystemExit) has to reach the
    # same consensus, or the peers strand in it. See
    # test_callback_interrupt_joins_the_ddp_failure_consensus.
    assert "except BaseException" in fire_body
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


def test_on_save_requires_a_resumable_checkpoint(monkeypatch):
    # Regression: the adapter write set checkpoint_written, then a failing
    # save_optimizer_state / save_trainer_state was swallowed with a printed
    # warning -- so on_save still fired and integrations (hub uploaders,
    # checkpoint trackers) recorded checkpoint-N as a resume point. It is not
    # one: save_trainer_state opens trainer_state.json before json.dump raises,
    # leaving the file truncated and unparseable.
    #
    # HF never reaches on_save here. _save_checkpoint calls
    # _save_optimizer_and_scheduler and state.save_to_json with no fallback, so
    # the failure propagates out of _maybe_log_save_evaluate before
    # callback_handler.on_save. Measured on transformers 5.14.1 and 4.57.6 with
    # an ExportableState callback returning a datetime: train() raises
    # TypeError("Object of type datetime is not JSON serializable") and the
    # on_save log stays empty. _prune_stale_checkpoints in this same function
    # is already gated on the completeness flag, for the same reason.
    import json

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class SaveSpy:
        """Shape of a stock integration reacting to on_save."""
        def __init__(self):
            self.advertised = []

        def on_save(self, args, state, control, **kwargs):
            self.advertised.append(f"checkpoint-{state.global_step}")
            return control

    class UnserializableState:
        """A callback exporting bookkeeping json.dump cannot encode.
        _export_callback_states is duck-typed on state(), so this reaches the
        stateful_callbacks payload."""
        def state(self):
            import datetime

            return {"started_at": datetime.datetime(2020, 1, 1)}

    def run(extra_callbacks):
        spy = SaveSpy()
        out_dir = tempfile.mkdtemp()
        args = MLXTrainingConfig(
            max_steps=2,
            gradient_accumulation_steps=1,
            logging_steps=10 ** 6,
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
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [],
            args=args,
            callbacks=[spy] + extra_callbacks,
        )
        trainer._batches = _make_shape_guard_text_plan((10,) * 2)
        trainer._callback_batches_per_epoch = lambda _batches: 2
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        trainer.train()
        return spy, os.path.join(out_dir, "checkpoint-2")

    # Control: nothing fails, so the checkpoint is resumable and on_save fires.
    healthy, healthy_dir = run([])
    assert healthy.advertised == ["checkpoint-2"]
    with open(os.path.join(healthy_dir, "trainer_state.json")) as fh:
        assert json.load(fh)["global_step"] == 2

    # The resume state could not be written, so nothing may advertise it.
    broken, broken_dir = run([UnserializableState()])
    assert broken.advertised == []
    # The adapters are still kept on disk: the swallow itself is deliberate,
    # only the advertisement was wrong.
    assert os.path.exists(os.path.join(broken_dir, "adapters.safetensors"))
    # ... and the directory really is unresumable, which is what on_save claimed.
    with open(os.path.join(broken_dir, "trainer_state.json")) as fh:
        with pytest.raises(json.JSONDecodeError):
            json.load(fh)


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
    # loop's accumulation cadence for grad_accum=2, batches_per_epoch=3. On a
    # max_steps run (no epoch flush) the microstep at the epoch boundary
    # (it % batches_per_epoch == 0) is a substep, so on_epoch_end would be
    # dropped without the substep-branch dispatch. An epoch-count run instead
    # forces the update there, mirroring HF's do_sync_step.
    grad_accum = 2
    batches_per_epoch = 3

    def boundary_is_update(epoch_flush_microbatches):
        accum_progress = 0
        seen = None
        for it in range(1, batches_per_epoch + 1):
            do_update = (accum_progress + 1 >= grad_accum)
            if (epoch_flush_microbatches
                    and it % epoch_flush_microbatches == 0):
                do_update = True
            if it % batches_per_epoch == 0:
                seen = do_update
            accum_progress = 0 if do_update else accum_progress + 1
        return seen

    assert boundary_is_update(None) is False
    assert boundary_is_update(batches_per_epoch) is True


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
    ends each epoch at its first update), so the skipped tail is smaller than
    grad_accum. Returns the terminal loop state and
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
    assert "_remaining = _epoch_stop_total_microbatches - batch_idx" in src
    assert "(_remaining // _epoch_flush_microbatches)" in src
    assert "total_steps - _skipped // grad_accum" not in src
    # The recompute is driven by the conceptual total micro-batches, which is set
    # for both epoch layouts (default cycled pass and torch_randperm), so it is not
    # gated behind _prepared_batches_include_epochs. The default (flag=False) path
    # multiplies the single materialized pass by the CEILED epoch count (every epoch
    # HF's outer loop would enter); truncating forfeited a fractional run's tail
    # epoch. The shrunk budget is clamped so it can only ever fall.
    assert "n_batches * math.ceil(float(args.num_train_epochs))" in src
    assert "n_batches * int(args.num_train_epochs)" not in src
    assert "total_steps = min(total_steps, _shrunk)" in src


def test_epoch_stop_skip_keeps_fractional_epoch():
    # Early-stopped epoch values stay fractional. When a callback
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
    # microstep % epoch_event_microbatches == 0, so a mid-epoch exit would drop
    # the event. Assert a post-loop dispatch closes the open epoch, guarded
    # against a double fire at a natural boundary, before on_train_end.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    tail = src[src.index("Close a truncated final epoch"):src.index("avg_loss = (")]
    assert "microstep % epoch_event_microbatches != 0" in tail
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

    An epoch boundary that lands on a non-update substep fires
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
    # An epoch boundary on a non-update substep (grad_accum=2, bpe=3)
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
    # Completed-step metrics survive partial stops. With
    # gradient_accumulation_steps>1, a stop from on_substep_end (or an
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
    # Pure-logic proof of that fix. grad_accum=2, a stop lands on the
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
    # Discard side: when a mid-epoch substep honors should_epoch_stop it
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

    # Single process: pad_source must be empty (cycle padding is multi-rank only)
    # and the partial tail stays short.
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

    # An mx-returning tokenizer must stage host_valued=False end to end.
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

    # Resume skip parity: first batch equals sync at the skip offset, once only.
    sync_all = _window_batches(rows, window=4)
    skipped = _window_stream(rows, window=4, prefetch_batches=2,
                             prefetch_skip_batches=2)
    first_after_skip = [row[0] for row in next(iter(skipped))[0].tolist()]
    assert first_after_skip == sync_all[2][0]

    # Eligibility is recorded synchronously and the orphan gate survives teardown.
    _T, trainer = _streaming_text_trainer(
        per_device_train_batch_size=2, max_seq_length=64,
        streaming_prefetch_batches=2)
    trainer.train_dataset = _CountingTextRows(rows)
    _b, _stream = trainer._prepare_data(is_vlm=False)
    assert trainer._mlx_prefetch_control.get("eligible") is True

    class FakeOrphan:
        orphaned = True
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


def test_max_eval_batches_rejects_non_integer_values():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    trainer = MLXTrainer(
        _MinimalTextModel(), _streaming_text_tokenizer(),
        _CountingTextRows(({"text": "10 1"},)),
        args=MLXTrainingConfig(streaming=True, max_steps=1, max_seq_length=8))
    for bad in (True, 1.9, 0, "2"):
        trainer.args.max_eval_batches = bad
        with pytest.raises(ValueError, match="max_eval_batches"):
            trainer._create_text_eval_batches(
                _CountingTextRows(({"text": "10 1"},)), 1, False, False)


def test_distributed_failure_reraises_interrupts_unwrapped():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    trainer = MLXTrainer(
        _MinimalTextModel(), _streaming_text_tokenizer(),
        _CountingTextRows(({"text": "10 1"},)),
        args=MLXTrainingConfig(streaming=True, max_steps=1, max_seq_length=8))
    for interrupt in (KeyboardInterrupt, SystemExit):
        with pytest.raises(interrupt):
            trainer._raise_distributed_failure_from_any(True, "save", interrupt())
        assert not trainer.stop_requested  # reuse must not inherit a stop
    with pytest.raises(RuntimeError, match="failed during save"):
        trainer._raise_distributed_failure_from_any(True, "save", ValueError("x"))
    assert trainer.stop_requested


def test_lazy_text_mixed_plain_and_prompt_completion_rejects_any_order():
    from unsloth_zoo.mlx.utils import _iterate_lazy_text_training_batches
    rows = [{"text": "10 1"}, {"prompt": "10", "completion": " 1 2"}]
    for ordering in (rows, rows[::-1]):
        with pytest.raises(ValueError, match="mixed|requires prompt"):
            list(_iterate_lazy_text_training_batches(
                iter(list(ordering)), _StreamingTextTokenizer(), 1, 8, repeat=False))


def test_eval_batch_totals_aborts_before_pull_when_stopped():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    tr = MLXTrainer(_MinimalTextModel(), _streaming_text_tokenizer(),
        _CountingTextRows(({"text": "10 1"},)),
        args=MLXTrainingConfig(streaming=True, max_steps=1, max_seq_length=8))
    class _Boom:
        def __iter__(self): raise AssertionError("pulled despite stop")
    tr.stop_requested = True
    assert int(tr._evaluate_batch_totals(_Boom(), loss_fn=None)[1]) == 0


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
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _MLX_CONFIG_OPTIONAL_COPY_FIELDS,
    )

    field_names = [f.name for f in dataclasses.fields(MLXTrainingConfig)]
    assert field_names[-2:] == ["logging_dir", "run_name"], field_names[-4:]
    # They sit after the lazy-streaming fields, which keep the slots they hold on
    # main, and the optional-copy tuple stays an exact suffix of the declaration
    # order so a legacy positional copy still binds to a contiguous prefix.
    assert field_names[-5:-2] == [
        "max_eval_batches",
        "streaming_text_length_window_batches",
        "streaming_prefetch_batches",
    ], field_names[-7:]
    assert field_names[-len(_MLX_CONFIG_OPTIONAL_COPY_FIELDS):] == list(
        _MLX_CONFIG_OPTIONAL_COPY_FIELDS
    )
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

    suppressed = trainer._suppress_torch_only_final_artifacts()
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
    trainer._restore_final_artifact_modes(suppressed)
    assert artifact_cb._log_model is _LogModel.END
    assert checkpoint_cb._log_model is _LogModel.CHECKPOINT

    # ...and the training loop actually wires it around the real dispatch,
    # with a restore that survives a callback raising.
    source = inspect.getsource(MLXTrainer._train_inner)
    assert "_suppress_torch_only_final_artifacts()" in source
    assert 'finally:\n            self._restore_final_artifact_modes(' in source


def test_dvclive_artifact_mode_suppressed_for_on_train_end():
    # transformers' DVCLiveCallback.on_train_end takes the same Torch-only path as
    # WandbCallback: with log_model=True (or HF_DVCLIVE_LOG_MODEL=TRUE) it builds a
    # Torch Trainer around args/model to save the final artifact, so a real MLX run
    # dies with AttributeError ('MLXTrainingConfig' object has no attribute
    # 'full_determinism' on 5.x, 'batch_eval_metrics' on 4.57.x) after training and
    # the final adapter save both finished. Two things are lost: the caller's
    # MLXTrainOutput, and the self.live.end() that trails the artifact block, which
    # leaves the tracked run unfinalized. log_model="all" must NOT be suppressed:
    # it is a per-checkpoint on_save artifact that never builds a Trainer.
    import inspect

    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _MLXCallbackHandler,
    )

    integration_utils = pytest.importorskip(
        "transformers.integrations.integration_utils"
    )

    # Pin the upstream shape this suppression is written against, so the test
    # tracks transformers instead of a hand-copied snapshot of it.
    upstream = inspect.getsource(integration_utils.DVCLiveCallback.on_train_end)
    assert "Trainer(" in upstream, upstream
    assert "if self._log_model is True:" in upstream, upstream
    # live.end() trails the artifact block, so raising inside it skips the
    # finalization too -- that is why the fix suppresses instead of catching.
    assert (
        upstream.index("if self._log_model is True:")
        < upstream.index("self.live.end()")
    ), upstream

    class DVCLiveCallback:  # the class NAME is what the bridge matches on
        def __init__(self, log_model):
            self._log_model = log_model
            self._initialized = True
            self.ended = False

        def on_train_end(self, args, state, control, **kwargs):
            # Verbatim control flow of the upstream method asserted above.
            if self._log_model is True:
                raise AttributeError(
                    "'MLXTrainingConfig' object has no attribute 'full_determinism'"
                )
            self.ended = True

    class CustomDVCLiveCallback(DVCLiveCallback):
        """Subclassing the integration callback is a common recipe, and it
        inherits the same on_train_end."""

    class OtherCallback:
        def __init__(self):
            self._log_model = True

    artifact_cb = DVCLiveCallback(True)
    subclass_cb = CustomDVCLiveCallback(True)
    all_cb = DVCLiveCallback("all")
    off_cb = DVCLiveCallback(None)
    other_cb = OtherCallback()

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(max_steps=1)
    trainer.callback_handler = _MLXCallbackHandler(
        [artifact_cb, subclass_cb, all_cb, off_cb, other_cb],
        model=object(),
        processing_class=None,
        optimizer=None,
        lr_scheduler=None,
    )

    suppressed = trainer._suppress_torch_only_final_artifacts()
    assert [cb for cb, _ in suppressed] == [artifact_cb, subclass_cb]
    assert artifact_cb._log_model is False
    assert subclass_cb._log_model is False
    # The per-checkpoint mode logs args.output_dir from on_save and never builds a
    # Trainer, so it keeps working on MLX and must survive untouched -- as must a
    # same-shaped callback from another library.
    assert all_cb._log_model == "all"
    assert off_cb._log_model is None
    assert other_cb._log_model is True

    # The on_train_end dispatch now completes, so live.end() is reached.
    trainer.callback_handler.call_event(
        "on_train_end", trainer.args, object(), object(),
    )
    assert artifact_cb.ended is True
    assert subclass_cb.ended is True
    assert all_cb.ended is True

    # The user's callbacks get their requested mode back afterwards.
    trainer._restore_final_artifact_modes(suppressed)
    assert artifact_cb._log_model is True
    assert subclass_cb._log_model is True

    # And the real transformers class is matched by the same MRO probe, whenever
    # the SDK the callback requires is actually installed.
    if integration_utils.is_dvclive_available():
        real_cb = integration_utils.DVCLiveCallback(log_model=True)
        real_cb._initialized = True
        trainer.callback_handler = _MLXCallbackHandler(
            [real_cb],
            model=object(),
            processing_class=None,
            optimizer=None,
            lr_scheduler=None,
        )
        real_suppressed = trainer._suppress_torch_only_final_artifacts()
        assert [cb for cb, _ in real_suppressed] == [real_cb]
        assert real_cb._log_model is False
        trainer._restore_final_artifact_modes(real_suppressed)
        assert real_cb._log_model is True


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
    # HF dispatches on_epoch_begin at the top of every epoch, including the
    # resumed partial one, and only then skips its already-trained batches. The
    # MLX loop fired begin at exact boundaries only, so a mid-epoch resume
    # delivered an on_epoch_end with no preceding begin.
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
    # An external controller owns stop_requested and may raise it between
    # construction and train(): Studio's cancel poller sets it as soon as the
    # trainer is registered, then reads it back after train() returns. An
    # unconditional clear at entry discarded that cancel and ran the whole job.
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
    # The stamp only separates runs if every train() closes its generation, a
    # raising one included, else a failed run's stop would block the next one.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer.train)
    bump = "self._run_generation = getattr(self, \"_run_generation\", 0) + 1"
    assert bump in src
    assert src.rindex("finally:") < src.index(bump)


def test_stateful_callbacks_exported_into_checkpoints(monkeypatch):
    # TrainerState declared stateful_callbacks but nothing wrote it, so
    # checkpoints carried no callback bookkeeping. HF populates it in
    # _save_checkpoint unconditionally (the opt-in flag gates only RESTORE).
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

    # Reusing that SAME trainer for a fresh train() must not keep serving run-1's
    # checkpoint bookkeeping: _resume_stateful_callbacks is cached and the restore
    # is unconditional. HF rebuilds TrainerState from the live callbacks each run
    # and only loads trainer_state.json when resume_from_checkpoint is given.
    seen = []

    class Recorder:
        def on_train_begin(self, args, state, control, **kwargs):
            seen.append(dict(state.stateful_callbacks or {}))
            return control

    resumed.callback_handler.callbacks.append(Recorder())
    resumed.train()
    assert seen == [{}], seen
    assert resumed.state.stateful_callbacks["Patience"]["attributes"]["counter"] == 6


def test_checkpoint_epoch_reaches_resumed_lifecycle_events(monkeypatch):
    # The checkpoint payload carried global_step but not state.epoch, so
    # _init_callback_state opened every resumed run at epoch=None: callbacks saw
    # no progress at on_train_begin, and a no-op resume (checkpoint already at
    # max_steps, loop body never runs) kept None through on_train_end, where
    # HF's stock NotebookProgressCallback does int(state.epoch). HF checkpoints
    # TrainerState wholesale, so epoch is restored alongside global_step.
    import json

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class EpochSpy:
        def __init__(self):
            self.begin = []
            self.end = []

        def on_train_begin(self, args, state, control, **kwargs):
            self.begin.append(state.epoch)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.end.append(state.epoch)
            return control

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    out_dir = tempfile.mkdtemp()

    def build(spy):
        args = MLXTrainingConfig(
            # 4 rows at batch size 1 = 4 micro-batches per epoch, so the
            # checkpoint at step 2 sits exactly half way through epoch 1.
            per_device_train_batch_size=1,
            max_steps=4,
            gradient_accumulation_steps=1,
            logging_steps=100,
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
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [{"text": f"row {i}"} for i in range(4)],
            args=args,
            callbacks=[spy],
        )
        trainer._prepare_data = lambda _is_vlm: (
            [make_batch(10) for _ in range(4)], None,
        )
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer

    build(EpochSpy()).train()
    with open(os.path.join(out_dir, "checkpoint-2", "trainer_state.json")) as fh:
        assert json.load(fh)["epoch"] == pytest.approx(0.5)

    # Mid-epoch resume: on_train_begin reports the checkpoint's progress, which
    # is what HF's restored TrainerState carries there.
    mid = EpochSpy()
    build(mid).train(
        resume_from_checkpoint=os.path.join(out_dir, "checkpoint-2"))
    assert mid.begin == [pytest.approx(0.5)], mid.begin

    # No-op resume: the loop body never runs, so nothing else can supply the
    # epoch and it must stay the checkpoint's through on_train_end.
    noop = EpochSpy()
    reused = build(noop)
    reused.train(resume_from_checkpoint=os.path.join(out_dir, "checkpoint-4"))
    assert noop.begin == [pytest.approx(1.0)], noop.begin
    assert noop.end == [pytest.approx(1.0)], noop.end

    # Reusing that trainer for a fresh train() must not serve the checkpoint's
    # epoch: HF only reads trainer_state.json when resume_from_checkpoint is given.
    reused.train()
    assert noop.begin[1] is None, noop.begin


def test_pre_optimizer_step_callback_fires_before_each_update(monkeypatch):
    # HF dispatches on_pre_optimizer_step immediately before optimizer.step().
    # The MLX loop only fired on_optimizer_step, so a callback relying on the
    # pre-update hook was silently inert for the whole run.
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
    # The callback epoch length must come from the rows batching retained, not
    # len(dataset): rows under two tokens are dropped and the tail partial batch
    # is not emitted, so ceil(len(ds) / batch) overshoots the real cycle and the
    # epoch events land off dataset boundaries.
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


def _periodic_log_steps(state):
    """Steps of the per-window training logs, with the final summary dropped.

    train() ends with one aggregate on_log (train_loss / train_runtime / ...),
    like HF's Trainer.log(metrics) immediately before on_train_end, so the tail
    entry is a run summary rather than a training-window log. Asserted here so
    callers keep pinning both the cadence and the summary.
    """
    entries = list(state.log_history)
    assert entries and "train_runtime" in entries[-1], entries[-1:]
    return [entry["step"] for entry in entries[:-1]]


def test_log_history_persisted_and_restored_across_resume(monkeypatch):
    # trainer_state.json carried the native loss history but not the
    # callback-visible TrainerState.log_history, which _init_callback_state reset
    # to [], so a resumed run reported only its post-resume entries. HF restores
    # the whole TrainerState, log_history included, from trainer_state.json.
    import json

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    out_dir = tempfile.mkdtemp()
    args = MLXTrainingConfig(
        max_steps=4,
        gradient_accumulation_steps=1,
        logging_steps=1,
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

    def build():
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [{"text": f"row {i}"} for i in range(4)],
            args=args,
        )
        trainer._prepare_data = lambda _is_vlm: ([make_batch(10) for _ in range(4)], None)
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer

    first = build()
    first.train()
    assert _periodic_log_steps(first.state) == [1, 2, 3, 4]

    with open(os.path.join(out_dir, "checkpoint-2", "trainer_state.json")) as fh:
        saved = json.load(fh)
    assert [entry["step"] for entry in saved["log_history"]] == [1, 2]

    resumed = build()
    resumed.train(resume_from_checkpoint=os.path.join(out_dir, "checkpoint-2"))
    assert _periodic_log_steps(resumed.state) == [1, 2, 3, 4]

    # Reusing that SAME trainer for a fresh run must start from an empty history
    # (HF only loads trainer_state.json when resume_from_checkpoint is given).
    resumed.train()
    assert _periodic_log_steps(resumed.state) == [1, 2, 3, 4]

    # A pre-fix checkpoint has no log_history key and must stay resumable.
    legacy_dir = os.path.join(out_dir, "checkpoint-2")
    with open(os.path.join(legacy_dir, "trainer_state.json")) as fh:
        legacy = json.load(fh)
    legacy.pop("log_history")
    with open(os.path.join(legacy_dir, "trainer_state.json"), "w") as fh:
        json.dump(legacy, fh)
    legacy_run = build()
    legacy_run.train(resume_from_checkpoint=legacy_dir)
    assert _periodic_log_steps(legacy_run.state) == [3, 4]


def test_fractional_step_intervals_resolve_against_total_steps(monkeypatch):
    # HF accepts these as a step count or a ratio in (0, 1) of the total steps,
    # expanded in TrainerState.compute_steps. int(ratio) turned 0.5 into 0, which
    # silently disabled logging and checkpointing while the synthesized strategy
    # still said "steps", and made DefaultFlowCallback take global_step % 0.
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import (
        MLXTrainer, MLXTrainingConfig, _resolve_interval_steps,
    )

    # Mirrors transformers' TrainerState.compute_steps: ceil(max_steps * ratio),
    # with plain counts passed through untouched.
    assert _resolve_interval_steps(0.1, 20) == 2
    assert _resolve_interval_steps(0.25, 20) == 5
    assert _resolve_interval_steps(0.5, 20) == 10
    assert _resolve_interval_steps(2, 20) == 2
    assert _resolve_interval_steps(0, 20) == 0
    assert _resolve_interval_steps(None, 20) == 0

    _patch_value_and_grad_with_aux(monkeypatch)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    class Flow:
        """Shaped like transformers' DefaultFlowCallback."""
        def __init__(self):
            self.logged_at = []

        def on_step_end(self, args, state, control, **kwargs):
            if args.logging_strategy == "steps" and state.global_step % state.logging_steps == 0:
                self.logged_at.append(state.global_step)
            return control

    out_dir = tempfile.mkdtemp()
    flow = Flow()
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [{"text": f"row {i}"} for i in range(4)],
        args=MLXTrainingConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            logging_steps=0.5,
            save_steps=0.5,
            eval_steps=0.25,
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            output_dir=out_dir,
        ),
        callbacks=[flow],
    )
    trainer._prepare_data = lambda _is_vlm: ([make_batch(10) for _ in range(4)], None)
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None
    trainer.train()

    # 0.5 of 4 steps is every 2 steps, not "never" (and not a modulo by zero).
    assert trainer.state.logging_steps == 2
    assert trainer.state.save_steps == 2
    assert trainer.state.eval_steps == 1
    assert flow.logged_at == [2, 4]
    assert _periodic_log_steps(trainer.state) == [2, 4]
    checkpoints = sorted(
        name for name in os.listdir(out_dir) if name.startswith("checkpoint-")
    )
    assert checkpoints == ["checkpoint-2", "checkpoint-4"]


class _FakeClock:
    """Wall clock that only moves when the loop consumes a micro-batch.

    perf_counter() is a pure read, so `train_time += perf_counter() - tic`
    charges exactly COST[i] seconds to micro-batch i, making the reported
    tokens/s and step times exact, checkable numbers.
    """

    def __init__(self, real):
        self._real = real
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_forced_epoch_log_splits_pending_wall_clock_from_committed(monkeypatch):
    # Regression: the committed/pending split covered loss/tokens/steps but not
    # train_time. With the epoch boundary on a NON-update microstep (grad_accum=2,
    # batches-per-epoch=3), the forced log reported COMMITTED tokens yet was
    # charged the pending micro-batch's duration, understating its tokens/s and
    # then hiding that duration from the window that owned it.
    import tempfile
    import time as _time

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    # micro-batch: 1 2 3 4 5 6   (updates at 2, 4 -> no; grad_accum=2 -> 2, 4, 6)
    # epoch boundaries at 3 and 6; microstep 3 is a substep, and it is expensive.
    costs = [1.0, 1.0, 10.0, 1.0, 1.0, 1.0]
    clock = _FakeClock(_time)
    monkeypatch.setattr(trainer_mod, "time", clock)

    # The per-micro-batch input-token count is this test's clock hook, and it only
    # runs when include_num_input_tokens_seen is enabled, so opt in explicitly.
    consumed = {"i": 0}
    real_count = trainer_mod._mlx_batch_input_token_count

    def _timed_count(batch_data, *args, **kwargs):
        index = consumed["i"]
        consumed["i"] = index + 1
        clock.now += costs[index] if index < len(costs) else 1.0
        return real_count(batch_data, *args, **kwargs)

    monkeypatch.setattr(trainer_mod, "_mlx_batch_input_token_count", _timed_count)

    class ForceLogAtEpochEnd:
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_log = True      # HF logging_strategy="epoch"
            return control

    args = MLXTrainingConfig(
        max_steps=3,
        gradient_accumulation_steps=2,
        logging_steps=100,               # only the forced + final logs fire
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=tempfile.mkdtemp(),
    )
    args.include_num_input_tokens_seen = "all"
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [],
        args=args,
        callbacks=[ForceLogAtEpochEnd()],
    )
    trainer._batches = _make_shape_guard_text_plan((10,) * 6)
    trainer._callback_batches_per_epoch = lambda _batches: 3
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None

    trainer.train()

    # Two logs: the forced epoch-end log at microstep 3 (committed = micro 1+2)
    # and the final-step log (committed = micro 3+4 and 5+6).
    assert len(trainer._tokens_per_second_history) == 2
    tokens = trainer._global_token_count_history
    assert len(tokens) == 2

    # Window 1 owns micro-batches 1 and 2 -> 1.0 + 1.0 = 2.0 s.
    # Window 2 owns micro-batches 3..6 -> 10.0 + 1.0 + 1.0 + 1.0 = 13.0 s.
    assert trainer._tokens_per_second_history[0] == pytest.approx(
        tokens[0] / 2.0, rel=1e-9,
    )
    assert trainer._tokens_per_second_history[1] == pytest.approx(
        tokens[1] / 13.0, rel=1e-9,
    )
    # _step_times is the window's wall clock over its micro-batch count.
    assert trainer._step_times == pytest.approx([2.0 / 2, 13.0 / 4], rel=1e-9)
    # Guard the guard: the unsplit clock charged window 1 all 12.0 s (micro 3
    # included) and left window 2 with only 3.0 s.
    assert trainer._tokens_per_second_history[0] != pytest.approx(
        tokens[0] / 12.0, rel=1e-9,
    )
    assert trainer._tokens_per_second_history[1] != pytest.approx(
        tokens[1] / 3.0, rel=1e-9,
    )


def test_pending_wall_clock_folds_only_on_an_applied_update():
    # Source guard for the split above: the per-micro-batch duration lands in the
    # PENDING clock, folds into COMMITTED in the same do_update branch as
    # pending_losses, and is discarded with the pending tokens.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "pending_time += time.perf_counter() - tic" in src
    assert "train_time += time.perf_counter() - tic" not in src
    fold = src.index("pending_time += time.perf_counter() - tic")
    fold_block = src[fold:fold + 200]
    assert "if do_update:" in fold_block
    assert "train_time += pending_time" in fold_block
    assert "pending_time = 0" in fold_block
    # The forced-log helper still resets only the committed clock.
    log_body = src[src.index("def _run_training_log("):src.index("def _run_eval(")]
    assert "train_time = 0" in log_body
    assert "pending_time" not in log_body
    # The mid-epoch abandon drops the pending clock with the pending tokens.
    discard = src.index("pending_losses = 0", src.index("grad_accum_state = None      #"))
    assert "pending_time = 0" in src[discard:discard + 500]


def test_eval_request_from_eval_log_is_cleared_before_on_evaluate(monkeypatch):
    # Regression: _run_eval cleared control.should_evaluate BEFORE dispatching the
    # eval-metrics on_log, so a callback requesting evaluation from on_log had its
    # fresh should_evaluate=True survive on_evaluate. HF clears it inside
    # CallbackHandler.on_evaluate, i.e. after evaluate() logged its metrics. With
    # the early reset, _maybe_callback_epoch_end saw the stale flag and ran a
    # SECOND full evaluation at the same global_step.
    import inspect
    import tempfile

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    src = inspect.getsource(MLXTrainer._train_inner)
    eval_body = src[src.index("def _run_eval("):src.index("def _run_best_tracking(")]
    reset = eval_body.index("self.control.should_evaluate = False", eval_body.index("metrics = self._last_eval_metrics"))
    assert eval_body.index('_fire("on_log", logs=dict(metrics))') < reset
    assert reset < eval_body.index('_fire("on_evaluate"')

    class EvalFromEvalLog:
        def __init__(self):
            self.evaluates = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(key.startswith("eval_") for key in (logs or {})):
                control.should_evaluate = True
            return control

        def on_evaluate(self, args, state, control, **kwargs):
            self.evaluates.append((state.global_step, control.should_evaluate))
            return control

    spy = EvalFromEvalLog()
    args = MLXTrainingConfig(
        max_steps=4,
        gradient_accumulation_steps=1,
        logging_steps=100,
        eval_steps=4,                    # eval on the last step == epoch boundary
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
    trainer._batches = _make_shape_guard_text_plan((10,) * 4)
    trainer._callback_batches_per_epoch = lambda _batches: 4
    trainer.eval_dataset = [{"input_ids": [1, 2, 3, 4]}]
    trainer._eval_batches_labeled = ["batch-0"]
    # The eval maths is exercised elsewhere; here only the control-flag lifecycle
    # around the on_log/on_evaluate dispatch matters.
    evaluations = []

    def _fake_evaluate(batches, loss_fn, is_vlm=False):
        evaluations.append(len(batches))
        trainer._last_eval_metrics = {"eval_loss": 1.25, "eval_perplexity": 3.5}
        return 1.25, 3.5

    trainer._evaluate = _fake_evaluate
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None

    trainer.train()

    # Exactly one evaluation at the boundary step, and HF's flag state inside
    # on_evaluate (already cleared), not the stale True.
    assert spy.evaluates == [(4, False)]
    assert evaluations == [1], "the eval ran once, not twice at the same step"
    eval_records = [
        record for record in trainer.state.log_history
        if any(key.startswith("eval_") for key in record)
    ]
    assert [record["step"] for record in eval_records] == [4]


def test_max_steps_run_reports_hf_epoch_total_to_callbacks(monkeypatch):
    # Regression: MLXTrainingConfig defaults num_train_epochs to -1, so a max_steps
    # run reported state.num_train_epochs = 0 even while the finite batch plan
    # dispatched real callback epochs. HF instead derives it as
    # ceil(max_steps / num_update_steps_per_epoch), so callbacks were reading
    # metadata contradicting their own epoch events, and dividing progress by zero.
    import tempfile

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class EpochTotalSpy:
        def __init__(self):
            self.at_train_begin = None
            self.epoch_ends = []
            self.progress = []

        def on_train_begin(self, args, state, control, **kwargs):
            self.at_train_begin = state.num_train_epochs
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            self.epoch_ends.append(state.epoch)
            # What a progress-normalizing callback does; ZeroDivisionError at 0.
            self.progress.append(state.epoch / state.num_train_epochs)
            return control

    spy = EpochTotalSpy()
    args = MLXTrainingConfig(
        max_steps=6,
        gradient_accumulation_steps=1,
        logging_steps=100,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=tempfile.mkdtemp(),
    )
    assert args.num_train_epochs == -1, "the default max_steps config"
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [],
        args=args,
        callbacks=[spy],
    )
    trainer._batches = _make_shape_guard_text_plan((10,) * 6)
    trainer._callback_batches_per_epoch = lambda _batches: 4
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None

    trainer.train()

    # 4 micro-batches per epoch at grad_accum=1 -> 4 updates per epoch, and
    # ceil(6 / 4) = 2 epochs, exactly HF's arithmetic.
    assert spy.at_train_begin == 2
    assert trainer.state.num_train_epochs == 2
    # The metadata matches the epoch events actually dispatched.
    assert spy.epoch_ends and max(spy.epoch_ends) <= spy.at_train_begin
    assert spy.progress == [pytest.approx(value / 2) for value in spy.epoch_ends]


def test_callback_num_train_epochs_mirrors_hf_arithmetic():
    # Unit coverage of the paths one run cannot reach: epoch-count runs pass
    # through untouched, and a streaming / length-less plan stays at 0.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(
        max_steps=50, per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
    )
    trainer._distributed_world_size = 1
    trainer._mlx_train_dataset_for_batches = list(range(8))   # one pass = 4 micro
    trainer.train_dataset = trainer._mlx_train_dataset_for_batches
    trainer._prepared_batches_include_epochs = False

    # grad_accum=1 -> 4 updates/epoch -> ceil(50 / 4) = 13.
    assert trainer._callback_num_train_epochs(50, list(range(100))) == 13
    # grad_accum=3 -> ceil(4 / 3) = 2 updates/epoch -> ceil(50 / 2) = 25.
    trainer.args.gradient_accumulation_steps = 3
    assert trainer._callback_num_train_epochs(50, list(range(100))) == 25
    # A run shorter than one epoch still reports one epoch, like HF's
    # `max_steps // nupe + int(max_steps % nupe > 0)` for max_steps < nupe.
    trainer.args.gradient_accumulation_steps = 1
    assert trainer._callback_num_train_epochs(2, list(range(100))) == 1
    # Streaming: no finite plan, no epoch events, field left alone.
    assert trainer._callback_num_train_epochs(50, None) == 0

    # Epoch-count runs are unchanged.
    epochs = MLXTrainer.__new__(MLXTrainer)
    epochs.args = MLXTrainingConfig(num_train_epochs=3, max_steps=-1)
    epochs._prepared_batches_include_epochs = True
    assert epochs._callback_num_train_epochs(0, list(range(12))) == 3


def test_callback_num_train_epochs_honors_max_steps_over_configured_epochs():
    # HF derives the total from max_steps whenever it is set, ignoring
    # num_train_epochs, which a real TrainingArguments leaves at 3.0.
    # MLXTrainingConfig.max_steps defaults to 60, so this is the common path.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(
        max_steps=60, num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
    )
    trainer._distributed_world_size = 1
    trainer._mlx_train_dataset_for_batches = list(range(16))  # one pass = 8 micro
    trainer.train_dataset = trainer._mlx_train_dataset_for_batches
    trainer._prepared_batches_include_epochs = False

    # ceil(8 / 4) = 2 updates per epoch -> ceil(60 / 2) = 30, not the configured 3.
    assert trainer._callback_num_train_epochs(60, list(range(240))) == 30
    # A max_steps run cut short of one pass reports the truncated total, not 3.
    trainer.args.gradient_accumulation_steps = 1
    assert trainer._callback_num_train_epochs(3, list(range(3))) == 1
    # Streaming has no boundaries to derive from: keep the configured value
    # rather than dropping back to the ZeroDivisionError-prone 0.
    assert trainer._callback_num_train_epochs(60, None) == 3


def test_fractional_num_train_epochs_reports_a_ceiled_callback_total(monkeypatch):
    # num_train_epochs is a float in TrainingArguments/SFTConfig and a
    # fractional value is supported. HF reports the CEILED count in
    # TrainerState -- `num_train_epochs = math.ceil(args.num_train_epochs)`
    # (transformers set_initial_training_values, identical in 5.14.1 and
    # 4.57.6) -- which is the same rounding its step budget uses
    # (`max_steps = ceil(num_train_epochs * num_update_steps_per_epoch)`), and
    # the MLX budget already matched. The callback total truncated instead, so
    # a 1.5-epoch run advertised 1 epoch while state.epoch climbed to 1.5:
    # every callback normalizing progress by the total read 150 percent, and
    # the two numbers HF keeps consistent contradicted each other.
    _patch_value_and_grad_with_aux(monkeypatch)

    class EpochTotalSpy:
        def __init__(self):
            self.at_train_begin = None
            self.progress = []

        def on_train_begin(self, args, state, control, **kwargs):
            self.at_train_begin = state.num_train_epochs
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            self.progress.append(state.epoch / state.num_train_epochs)
            return control

    spy = EpochTotalSpy()
    trainer, _batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=4, grad_accum=2, epochs=1.5,
        callbacks=[spy],
    )
    trainer.train()

    # Real transformers on this shape (4 rows at batch 1, grad_accum 2,
    # num_train_epochs=1.5): max_steps = ceil(1.5 * 2) = 3, num_train_epochs =
    # ceil(1.5) = 2, final state.epoch = 1.5. Measured on 5.14.1 and 4.57.6.
    assert trainer.state.max_steps == 3
    assert spy.at_train_begin == 2
    assert trainer.state.num_train_epochs == 2
    assert trainer.state.epoch == pytest.approx(1.5)
    # Progress never exceeds 100 percent now that the total covers the epochs
    # the run actually reports.
    assert spy.progress == [pytest.approx(0.5), pytest.approx(0.75)]
    assert max(spy.progress) <= 1.0


def test_callback_num_train_epochs_ceils_like_hf():
    # Unit coverage of the rounding itself, including the values one run cannot
    # reach. transformers uses math.ceil for the epoch-based branch, so any
    # fraction rounds UP and whole counts are untouched.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    def total(epochs):
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.args = MLXTrainingConfig(num_train_epochs=epochs, max_steps=-1)
        trainer._prepared_batches_include_epochs = True
        return trainer._callback_num_train_epochs(0, list(range(12)))

    assert total(1.5) == 2
    assert total(2.5) == 3
    assert total(0.5) == 1
    assert total(0.01) == 1
    assert total(2.000001) == 3
    # Whole counts, int or float, are byte-identical to the old behaviour.
    for epochs in (1, 2, 3, 10):
        assert total(epochs) == epochs
        assert total(float(epochs)) == epochs
    # The "use max_steps instead" sentinel and 0 stay at 0, so the max_steps
    # derivation below them still runs.
    assert total(-1) == 0
    assert total(0) == 0


def test_callback_best_metric_persisted_across_resume_without_native_tracking(monkeypatch):
    # The callback-visible watermark (TrainerState.best_metric) advances on every
    # eval whenever metric_for_best_model is set, but the NATIVE best fields are
    # only written by _run_best_tracking, gated on load_best_model_at_end or
    # early_stopping_patience. With both off the checkpoint persisted only the null
    # native value, so on resume EarlyStoppingCallback saw best_metric=None and
    # called the first post-resume eval a new best. HF has no such split: it
    # checkpoints and reloads the whole TrainerState.
    import json

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    class BestSpy:
        """Records the watermark callbacks see, before it is updated for this eval."""
        def __init__(self):
            self.seen = []

        def on_evaluate(self, args, state, control, **kwargs):
            self.seen.append((state.global_step, state.best_metric,
                              state.best_global_step))
            return control

    out_dir = tempfile.mkdtemp()

    def build(max_steps, eval_losses):
        spy = BestSpy()
        args = MLXTrainingConfig(
            max_steps=max_steps,
            gradient_accumulation_steps=1,
            logging_steps=100,
            eval_steps=2,
            save_steps=2,
            # Native best tracking OFF: only the callback-visible watermark moves.
            load_best_model_at_end=False,
            early_stopping_patience=0,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
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
        trainer._prepare_data = lambda _is_vlm: (
            [make_batch(10) for _ in range(4)], None,
        )
        trainer.eval_dataset = [{"input_ids": [1, 2, 3, 4]}]
        trainer._eval_batches_labeled = ["batch-0"]
        pending = list(eval_losses)

        def _fake_evaluate(batches, loss_fn, is_vlm=False):
            value = pending.pop(0)
            trainer._last_eval_metrics = {"eval_loss": value}
            return value, 2.0

        trainer._evaluate = _fake_evaluate
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer, spy

    first, first_spy = build(2, [0.9])
    first.train()
    # Native tracking never ran; only the callback-visible watermark advanced.
    assert first._best_metric is None and first._best_step is None
    assert first.state.best_metric == 0.9
    assert first.state.best_global_step == 2

    ckpt = os.path.join(out_dir, "checkpoint-2")
    with open(os.path.join(ckpt, "trainer_state.json")) as fh:
        saved = json.load(fh)
    assert saved["best_metric"] is None, "native tracking is off"
    assert saved["callback_best_metric"] == pytest.approx(0.9)
    assert saved["callback_best_step"] == 2

    # Resume: the first post-resume eval is WORSE (1.5 > 0.9). Callbacks must see
    # the restored 0.9 watermark, and it must survive the worse metric.
    resumed, resumed_spy = build(4, [1.5])
    resumed.train(resume_from_checkpoint=ckpt)
    assert resumed_spy.seen == [(4, pytest.approx(0.9), 2)]
    assert resumed.state.best_metric == pytest.approx(0.9)
    assert resumed.state.best_global_step == 2

    # Reusing that same trainer for a FRESH run must not carry the watermark over
    # (HF only loads trainer_state.json when resume_from_checkpoint is given).
    fresh_spy = BestSpy()
    resumed.callback_handler.callbacks = [
        cb for cb in resumed.callback_handler.callbacks
        if not isinstance(cb, BestSpy)
    ] + [fresh_spy]
    fresh_pending = [0.7]

    def _fresh_evaluate(batches, loss_fn, is_vlm=False):
        value = fresh_pending.pop(0)
        resumed._last_eval_metrics = {"eval_loss": value}
        return value, 2.0

    resumed._evaluate = _fresh_evaluate
    resumed.args.max_steps = 2
    resumed.train()
    assert fresh_spy.seen == [(2, None, None)], "no phantom best on a fresh run"

    # A pre-fix checkpoint has no callback_best_* keys and must stay resumable,
    # falling back to the native value exactly as before.
    saved.pop("callback_best_metric")
    saved.pop("callback_best_step")
    with open(os.path.join(ckpt, "trainer_state.json"), "w") as fh:
        json.dump(saved, fh)
    legacy, legacy_spy = build(4, [1.5])
    legacy.train(resume_from_checkpoint=ckpt)
    assert legacy_spy.seen == [(4, None, None)]


@pytest.mark.parametrize("best_weights_present", [False, True])
def test_missing_best_weights_clears_callback_best_state(monkeypatch, best_weights_present):
    # Resuming a checkpoint copied into an output_dir without best/ restarts the
    # NATIVE best tracking, but the callback-visible watermark was still restored
    # from the checkpoint. EarlyStoppingCallback then measured the first eval
    # against weights that no longer exist (patience 1 stops immediately) while
    # _run_best_tracking called that same eval a new best and overwrote the
    # watermark with that worse value. HF keeps one watermark for both, and its
    # _determine_best_metric only ever moves it in the improving direction.
    import json
    import shutil

    import mlx.core as mx
    from transformers import EarlyStoppingCallback

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    class BestSpy:
        def __init__(self):
            self.seen = []

        def on_evaluate(self, args, state, control, **kwargs):
            self.seen.append((state.global_step, state.best_metric,
                              state.best_global_step, state.best_model_checkpoint))
            return control

    def build(out_dir, max_steps, eval_losses, callbacks):
        args = MLXTrainingConfig(
            max_steps=max_steps,
            gradient_accumulation_steps=1,
            logging_steps=1000,
            eval_steps=2,
            save_steps=2,
            load_best_model_at_end=True,
            early_stopping_patience=0,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
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
            [{"text": f"row {i}"} for i in range(8)],
            args=args,
            callbacks=list(callbacks),
        )
        trainer._prepare_data = lambda _is_vlm: (
            [make_batch(10) for _ in range(max_steps + 2)], None,
        )
        trainer.eval_dataset = [{"input_ids": [1, 2, 3, 4]}]
        trainer._eval_batches_labeled = ["batch-0"]
        pending = list(eval_losses)

        def _fake_evaluate(batches, loss_fn, is_vlm=False):
            value = pending.pop(0)
            trainer._last_eval_metrics = {"eval_loss": value}
            return value, 2.0

        trainer._evaluate = _fake_evaluate
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer

    dir_a = tempfile.mkdtemp()
    dir_b = tempfile.mkdtemp()
    first = build(dir_a, 2, [0.9], [])
    first.train()
    assert (first._best_metric, first._best_step) == (0.9, 2)
    assert os.path.isdir(os.path.join(dir_a, "best"))

    ckpt = os.path.join(dir_a, "checkpoint-2")
    with open(os.path.join(ckpt, "trainer_state.json")) as fh:
        saved = json.load(fh)
    assert saved["callback_best_metric"] == pytest.approx(0.9)

    # Copy the checkpoint alone into a fresh output_dir; best/ follows only in
    # the control leg.
    shutil.copytree(ckpt, os.path.join(dir_b, "checkpoint-2"))
    if best_weights_present:
        shutil.copytree(os.path.join(dir_a, "best"), os.path.join(dir_b, "best"))

    spy = BestSpy()
    resumed = build(dir_b, 8, [1.5, 1.4, 1.3],
                    [spy, EarlyStoppingCallback(early_stopping_patience=1)])
    resumed.train(resume_from_checkpoint=os.path.join(dir_b, "checkpoint-2"))

    if best_weights_present:
        # Weights are there, so the restored watermark is real: the first worse
        # eval exhausts patience and the best stays where it was.
        assert spy.seen == [(4, pytest.approx(0.9), 2, f"{dir_b}/best")]
        assert resumed.state.global_step == 4
        assert (resumed._best_metric, resumed._best_step) == (0.9, 2)
        assert resumed.state.best_metric == pytest.approx(0.9)
    else:
        # Tracking restarted, so callbacks must see no watermark and the run must
        # not stop against a model it cannot restore.
        assert spy.seen[0] == (4, None, None, None)
        assert resumed.state.global_step == 8
        assert [step for step, *_ in spy.seen] == [4, 6, 8]
        # Native and callback best stay in lockstep, and the watermark only improves.
        assert (resumed._best_metric, resumed._best_step) == (1.3, 8)
        assert resumed.state.best_metric == pytest.approx(1.3)
        assert resumed.state.best_global_step == 8
        assert [m for _, m, *_ in spy.seen] == [None, pytest.approx(1.5), pytest.approx(1.4)]


def test_externally_cancelled_eval_is_not_dispatched_to_callbacks(monkeypatch):
    # Regression: stop_requested is an externally owned property -- a controller
    # (the Studio cancel button) may set it at any time -- and
    # _evaluate_batch_totals deliberately skips every remaining eval batch while
    # it is set. With zero batches scored _evaluate returns eval_loss 0.0, yet
    # _run_eval dispatched that phantom result anyway: EarlyStoppingCallback read
    # 0.0 as an improvement and reset its patience counter, and
    # _update_callback_best_metric latched a watermark no real evaluation can
    # ever beat. _run_checkpoint then persisted both into checkpoint-N, so the
    # corruption survived into the resumed run.
    #
    # A CALLBACK stop must keep its real evaluation: it is deliberately left in
    # control.should_training_stop until after this step's log/eval/save (HF's
    # _maybe_log_save_evaluate runs _determine_best_metric and the checkpoint
    # before the loop honors should_training_stop), so it never reaches this
    # gate. The second half of this test pins that.
    import inspect
    import json

    import mlx.core as mx
    import mlx.nn as nn
    from transformers import EarlyStoppingCallback

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def eval_capable_lm():
        """_tiny_lm_for_loop_tests' train() takes no mode, so nn.Module.eval()
        (which calls train(False)) raises. The real _evaluate needs both."""
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(128, 4)
                self.proj = nn.Linear(4, 128, bias=False)
                self._config = {"model_type": "tiny"}

            def __call__(self, input_ids):
                return self.proj(self.embed(input_ids))

            def train(self, mode=True):
                return self

            def eval(self):
                return self

            @property
            def state(self):
                return []

        return TinyLM()

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    class EvalSpy:
        def __init__(self):
            self.evaluated = []
            self.logged = []

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self.evaluated.append((state.global_step, (metrics or {}).get("eval_loss")))
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "eval_loss" in logs:
                self.logged.append((state.global_step, logs["eval_loss"]))
            return control

    class StopAtLastEval:
        """external=True stands in for a controller cancelling the run;
        external=False is an ordinary callback stop."""
        def __init__(self, step, external):
            self.step = step
            self.external = external
            self.trainer = None

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == self.step:
                if self.external:
                    self.trainer.stop_requested = True
                else:
                    control.should_training_stop = True
            return control

    def run(external):
        out_dir = tempfile.mkdtemp()
        spy = EvalSpy()
        early = EarlyStoppingCallback(early_stopping_patience=5)
        stopper = StopAtLastEval(6, external)
        args = MLXTrainingConfig(
            max_steps=6,
            gradient_accumulation_steps=1,
            logging_steps=10 ** 6,
            eval_steps=2,
            save_steps=6,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Native tracking off: this pins the callback-visible state only.
            load_best_model_at_end=False,
            early_stopping_patience=0,
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            output_dir=out_dir,
        )
        mx.random.seed(1234)
        trainer = MLXTrainer(
            eval_capable_lm(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [{"text": f"row {i}"} for i in range(6)],
            eval_dataset=[{"input_ids": [1, 2, 3, 4]}],
            args=args,
            callbacks=[spy, early, stopper],
        )
        stopper.trainer = trainer
        trainer._prepare_data = lambda _is_vlm: (
            [make_batch(10) for _ in range(6)], None,
        )
        # Real eval batches: the abort under test lives in _evaluate_batch_totals,
        # so _evaluate must not be stubbed out.
        trainer._eval_batches_labeled = [make_batch(10)]
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        output = trainer.train()
        with open(os.path.join(out_dir, "checkpoint-6", "trainer_state.json")) as fh:
            saved = json.load(fh)
        return trainer, spy, early, output, saved

    cancelled, spy, early, output, saved = run(external=True)
    # Steps 2 and 4 scored real batches; the frozen optimizer makes them equal.
    real_loss = spy.evaluated[0][1]
    assert real_loss > 0.0
    # Step 6 evaluated nothing, so it must not reach on_log or on_evaluate.
    assert spy.evaluated == [(2, real_loss), (4, real_loss)]
    assert spy.logged == [(2, real_loss), (4, real_loss)]
    # The watermark stays on the last real evaluation instead of a phantom 0.0.
    assert cancelled.state.best_metric == pytest.approx(real_loss)
    assert cancelled.state.best_global_step == 2
    # Step 4 failed to improve on step 2, and no phantom improvement reset it.
    assert early.early_stopping_patience_counter == 1
    assert output["eval_metrics"]["eval_loss"] == pytest.approx(real_loss)
    # None of it reaches the checkpoint a resume would restore from.
    assert saved["callback_best_metric"] == pytest.approx(real_loss)
    assert saved["callback_best_step"] == 2
    assert saved["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"] == {
        "early_stopping_patience_counter": 1,
    }

    # A callback stop is untouched: step 6 still runs and dispatches a real eval.
    _, cb_spy, cb_early, cb_output, cb_saved = run(external=False)
    assert cb_spy.evaluated == [
        (2, real_loss), (4, real_loss), (6, real_loss),
    ]
    assert cb_early.early_stopping_patience_counter == 2
    assert cb_output["eval_metrics"]["eval_loss"] == pytest.approx(real_loss)
    assert cb_saved["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"] == {
        "early_stopping_patience_counter": 2,
    }

    # The rank agreement is invisible at world size 1, so pin it at the source:
    # the suppression is decided by the OR-reduce, or a cancel landing on one
    # rank after _evaluate's last eval-status collective returns that rank alone
    # and strands its peers inside _fire.
    src = inspect.getsource(MLXTrainer._train_inner)
    eval_body = src[src.index("def _run_eval("):src.index("def _run_best_tracking(")]
    guard = eval_body.index("if self._distributed_should_stop():")
    assert guard < eval_body.index('_fire("on_log", logs=dict(metrics))')
    assert guard < eval_body.index("self._update_callback_best_metric(metrics)")


def test_checkpoint_includes_committed_unlogged_loss_totals(monkeypatch):
    # train_loss_token_sum/_total are written only by _run_training_log, so a
    # checkpoint taken on a step whose applied updates are not logged yet (a save
    # cadence out of phase with the log cadence) persisted totals covering fewer
    # steps than its own global_step, and the resumed run's final train_loss lost
    # them. The payload must fold the committed window in without mutating the
    # live accumulators, or the later log of the same window double counts.
    import json

    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    class StopAfter:
        """Stands in for a crash/cancel right after the step-2 checkpoint."""
        def __init__(self, step):
            self.step = step

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= self.step:
                control.should_training_stop = True
            return control

    out_dir = tempfile.mkdtemp()

    def build(stop_after=None):
        args = MLXTrainingConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            # No log cadence: the only forced log is the final step, so the
            # step-2 checkpoint lands with steps 1-2 committed but unlogged.
            logging_steps=100,
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
        # Same init in every run so the frozen-weight losses are comparable
        # across the reference run and the interrupted/resumed pair.
        mx.random.seed(1234)
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [{"text": f"row {i}"} for i in range(4)],
            args=args,
            callbacks=[StopAfter(stop_after)] if stop_after else [],
        )
        trainer._prepare_data = lambda _is_vlm: (
            [make_batch(10) for _ in range(4)], None,
        )
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        return trainer

    # Reference: one uninterrupted 4-step run. The frozen optimizer keeps every
    # per-batch loss identical, so its totals are exactly twice a 2-step run's.
    whole = build()
    whole.train()
    assert whole._train_loss_token_total > 0

    with open(os.path.join(out_dir, "checkpoint-2", "trainer_state.json")) as fh:
        saved = json.load(fh)
    assert saved["global_step"] == 2
    assert saved["log_history"] == [], "the step-2 window was never logged"
    # The checkpoint covers the two applied steps, not zero of them ...
    assert saved["train_loss_token_total"] == whole._train_loss_token_total // 2
    assert saved["train_loss_token_sum"] == pytest.approx(
        whole._train_loss_token_sum / 2
    )
    # ... and folding it into the payload must not double count in the live run:
    # the final totals still cover exactly the 4 applied steps.
    assert whole._train_loss_token_total == 2 * saved["train_loss_token_total"]

    # End to end: stop right after the step-2 checkpoint, resume, and the final
    # token-weighted train loss matches the uninterrupted run's.
    interrupted = build(stop_after=2)
    interrupted.train()
    resumed = build()
    resumed.train(resume_from_checkpoint=os.path.join(out_dir, "checkpoint-2"))
    assert resumed._train_loss_token_total == whole._train_loss_token_total
    assert resumed._train_loss_token_sum == pytest.approx(
        whole._train_loss_token_sum
    )
    assert (
        resumed._train_loss_token_sum / resumed._train_loss_token_total
    ) == pytest.approx(
        whole._train_loss_token_sum / whole._train_loss_token_total
    )


def test_log_payloads_carry_epoch(monkeypatch):
    # HF's Trainer.log does `logs["epoch"] = self.state.epoch` on every payload,
    # so log_history entries keep their epoch once state.epoch has moved on.
    # Without the stamp the persisted history has no recoverable epoch series.
    import mlx.core as mx
    from transformers import TrainerCallback

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    seen = []

    class Spy(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            seen.append((state.global_step, dict(logs or {})))

    args = MLXTrainingConfig(
        max_steps=4,
        gradient_accumulation_steps=1,
        logging_steps=1,
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
        callbacks=[Spy()],
    )
    trainer._prepare_data = lambda _is_vlm: ([make_batch(10) for _ in range(4)], None)
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None
    trainer.train()

    assert seen, "no on_log events fired"
    for step, logs in seen:
        assert "epoch" in logs, f"step {step} logs missing epoch: {sorted(logs)}"
    # The stamp is the live value, and it is preserved per entry in the history
    # that this trainer persists to trainer_state.json.
    assert [logs["epoch"] for _, logs in seen] == [
        entry["epoch"] for entry in trainer.state.log_history if "epoch" in entry
    ]
    assert [logs["epoch"] for _, logs in seen] == sorted(
        logs["epoch"] for _, logs in seen
    )


def test_integration_callback_args_cover_stock_trackio_and_swanlab():
    # transformers' own TrackioCallback / SwanLabCallback read TrainingArguments
    # fields straight off args inside on_train_begin, so any field this compat
    # shim omits aborts a real MLX run with AttributeError before step 1.
    import inspect
    import re

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    integration_utils = pytest.importorskip(
        "transformers.integrations.integration_utils"
    )

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = MLXTrainingConfig(output_dir="out")
    trainer.eval_dataset = None
    trainer._ensure_callback_args_compat()
    args = trainer.args

    # HF's own TrainingArguments defaults, so an unconfigured run reaches the
    # tracking SDK exactly as it would from a Torch Trainer.
    assert args.project == "huggingface"
    for name in (
        "trackio_space_id", "trackio_bucket_id", "trackio_static_space_id",
        "hub_private_repo", "resume_from_checkpoint",
    ):
        assert getattr(args, name) is None

    reader = re.compile(r"(?<![\w.])args\.(\w+)")
    for name in ("TrackioCallback", "SwanLabCallback"):
        callback = getattr(integration_utils, name, None)
        if callback is None:
            continue
        for field in sorted(set(reader.findall(inspect.getsource(callback)))):
            assert hasattr(args, field), f"{name} reads args.{field}"

    # A caller that configures the run keeps their value across the next run.
    args.project = "my-project"
    args.hub_private_repo = True
    trainer._ensure_callback_args_compat()
    assert args.project == "my-project"
    assert args.hub_private_repo is True


def test_callback_events_dispatch_on_every_rank(monkeypatch):
    # Regression for "Dispatch state-mutating callbacks on every rank". HF fires
    # callbacks in every process and expects host I/O to self-gate on
    # state.is_world_process_zero. Firing on rank 0 only left the peers'
    # process-local state un-mutated: an on_pre_optimizer_step callback that
    # overrides optimizer.learning_rate updated rank 0 alone, so the peers applied
    # the same all-reduced gradient with a different LR and the replicas silently
    # diverged. The per-rank flags _init_callback_state already seeds also stayed
    # unobservable, since only rank 0 (always world-process-zero) ever saw them.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    # Homogeneous world of 2: every rank contributes an identical value, so the
    # all-sum is the local value doubled and average_gradients is the identity.
    monkeypatch.setattr(
        trainer_mod.mx.distributed, "all_sum",
        lambda value, group=None, stream=None: value * mx.array(2, dtype=value.dtype),
    )
    monkeypatch.setattr(
        trainer_mod.nn, "average_gradients", lambda grad, group=None, **kw: grad,
    )

    base_lr, override_lr = 1e-5, 0.05

    class OverrideLR:
        def __init__(self):
            self.calls = 0

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self.calls += 1
            kwargs["optimizer"].learning_rate = mx.array(override_lr)
            return control

    class GatedHostIO:
        """Stock-callback shape: host I/O gated on the per-rank flag."""

        def __init__(self):
            self.writes = 0
            self.peer_flags = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.peer_flags.append(state.is_world_process_zero)
            if state.is_world_process_zero:
                self.writes += 1
            return control

    def run(rank):
        def _pinned_ensure_distributed(self):
            self._distributed_world = object()
            self._distributed_rank = rank
            self._distributed_world_size = 2
            self._distributed_is_main_process = (rank == 0)
            self._distributed_initialized = True
            return self._distributed_world

        monkeypatch.setattr(
            MLXTrainer, "_ensure_distributed", _pinned_ensure_distributed,
        )
        seen_lr = []
        lr_cb, io_cb = OverrideLR(), GatedHostIO()
        args = MLXTrainingConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            logging_steps=2,
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
            callbacks=[lr_cb, io_cb],
        )
        trainer._batches = _make_shape_guard_text_plan((8, 8, 8, 8))
        trainer.save_model = lambda *_a, **_kw: None

        def _recording_optimizer(_total_steps):
            optimizer = types.SimpleNamespace(
                learning_rate=mx.array(base_lr), state={},
            )
            # The fused update reads optimizer.learning_rate, so the value held
            # here is what actually moves this rank's parameters.
            optimizer.update = lambda _model, _grad: seen_lr.append(
                round(float(mx.array(optimizer.learning_rate).item()), 6)
            )
            return optimizer

        trainer._build_optimizer = _recording_optimizer
        trainer.train()
        return lr_cb, io_cb, seen_lr

    rank0_lr_cb, rank0_io_cb, rank0_seen = run(0)
    rank1_lr_cb, rank1_io_cb, rank1_seen = run(1)

    # The callback runs on the peer too, so both ranks step with the same LR.
    assert rank0_lr_cb.calls == 4
    assert rank1_lr_cb.calls == 4
    assert rank0_seen == [override_lr] * 4
    assert rank1_seen == rank0_seen
    # Guard the guard: rank-0-only dispatch left the peer on the un-overridden LR.
    assert rank1_seen != [base_lr] * 4

    # Cost check: the peer sees the real flag, so a stock-shaped callback still
    # does its host I/O exactly once across the world.
    assert rank0_io_cb.peer_flags and all(rank0_io_cb.peer_flags)
    assert rank1_io_cb.peer_flags and not any(rank1_io_cb.peer_flags)
    # Two cadence logs (logging_steps=2 over 4 steps) plus the run summary
    # train() dispatches before on_train_end.
    assert rank0_io_cb.writes == 3
    assert rank1_io_cb.writes == 0


def test_training_config_exposes_sanitized_dict_for_integration_callbacks():
    # Regression for "Add the serialization method required by NeptuneCallback".
    # HF's NeptuneCallback reads the config through args.to_sanitized_dict()
    # (integrations/integration_utils.py) with a bare attribute access, so
    # omitting it raised AttributeError out of on_train_begin and aborted the
    # run. HF reports the resolved batch sizes and keeps only exact
    # bool/int/float/str, stringifying anything a tracker cannot store.
    import json

    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    config = MLXTrainingConfig(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=3,
        compile_arch_overrides={"a": "b"},
    )
    sanitized = config.to_sanitized_dict()

    assert sanitized["train_batch_size"] == 2
    assert sanitized["eval_batch_size"] == 3
    # Every raw field survives; only the values are coerced.
    assert set(config.to_dict()) <= set(sanitized)
    assert all(type(value) in (bool, int, float, str) for value in sanitized.values())
    # A tracker must be able to serialize the whole payload.
    json.dumps(sanitized)
    # Non-scalars are stringified rather than dropped, and bool stays bool.
    assert sanitized["compile_arch_overrides"] == str({"a": "b"})
    assert isinstance(sanitized["packing"], bool)

    # Unset eval batch size falls back to the train batch size, as HF reports it.
    assert MLXTrainingConfig(
        per_device_train_batch_size=4,
    ).to_sanitized_dict()["eval_batch_size"] == 4


def test_concat_streaming_eval_is_infinite_when_any_child_is():
    from unsloth_zoo.mlx.trainer import _mlx_stream_declares_infinite

    class RepeatExamplesIterable:  # HF infinite marker: num_times=None
        num_times = None
        ex_iterable = None

    class _FiniteChild:  # no infinite markers -> finite
        ex_iterable = None

    class HorizontallyConcatenatedMultiSourcesExamplesIterable:
        def __init__(self, children): self.ex_iterables = children

    class VerticallyConcatenatedMultiSourcesExamplesIterable:
        def __init__(self, children): self.ex_iterables = children

    class _Src:
        def __init__(self, ex): self._ex_iterable = ex

    inf, fin = RepeatExamplesIterable(), _FiniteChild()
    # Horizontal concat ends with the longest child, so any infinite child wins.
    assert _mlx_stream_declares_infinite(
        _Src(HorizontallyConcatenatedMultiSourcesExamplesIterable([inf, fin]))) is True
    assert _mlx_stream_declares_infinite(
        _Src(HorizontallyConcatenatedMultiSourcesExamplesIterable([fin, inf]))) is True
    assert _mlx_stream_declares_infinite(
        _Src(HorizontallyConcatenatedMultiSourcesExamplesIterable([fin, fin]))) is False
    # Vertical (sequential) is infinite if any child is.
    assert _mlx_stream_declares_infinite(
        _Src(VerticallyConcatenatedMultiSourcesExamplesIterable([inf, fin]))) is True


@pytest.mark.parametrize("axis", [0, 1])
def test_real_hf_concat_infinite_detection_matches_actual_termination(axis):
    """Pin the detector against real ``datasets`` concat termination."""
    from datasets import IterableDataset, concatenate_datasets
    from unsloth_zoo.mlx.trainer import _mlx_stream_declares_infinite

    # Generator-backed: the arrow path rejects a ragged axis=1 concat.
    def rows(column, n):
        return lambda: ({column: str(i)} for i in range(n))

    finite = IterableDataset.from_generator(rows("a", 3))
    other = IterableDataset.from_generator(rows("b" if axis else "a", 2))
    for children, expected in (
        ((finite, other), False),
        ((finite, other.repeat(None)), True),
        ((finite.repeat(None), other), True),
    ):
        combined = concatenate_datasets(list(children), axis=axis)
        assert _mlx_stream_declares_infinite(combined) is expected
        if not expected:  # a finite stream must actually terminate
            assert sum(1 for _ in combined) < 1000
        else:  # an infinite stream must keep producing past every child's length
            produced = 0
            for _ in combined:
                produced += 1
                if produced > 1000:
                    break
            assert produced > 1000


def _prefetch_probe():
    """Prefetch producer stub recording its pause/resume/close lifecycle."""
    events = []

    class Probe:
        orphaned = False

        def quiesce(self):
            events.append("quiesce")

        def resume(self):
            events.append("resume")

        def close(self):
            events.append("close")
            return True

        def orphan_alive(self):
            return False

    return events, Probe()


def test_streaming_prefetch_is_paused_across_a_cadence_eval(monkeypatch):
    # The eval block moved into the _run_eval helper. The streaming prefetch
    # producer shares the tokenizer the eval pass re-enters, so it must still be
    # paused for the whole _evaluate and resumed afterwards, even if eval raises.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    events, probe = _prefetch_probe()

    class InstallProbe:
        def on_train_begin(self, args, state, control, **kwargs):
            # After train()'s entry _close_active_batch_iterator(), so the probe
            # survives to the eval.
            trainer._mlx_prefetch_control = {"prefetcher": probe}
            return control

    args = MLXTrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=1,
        logging_steps=100,
        eval_steps=2,
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
        callbacks=[InstallProbe()],
    )
    trainer._batches = _make_shape_guard_text_plan((10,) * 2)
    trainer._callback_batches_per_epoch = lambda _batches: 2
    trainer.eval_dataset = [{"input_ids": [1, 2, 3, 4]}]
    trainer._eval_batches_labeled = ["batch-0"]

    def _fake_evaluate(batches, loss_fn, is_vlm=False):
        events.append("evaluate")
        trainer._last_eval_metrics = {"eval_loss": 1.25, "eval_perplexity": 3.5}
        return 1.25, 3.5

    trainer._evaluate = _fake_evaluate
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None

    trainer.train()
    assert events[:3] == ["quiesce", "evaluate", "resume"]

    # A failing eval must still resume the producer (the finally), not leave it
    # paused for the rest of the run.
    events.clear()

    def _raising_evaluate(batches, loss_fn, is_vlm=False):
        events.append("evaluate")
        raise RuntimeError("eval blew up")

    trainer._evaluate = _raising_evaluate
    with pytest.raises(RuntimeError, match="eval blew up"):
        trainer.train()
    assert events[:3] == ["quiesce", "evaluate", "resume"]


def test_train_entry_clears_stale_stop_before_closing_prior_iterator():
    # Both entry actions survive. The stale-stop clear is a local assignment that
    # cannot fail; the iterator close can block or propagate an interrupt, so
    # running it first would leave a prior run's stop latched for the next train().
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer.train)
    assert src.index("self.stop_requested = False") < src.index(
        "self._close_active_batch_iterator()"
    )
    assert src.index("self._close_active_batch_iterator()") < src.index(
        "self._resume_from_checkpoint = resume_from_checkpoint"
    )


def test_step_checkpoint_save_surfaces_a_base_exception(monkeypatch):
    # The step-checkpoint save moved into the _run_checkpoint helper and must keep
    # catching BaseException: a KeyboardInterrupt inside the rank-0 write has to
    # travel through the distributed consensus, not unwind that rank alone.
    from unsloth_zoo.mlx import trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    def _boom(*_a, **_kw):
        raise KeyboardInterrupt("interrupted mid-write")

    monkeypatch.setattr(trainer_mod, "save_trainable_adapters", _boom)

    args = MLXTrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=1,
        logging_steps=100,
        save_steps=1,
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
    )
    trainer._batches = _make_shape_guard_text_plan((10,) * 2)
    trainer._callback_batches_per_epoch = lambda _batches: 2
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None

    with pytest.raises(KeyboardInterrupt, match="interrupted mid-write"):
        trainer.train()


def test_appended_config_fields_union_keeps_legacy_dumps_wholesale():
    # The appended-field set must be the union of every field either side added,
    # or a full-field dump from a release predating any of them stops counting as
    # a wholesale copy and its default warmup_steps overrides warmup_ratio.
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    appended = (
        "compile_max_variants",
        "label_smoothing_factor",
        "report_grad_norm",
        "max_eval_batches",
        "streaming_text_length_window_batches",
        "streaming_prefetch_batches",
        "logging_dir",
        "run_name",
    )
    field_names = {f.name for f in dataclasses.fields(MLXTrainingConfig)}
    assert set(appended) <= field_names

    source = MLXTrainingConfig(warmup_ratio=0.1)
    for dropped in appended:
        legacy = {
            f.name: getattr(source, f.name)
            for f in dataclasses.fields(MLXTrainingConfig)
            if f.name != dropped
        }
        restored = MLXTrainingConfig(**legacy)
        assert restored._unsloth_mlx_warmup_steps_explicit is False, dropped
        assert restored.warmup_ratio == 0.1, dropped

    # And every appended field dropped at once (the oldest dump we still accept).
    legacy = {
        f.name: getattr(source, f.name)
        for f in dataclasses.fields(MLXTrainingConfig)
        if f.name not in appended
    }
    assert MLXTrainingConfig(**legacy)._unsloth_mlx_warmup_steps_explicit is False

def test_input_token_counting_is_gated_on_its_argument(monkeypatch):
    # HF counts input tokens only when args.include_num_input_tokens_seen is
    # enabled; "no"/False (the default _ensure_callback_args_compat applies) skips
    # the whole block, so state.num_input_tokens_seen stays 0 and no gather runs
    # (transformers trainer.py: _track_num_input_tokens on 5.x, the inline
    # include_num_input_tokens_seen block on 4.57.x). The MLX loop counted and
    # all-reduced unconditionally, so an opted-out run showed callbacks a nonzero
    # counter and paid an extra lockstep collective on every micro-batch.
    import tempfile

    from transformers import TrainerCallback

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    widths = (8, 9, 10, 11)

    class TokenSpy(TrainerCallback):
        def __init__(self):
            self.per_step = []

        def on_step_end(self, args, state, control, **kwargs):
            self.per_step.append(int(state.num_input_tokens_seen))
            return control

    def run(flag):
        spy = TokenSpy()
        args = MLXTrainingConfig(
            max_steps=len(widths),
            gradient_accumulation_steps=1,
            logging_steps=len(widths),
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            output_dir=tempfile.mkdtemp(),
        )
        if flag is not None:
            args.include_num_input_tokens_seen = flag
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [],
            args=args,
            callbacks=[spy],
        )
        collectives = []
        inner = trainer._distributed_all_sum

        def counting_all_sum(value, stream=None):
            collectives.append(1)
            return inner(value, stream=stream)

        trainer._distributed_all_sum = counting_all_sum
        trainer._batches = _make_shape_guard_text_plan(widths)
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        trainer.train()
        return spy.per_step, len(collectives), trainer.state.num_input_tokens_seen

    # Enabled: the running global total of forwarded input positions, matching
    # real transformers.Trainer on the same widths.
    on_steps, on_collectives, on_final = run("all")
    assert on_steps == [8, 17, 27, 38]
    assert on_final == 38

    # Every disabled spelling HF accepts leaves the counter untouched.
    for disabled in (None, False, "no"):
        off_steps, off_collectives, off_final = run(disabled)
        assert off_steps == [0, 0, 0, 0], disabled
        assert off_final == 0, disabled
        # ...and skips exactly one all-reduce per micro-batch.
        assert off_collectives == on_collectives - len(widths), disabled

    # "non_padding" is an enabled mode, not an opt-out.
    assert run("non_padding")[0] == [8, 17, 27, 38]


def test_epoch_final_microbatch_forces_optimizer_step_budget():
    # HF sets do_sync_step on an epoch's last micro-batch
    # (transformers _run_epoch: `(step + 1) == steps_in_epoch`) and sizes the run
    # with num_update_steps_per_epoch = ceil(len_dataloader / grad_accum). Flooring
    # the whole stream left the epoch's tail un-applied at on_epoch_end and folded
    # it into the next epoch's window. Measured against transformers 5.14.1 and
    # 4.57.6: 3 micro-batches at grad_accum=2 over 2 epochs runs 4 steps.
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _mlx_steps_per_epoch,
        _resolve_training_steps,
    )

    def steps(n_batches, grad_accum, epochs, includes_epochs=False):
        args = MLXTrainingConfig(
            max_steps=-1,
            num_train_epochs=epochs,
            gradient_accumulation_steps=grad_accum,
        )
        return _resolve_training_steps(
            args, [0] * n_batches, None, includes_epochs=includes_epochs,
        )

    assert steps(3, 2, 2) == 4          # HF: 4, the old floor gave 3
    assert steps(2, 4, 4) == 4          # HF: 4, the old floor gave 2
    assert steps(5, 2, 2) == 6          # HF: 6, the old floor gave 5
    # Divisible epochs are untouched: ceil == floor, so the numerics do not move.
    assert steps(4, 2, 2) == 4
    assert steps(8, 4, 3) == 6
    assert steps(6, 1, 2) == 12
    # torch_randperm layout ceils the per-epoch slice, not the whole stream.
    assert steps(6, 2, 2, includes_epochs=True) == 4
    assert steps(8, 2, 2, includes_epochs=True) == 4
    # An epoch shorter than one accumulation window still costs a full step.
    assert _mlx_steps_per_epoch(2, 4) == 1
    assert _mlx_steps_per_epoch(3, 2) == 2
    assert _mlx_steps_per_epoch(4, 2) == 2


def test_fractional_num_train_epochs_keeps_its_step_budget():
    # num_train_epochs is a float in TrainingArguments/SFTConfig
    # (`num_train_epochs: float = field(default=3.0, ...)`, identical in
    # transformers 5.14.1 and 4.57.6) and a fractional value is supported:
    # Trainer.set_initial_training_values takes the epoch-based branch and sets
    #   max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    # with num_update_steps_per_epoch = max(ceil(len_dataloader / grad_accum), 1).
    # Truncating the epoch count instead collapsed 1.5 onto 1.0 and sent any
    # value below 1.0 down the no-epoch path, which budgets a whole floored pass.
    # Every expectation below was measured against a real transformers.Trainer on
    # both 5.14.1 and 4.57.6, which agree cell for cell.
    import math

    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _mlx_epoch_microbatches,
        _resolve_training_steps,
    )

    def steps(n_batches, grad_accum, epochs, includes_epochs=False):
        args = MLXTrainingConfig(
            max_steps=-1,
            num_train_epochs=epochs,
            gradient_accumulation_steps=grad_accum,
        )
        return _resolve_training_steps(
            args, [0] * n_batches, None, includes_epochs=includes_epochs,
        )

    # The item's example: 8 micro-batches, grad_accum 2, 1.5 epochs is six
    # updates (ceil(1.5 * 4)), not the four a truncated 1.0 would budget.
    assert steps(8, 2, 1.5) == 6
    # Below one epoch the run is shorter than a pass, never a full floored pass.
    assert steps(8, 2, 0.5) == 2
    assert steps(4, 1, 0.5) == 2
    assert steps(4, 2, 0.5) == 1
    assert steps(3, 1, 0.5) == 2
    # Ragged epochs ceil the per-epoch cost first, then scale by the epochs.
    assert steps(3, 2, 1.5) == 3           # ceil(1.5 * ceil(3/2))
    assert steps(3, 2, 2.5) == 5
    assert steps(5, 2, 1.5) == 5           # ceil(1.5 * ceil(5/2))
    assert steps(5, 1, 2.5) == 13
    assert steps(4, 3, 1.5) == 3
    # A whole number of epochs is byte-identical to the integer budget, so no
    # existing run changes: ceil(E * steps_per_epoch) == E * steps_per_epoch.
    for n_batches in (3, 4, 5, 6, 8):
        for grad_accum in (1, 2, 3, 4):
            for epochs in (1, 2, 3):
                per_epoch = max(1, math.ceil(n_batches / grad_accum))
                assert steps(n_batches, grad_accum, epochs) == epochs * per_epoch
                # A float that happens to be whole must land on the same budget.
                assert (steps(n_batches, grad_accum, float(epochs))
                        == epochs * per_epoch)
    # The prebuilt-every-epoch layout is unchanged as well.
    assert steps(6, 2, 2, includes_epochs=True) == 4
    assert steps(8, 2, 2, includes_epochs=True) == 4

    # A sub-one epoch count still declares epoch boundaries: it must not fall
    # back to the flat no-epoch model, which is what dropped the budget onto a
    # whole floored pass.
    half = MLXTrainingConfig(
        max_steps=-1, num_train_epochs=0.5, gradient_accumulation_steps=2,
    )
    assert _mlx_epoch_microbatches(half, [0] * 8) == 8
    # max_steps still wins over any epoch count, fractional included.
    capped = MLXTrainingConfig(
        max_steps=7, num_train_epochs=1.5, gradient_accumulation_steps=2,
    )
    assert _resolve_training_steps(capped, [0] * 8, None) == 7
    assert _mlx_epoch_microbatches(capped, [0] * 8) is None


def test_fractional_epoch_shape_catalog_covers_the_partial_epoch():
    # The compiled text-shape catalog enumerates the micro-batches the run will
    # fetch. A fractional num_train_epochs stops part-way through its last epoch,
    # so flooring the budget to whole epochs under-enumerated that tail and left
    # its widths out of the plan, which a strict compile scope then rejects at
    # runtime. The enumeration reuses the runtime's own step -> micro-batch
    # mapping, so whole epoch counts are unchanged.
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _mlx_microstep_for_step,
        _mlx_microstep_phase,
        _plan_single_process_text_shapes,
        _resolve_training_steps,
    )

    # 4 micro-batches/epoch, grad_accum 2, 1.5 epochs -> 3 updates -> the first
    # epoch's 4 micro-batches plus 2 more, which is what the loop visits.
    plan = _make_shape_guard_text_plan((10, 11, 30, 12))
    args = MLXTrainingConfig(
        max_steps=-1,
        num_train_epochs=1.5,
        gradient_accumulation_steps=2,
        compile_max_variants=16,
    )
    total_steps = _resolve_training_steps(args, plan, None)
    assert total_steps == 3
    shape_plan, report, allowed, _ = _plan_single_process_text_shapes(
        plan, None, args=args, total_steps=total_steps, is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=args),
    )
    assert allowed
    visited = _mlx_microstep_for_step(total_steps, 4, 2)
    assert visited == 6
    # Every micro-batch the loop will fetch is in the catalog, the partial
    # second epoch included.
    for microstep in range(visited):
        index = plan.batch_index_for_visit(microstep)
        assert shape_plan.allows(
            plan.batch_family(index),
            plan.batch_width(index),
            _mlx_microstep_phase(report.compile_scope, 2, microstep, 4),
        )
    # Whole epochs land exactly on a boundary, matching the old floored form.
    for epoch_microbatches in (3, 4, 5, 8):
        for grad_accum in (1, 2, 3):
            per_epoch = max(1, -(-epoch_microbatches // grad_accum))
            for epochs in (1, 2, 3):
                assert (
                    _mlx_microstep_for_step(
                        epochs * per_epoch, epoch_microbatches, grad_accum,
                    )
                    == epochs * epoch_microbatches
                )


def test_max_steps_and_single_pass_budgets_keep_the_flat_model():
    # The forced flush is scoped to epoch-count runs. A max_steps run keeps its
    # fixed budget (batches_per_epoch there is an approximation that must not move
    # optimizer steps) and a run with no declared epoch count keeps one floored
    # pass, so neither changes numerics.
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _mlx_epoch_microbatches,
        _resolve_training_steps,
    )

    max_steps_args = MLXTrainingConfig(
        max_steps=7, num_train_epochs=2, gradient_accumulation_steps=2,
    )
    assert _resolve_training_steps(max_steps_args, [0] * 3, None) == 7
    assert _mlx_epoch_microbatches(max_steps_args, [0] * 3) is None

    no_epochs = MLXTrainingConfig(
        max_steps=-1, num_train_epochs=-1, gradient_accumulation_steps=2,
    )
    assert _resolve_training_steps(no_epochs, [0] * 3, None) == 1  # 3 // 2
    assert _mlx_epoch_microbatches(no_epochs, [0] * 3) is None

    # Streaming has no materialized epoch, so it keeps the flat cursor too.
    streaming = MLXTrainingConfig(
        max_steps=-1, num_train_epochs=2, gradient_accumulation_steps=2,
    )
    assert _mlx_epoch_microbatches(streaming, None) is None
    assert _mlx_epoch_microbatches(streaming, []) is None


def test_epoch_microbatches_agrees_with_callback_epoch_length():
    # The step budget, the forced epoch-final update, the resume cursor and the
    # on_epoch_begin/end dispatch must all use one epoch length, or the loop
    # flushes at a micro-batch the callbacks do not consider a boundary.
    import types as _types

    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _mlx_epoch_microbatches,
    )

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer._distributed_world_size = 1
    trainer.train_dataset = [{"text": "row"}] * 12
    trainer._mlx_train_dataset_for_batches = trainer.train_dataset
    for max_steps, epochs, includes, n_batches in (
        (-1, 2, False, 3),
        (-1, 2, False, 4),
        (-1, 4, False, 2),
        (-1, 2, True, 6),
        (-1, 2, True, 7),
        (-1, -1, False, 5),
        (6, 2, False, 5),
        (6, 2, True, 6),
    ):
        trainer.args = MLXTrainingConfig(
            max_steps=max_steps,
            num_train_epochs=epochs,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=2,
        )
        trainer._prepared_batches_include_epochs = includes
        resolved = _mlx_epoch_microbatches(
            trainer.args, [0] * n_batches, includes_epochs=includes,
        )
        if resolved is not None:
            assert resolved == trainer._callback_batches_per_epoch(
                [0] * n_batches
            ), (max_steps, epochs, includes, n_batches)
    del _types


def test_resume_cursor_is_epoch_aligned_for_ragged_epochs():
    # Once an epoch's tail forces a step, global_step no longer maps flatly onto
    # micro-batches, so rebuilding the cursor as global_step * grad_accum skips the
    # next epoch's opening micro-batch and cycles into an unplanned extra pass.
    # HF rebuilds it per epoch (epochs_trained * steps_in_epoch +
    # global_step % num_update_steps_per_epoch * grad_accum).
    from unsloth_zoo.mlx.trainer import _mlx_microstep_for_step

    # 3 micro-batches at grad_accum=2: steps consume 2, 1, 2, 1 micro-batches.
    consumed = [_mlx_microstep_for_step(step, 3, 2) for step in range(5)]
    assert consumed == [0, 2, 3, 5, 6]
    # The flat rule would have produced 4 at step 2, skipping micro-batch 3.
    assert consumed[2] != 2 * 2
    # 2 micro-batches at grad_accum=4: one step per epoch.
    assert [_mlx_microstep_for_step(s, 2, 4) for s in range(5)] == [
        0, 2, 4, 6, 8,
    ]
    # Divisible epochs keep the flat mapping exactly.
    assert [_mlx_microstep_for_step(s, 6, 2) for s in range(4)] == [0, 2, 4, 6]
    assert [_mlx_microstep_for_step(s, 4, 2) for s in range(5)] == [
        0, 2, 4, 6, 8,
    ]


def test_epoch_flush_microstep_phase_marks_the_forced_update():
    # The compiled step's argument signature changes when the epoch-final
    # micro-batch forces the update, so the shape guard's enumerated catalog must
    # use the same per-epoch phase the runtime materialize() call passes.
    from unsloth_zoo.mlx.shape_guard import (
        DDP_LOCAL_GRAD_SCOPE,
        FULL_STEP_SCOPE,
        phase_for_microstep,
    )
    from unsloth_zoo.mlx.trainer import _mlx_microstep_phase

    # 3 micro-batches per epoch, grad_accum=2. Micro-batch 2 (0-based) opens a
    # fresh window AND forces the update, so it traces the single-batch update.
    phases = [
        _mlx_microstep_phase(FULL_STEP_SCOPE, 2, m, 3) for m in range(6)
    ]
    assert phases == [
        "none_no_update", "tree_update", "single",
        "none_no_update", "tree_update", "single",
    ]
    # 3 micro-batches per epoch, grad_accum=3: the boundary already updates.
    assert _mlx_microstep_phase(FULL_STEP_SCOPE, 3, 2, 3) == "tree_update"
    # 4 micro-batches per epoch, grad_accum=3: the tail carries a partial tree.
    assert [
        _mlx_microstep_phase(FULL_STEP_SCOPE, 3, m, 4) for m in range(4)
    ] == ["none_no_update", "tree_no_update", "tree_update", "single"]
    # grad_accum=1 updates on every micro-batch, flush or not.
    assert _mlx_microstep_phase(FULL_STEP_SCOPE, 1, 2, 3) == "single"
    # The DDP local-gradient scope never sees do_update, so only the window
    # position matters, but it still restarts at the epoch boundary.
    assert [
        _mlx_microstep_phase(DDP_LOCAL_GRAD_SCOPE, 2, m, 3) for m in range(6)
    ] == ["none", "tree", "none", "none", "tree", "none"]
    # Without an epoch length it is exactly the flat mapping.
    for m in range(12):
        assert _mlx_microstep_phase(FULL_STEP_SCOPE, 3, m) == (
            phase_for_microstep(FULL_STEP_SCOPE, 3, m)
        )


def test_epoch_flush_shape_plan_admits_the_runtime_phase_sequence():
    # The planner enumerates the real micro-batch stream (whole epochs, not
    # total_steps * grad_accum) with the same phase helper the loop passes to
    # FiniteTextBatchPlan.materialize, so no visited signature is rejected.
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import (
        MLXTrainingConfig,
        _mlx_microstep_phase,
        _plan_single_process_text_shapes,
        _resolve_training_steps,
    )

    plan = _make_shape_guard_text_plan((10, 11, 30))
    args = MLXTrainingConfig(
        max_steps=-1,
        num_train_epochs=2,
        gradient_accumulation_steps=2,
        compile_max_variants=8,
    )
    total_steps = _resolve_training_steps(args, plan, None)
    assert total_steps == 4
    shape_plan, report, allowed, _ = _plan_single_process_text_shapes(
        plan, None, args=args, total_steps=total_steps, is_vlm=False,
        distributed_world_size=1,
        compile_policy=build_compile_policy(args=args),
    )
    assert allowed
    # Exactly the 6 micro-batches the loop visits, no phantom third pass.
    assert shape_plan.raw_catalog == frozenset(
        (
            report.compile_scope,
            _mlx_microstep_phase(report.compile_scope, 2, m, 3),
            plan.batch_family(plan.batch_index_for_visit(m)),
            plan.batch_width(plan.batch_index_for_visit(m)),
        )
        for m in range(6)
    )
    for microstep in range(6):
        assert shape_plan.allows(
            plan.batch_family(plan.batch_index_for_visit(microstep)),
            plan.batch_width(plan.batch_index_for_visit(microstep)),
            _mlx_microstep_phase(report.compile_scope, 2, microstep, 3),
        )


def test_max_steps_epoch_microbatches_use_the_exact_plan_cycle():
    # HF's forced epoch-final update is NOT conditional on max_steps: do_sync_step
    # is "(step + 1) % gradient_accumulation_steps == 0 or (step + 1) ==
    # steps_in_epoch", and steps_in_epoch is len(dataloader) whenever the
    # dataloader reports a length -- max_steps only decides when the run ends.
    # _mlx_epoch_microbatches returned None for every max_steps run, so the epoch's
    # last micro-batch landed on a non-update microstep.
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig, _mlx_epoch_microbatches

    args = MLXTrainingConfig(
        max_steps=3, gradient_accumulation_steps=2, num_train_epochs=3,
    )
    # An exact one-pass count from the plan drives the flush.
    plan = _make_shape_guard_text_plan((4, 5, 6, 4, 5, 6))
    assert _mlx_epoch_microbatches(args, plan) is None  # no cycle recorded
    plan_with_cycle = _make_shape_guard_text_plan((4, 5, 6, 4, 5, 6))
    object.__setattr__(plan_with_cycle, "_cycle_length", 3)
    assert plan_with_cycle.cycle_length == 3
    assert _mlx_epoch_microbatches(args, plan_with_cycle) == 3
    # A source with no cycle length keeps the flat model: _callback_batches_per_
    # epoch's dataset-size approximation cannot see what batching retained, so
    # forcing updates on it would move optimizer steps onto non-boundaries.
    assert _mlx_epoch_microbatches(args, [object()] * 6) is None
    # Epoch-count runs are untouched by the max_steps branch.
    epoch_args = MLXTrainingConfig(
        max_steps=-1, gradient_accumulation_steps=2, num_train_epochs=2,
    )
    assert _mlx_epoch_microbatches(epoch_args, [object()] * 5) == 5


def test_max_steps_ragged_pass_flushes_before_the_epoch_callbacks(monkeypatch):
    # End-to-end through the real loop on the path a user actually hits: a
    # max_steps run over a finite dataset gets a FiniteTextBatchPlan whose
    # cycle_length is the exact one-pass micro-batch count. With 3 micro-batches
    # per pass and grad_accum=2 the pass boundary fell on a NON-update microstep,
    # so on_epoch_end (and the epoch-cadence checkpoint it requests) observed the
    # model with the pass's last gradient still pending, and that gradient was
    # then folded into the next pass's accumulation window.
    #
    # Golden values from real transformers (5.14.1 AND 4.57.6): 6 rows at
    # per_device_train_batch_size=2, gradient_accumulation_steps=2, max_steps=3,
    # save_strategy="epoch" -> 3 optimizer steps, on_epoch_end at (2, 1.0) and
    # (3, 5/3), on_substep_end only at global_step 0 and 2, checkpoints at steps
    # 2 and 3. The flat model produced on_epoch_end (1, 1.0) / (3, 2.0) and wrote
    # its first checkpoint at step 1.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan

    _patch_value_and_grad_with_aux(monkeypatch)

    class RecordingPlan(FiniteTextBatchPlan):
        """FiniteTextBatchPlan that records the micro-batches the loop visits."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.visits = []

        def __getitem__(self, index):
            self.visits.append(int(index))
            return super().__getitem__(index)

    class Spy:
        def __init__(self):
            self.epoch_end = []
            self.step_end = []
            self.substep_end = []
            self.saves = []

        def on_epoch_end(self, args, state, control, **kwargs):
            self.epoch_end.append((state.global_step, round(float(state.epoch), 6)))
            control.should_save = True          # save_strategy="epoch"
            return control

        def on_step_end(self, args, state, control, **kwargs):
            self.step_end.append((state.global_step, round(float(state.epoch), 6)))
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            self.substep_end.append(state.global_step)
            return control

        def on_save(self, args, state, control, **kwargs):
            self.saves.append((state.global_step, round(float(state.epoch), 6)))
            return control

    from unsloth_zoo.mlx.utils import _FiniteTextRow

    # Two passes of 3 micro-batches; cycle_length records the one-pass count.
    rows = tuple(
        _FiniteTextRow(tuple(range(1, 7)), offset=1, labels=tuple(range(1, 7)))
        for _ in range(6)
    )
    plan = RecordingPlan(
        rows,
        tuple((index,) for index in range(6)),
        cycle_length=3,
        max_seq_length=64,
        pad_id=99,
    )
    assert plan.cycle_length == 3

    spy = Spy()
    args = MLXTrainingConfig(
        max_steps=3,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
        logging_steps=10 ** 6,
        save_steps=0,
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
    trainer._batches = plan
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None
    trainer.train()

    assert trainer._global_step == 3
    assert spy.step_end == [(1, 0.666667), (2, 1.0), (3, 1.666667)]
    assert spy.epoch_end == [(2, 1.0), (3, 1.666667)]
    # The pass's 3rd micro-batch forced its own update, so it is NOT a substep.
    assert spy.substep_end == [0, 2]
    assert spy.saves == [(2, 1.0), (3, 1.666667)]
    # The epoch checkpoint lands where HF's does, holding a model that HAS seen
    # the pass's last micro-batch. The flat model wrote checkpoint-1 instead.
    assert sorted(
        entry for entry in os.listdir(args.output_dir)
        if entry.startswith("checkpoint-")
    ) == ["checkpoint-2", "checkpoint-3"]
    # Step 1 took micro-batches 0+1, step 2 took the pass tail (2) alone, step 3
    # opened the next pass with 3+4 -- the tail never mixed across the boundary.
    assert plan.visits == [0, 1, 2, 3, 4]


def test_epoch_flush_wiring_present_in_source():
    # Guard the three call sites so the flush cannot be silently reverted:
    # the forced boundary update, the epoch-aligned resume cursor, and the
    # per-epoch phase handed to the compiled materialize().
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "_epoch_flush_microbatches = _mlx_epoch_microbatches(" in src
    assert "and it % _epoch_flush_microbatches == 0" in src
    assert "_resume_microstep = _mlx_microstep_for_step(" in src
    assert "batch_idx = _resume_microstep" in src
    assert "microstep = _resume_microstep" in src
    # The flat cursor survives only as the non-epoch fallback.
    assert src.count("= _resume_step * grad_accum") == 1
    assert "_resume_microstep = _resume_step * grad_accum" in src
    assert "range(_resume_microstep)" in src
    assert "phase=_mlx_microstep_phase(" in src


def _epoch_flush_loop_trainer(
    out_dir, *, microbatches_per_epoch, grad_accum, epochs, save_steps=0,
    callbacks=(),
):
    """Trainer wired to run the real loop over recorded micro-batch visits."""
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class RecordingBatches(list):
        def __init__(self, items):
            super().__init__(items)
            self.visits = []

        def __getitem__(self, index):
            self.visits.append(int(index))
            return list.__getitem__(self, index)

    def make_batch(seed):
        ids = mx.array([[seed + 1] * 6], dtype=mx.int32)
        lengths = mx.array([[0, 5]], dtype=mx.int32)
        return (ids, lengths, None)

    batches = RecordingBatches(
        [make_batch(i) for i in range(microbatches_per_epoch)]
    )
    args = MLXTrainingConfig(
        max_steps=-1,
        num_train_epochs=epochs,
        gradient_accumulation_steps=grad_accum,
        logging_steps=10 ** 6,
        save_steps=save_steps,
        warmup_steps=5,
        learning_rate=6e-4,
        lr_scheduler_type="linear",
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
        [{"text": f"row {i}"} for i in range(microbatches_per_epoch)],
        args=args,
        callbacks=list(callbacks),
    )
    trainer._prepare_data = lambda _is_vlm: (batches, None)
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None
    return trainer, batches


class _EpochScheduleSpy:
    """Records the (global_step, epoch) pairs HF callbacks observe."""

    def __init__(self):
        self.epoch_end = []
        self.step_end = []
        self.substep_end = []
        self.epoch_begin = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_begin.append((state.global_step, round(float(state.epoch), 6)))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_end.append((state.global_step, round(float(state.epoch), 6)))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.step_end.append((state.global_step, round(float(state.epoch), 6)))
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        self.substep_end.append((state.global_step, round(float(state.epoch), 6)))
        return control


def test_ragged_epoch_schedule_matches_transformers(monkeypatch):
    # End-to-end through the real loop. Golden values measured by running
    # transformers.Trainer (5.14.1 and 4.57.6) on the same shape: 6 rows at
    # per_device_train_batch_size=2 is 3 micro-batches per epoch, grad_accum=2,
    # num_train_epochs=2. HF runs 4 optimizer steps and fires on_epoch_end at
    # (2, 1.0) and (4, 2.0); the flat model ran 3 and fired at (1, 1.0), (3, 2.0)
    # with the epoch's last gradient still pending at the first on_epoch_end.
    _patch_value_and_grad_with_aux(monkeypatch)

    spy = _EpochScheduleSpy()
    trainer, batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=3, grad_accum=2, epochs=2,
        callbacks=[spy],
    )
    trainer.train()

    assert trainer.state.max_steps == 4
    assert trainer._global_step == 4
    assert spy.epoch_end == [(2, 1.0), (4, 2.0)]
    assert spy.step_end == [
        (1, 0.666667), (2, 1.0), (3, 1.666667), (4, 2.0),
    ]
    # Every micro-batch is visited exactly once per epoch and no step straddles
    # the boundary: visits 0,1 -> step 1; visit 2 -> step 2 (forced).
    assert batches.visits == [0, 1, 2, 0, 1, 2]


def test_divisible_epoch_schedule_is_unchanged(monkeypatch):
    # Backwards compatibility: when the epoch divides evenly by grad_accum the
    # ceil equals the floor, no micro-batch is ever a forced flush that was not
    # already an update, and the schedule is bit-for-bit the pre-fix one (which
    # also equals transformers: 8 rows at batch 2, grad_accum=2, 2 epochs -> 4).
    _patch_value_and_grad_with_aux(monkeypatch)

    spy = _EpochScheduleSpy()
    trainer, batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=4, grad_accum=2, epochs=2,
        callbacks=[spy],
    )
    trainer.train()

    assert trainer.state.max_steps == 4
    assert spy.epoch_end == [(2, 1.0), (4, 2.0)]
    assert spy.step_end == [(1, 0.5), (2, 1.0), (3, 1.5), (4, 2.0)]
    assert batches.visits == [0, 1, 2, 3, 0, 1, 2, 3]


@pytest.mark.parametrize(
    "microbatches_per_epoch,grad_accum,epochs,expect_substep_end",
    [
        # Golden values measured on real transformers 5.14.1 AND 4.57.6, which
        # agree cell for cell. per_device_train_batch_size=1, so
        # microbatches_per_epoch == len(dataloader) == steps_in_epoch.
        # Divisible epoch: substeps at micro-batches 1 and 3 of each epoch.
        (4, 2, 2, [(0, 0.0), (1, 0.5), (2, 1.0), (3, 1.5)]),
        # More substeps per epoch.
        (6, 2, 2, [(0, 0.0), (1, 0.333333), (2, 0.666667),
                   (3, 1.0), (4, 1.333333), (5, 1.666667)]),
        # Ragged epoch: the epoch's last micro-batch forces its own update, so
        # it is a step and not a substep.
        (5, 2, 2, [(0, 0.0), (1, 0.4), (3, 1.0), (4, 1.4)]),
        # Fractional epoch count, partial final epoch.
        (5, 2, 1.5, [(0, 0.0), (1, 0.4), (3, 1.0), (4, 1.4)]),
    ],
)
def test_substep_end_reports_the_last_completed_step_epoch(
    monkeypatch, microbatches_per_epoch, grad_accum, epochs, expect_substep_end,
):
    # HF advances the callback-visible epoch only on an optimizer step, next to
    # the global_step it belongs with:
    #   state.global_step += 1
    #   state.epoch = epoch + (step + 1) / steps_in_epoch
    #   on_step_end(...)
    # and the non-sync branch fires on_substep_end with the epoch UNTOUCHED
    # (transformers _inner_training_loop, identical in 5.14.1 and 4.57.6). The
    # MLX loop advanced state.epoch after EVERY micro-batch, so on_substep_end
    # reported an epoch one micro-batch AHEAD of the last completed step -- a
    # substep callback saw 0.25 where HF reports 0.0, and the value disagreed
    # with the global_step handed to it in the same call.
    _patch_value_and_grad_with_aux(monkeypatch)

    spy = _EpochScheduleSpy()
    trainer, _batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=microbatches_per_epoch,
        grad_accum=grad_accum, epochs=epochs, callbacks=[spy],
    )
    trainer.train()

    assert spy.substep_end == expect_substep_end
    # The invariant behind the goldens: a substep never moves the epoch, so it
    # always reports whatever the previous event left there.
    seen = dict(spy.epoch_begin)
    seen.update(dict(spy.step_end))
    for step, epoch in spy.substep_end:
        assert epoch == seen[step], (step, epoch, seen[step])


def test_short_epoch_saves_one_checkpoint_per_epoch(monkeypatch):
    # 2 micro-batches per epoch at grad_accum=4 used to run 2 optimizer steps for
    # 4 epochs, so an epoch-end save fired at global_step 0 (writing a
    # checkpoint-0) and epochs 2 and 3 both wrote checkpoint-1, silently
    # overwriting each other. transformers writes checkpoint-1..4.
    _patch_value_and_grad_with_aux(monkeypatch)

    class SaveEachEpochEnd:
        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

    out_dir = tempfile.mkdtemp()
    spy = _EpochScheduleSpy()
    trainer, _batches = _epoch_flush_loop_trainer(
        out_dir, microbatches_per_epoch=2, grad_accum=4, epochs=4,
        callbacks=[spy, SaveEachEpochEnd()],
    )
    trainer.train()

    assert trainer.state.max_steps == 4
    assert spy.epoch_end == [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)]
    assert sorted(
        d for d in os.listdir(out_dir) if d.startswith("checkpoint-")
    ) == ["checkpoint-1", "checkpoint-2", "checkpoint-3", "checkpoint-4"]
    assert not os.path.isdir(os.path.join(out_dir, "checkpoint-0"))


def test_epoch_aligned_resume_consumes_each_microbatch_once(monkeypatch):
    # A ragged-epoch resume must pick up at the epoch boundary the checkpoint
    # closed. The flat cursor (global_step * grad_accum) skipped the next epoch's
    # opening micro-batch and ran two micro-batches into a third dataset pass
    # num_train_epochs never authorised. transformers resumes the same run with
    # epochs_trained = global_step // num_update_steps_per_epoch and consumes
    # exactly the 3 remaining micro-batches.
    _patch_value_and_grad_with_aux(monkeypatch)

    out_dir = tempfile.mkdtemp()
    first, first_batches = _epoch_flush_loop_trainer(
        out_dir, microbatches_per_epoch=3, grad_accum=2, epochs=2, save_steps=1,
    )
    first.train()
    assert first_batches.visits == [0, 1, 2, 0, 1, 2]
    ckpt = os.path.join(out_dir, "checkpoint-2")
    assert os.path.isdir(ckpt), sorted(os.listdir(out_dir))

    spy = _EpochScheduleSpy()
    resumed, resumed_batches = _epoch_flush_loop_trainer(
        out_dir, microbatches_per_epoch=3, grad_accum=2, epochs=2, save_steps=1,
        callbacks=[spy],
    )
    resumed.train(resume_from_checkpoint=ckpt)

    # Epoch 2 only: 3 micro-batches, 2 optimizer steps, no repeat of epoch 1 and
    # no third pass.
    assert resumed_batches.visits == [0, 1, 2]
    assert resumed._global_step == 4
    assert spy.step_end == [(3, 1.666667), (4, 2.0)]
    assert spy.epoch_end == [(4, 2.0)]


def test_epoch_stop_budget_keeps_untruncated_epochs_whole(monkeypatch):
    # A should_epoch_stop callback truncates ONE epoch. Recomputing the remaining
    # budget with a flat floor also shortened the epochs that were never
    # truncated, silently dropping their last micro-batch. 5 micro-batches per
    # epoch at grad_accum=2 over 2 epochs, stopping epoch 1 after step 2:
    # transformers runs 5 steps (2 + a full 3) and fires on_epoch_end at
    # (2, 0.8) then (5, 2.0).
    _patch_value_and_grad_with_aux(monkeypatch)

    class StopEpochAtStepTwo:
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_epoch_stop = True
            return control

    spy = _EpochScheduleSpy()
    trainer, batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=5, grad_accum=2, epochs=2,
        callbacks=[spy, StopEpochAtStepTwo()],
    )
    trainer.train()

    assert trainer._global_step == 5
    assert spy.epoch_end == [(2, 0.8), (5, 2.0)]
    # Epoch 1 abandoned its 5th micro-batch; epoch 2 ran all five of its own.
    assert batches.visits == [0, 1, 2, 3, 0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "microbatches_per_epoch,epochs,stop_at,expect_steps,expect_epoch_end,expect_visits",
    [
        # 10 rows at per_device_train_batch_size=2 is 5 micro-batches per epoch,
        # grad_accum=2 (3 optimizer steps per epoch), num_train_epochs=1.5 ->
        # max_steps = ceil(1.5 * 3) = 5. Stopping epoch 1 after its first update
        # leaves the whole authorized second epoch: HF runs 1 + 3 = 4 steps.
        (5, 1.5, 1, 4, [(1, 0.4), (4, 2.0)], [0, 1, 0, 1, 2, 3, 4]),
        # num_train_epochs=2.5 -> ceil(2.5 * 3) = 8 authorized steps; the stop
        # forfeits only epoch 1's tail, so HF runs 1 + 3 + 3 = 7.
        (5, 2.5, 1, 7, [(1, 0.4), (4, 2.0), (7, 3.0)],
         [0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]),
        # A DIVISIBLE epoch (4 micro-batches, grad_accum=2) is affected too: the
        # defect is the truncated epoch count, not the ragged tail. 2 steps per
        # epoch, ceil(1.5 * 2) = 3 authorized; HF runs 1 + 2 = 3.
        (4, 1.5, 1, 3, [(1, 0.5), (3, 2.0)], [0, 1, 0, 1, 2, 3]),
        # The ceiled horizon must never GROW the budget past what the fractional
        # run authorized. 7 micro-batches per epoch, grad_accum=2 -> 4 steps per
        # epoch, ceil(1.5 * 4) = 6 authorized. Stopping at step 3 and then running
        # a whole second epoch would reach 3 + 4 = 7, but HF stops at max_steps:
        # 6 steps, with the last epoch left partial at 1.857143.
        (7, 1.5, 3, 6, [(3, 0.857143), (6, 1.857143)],
         [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]),
    ],
)
def test_fractional_epoch_stop_keeps_the_authorized_tail_epochs(
    monkeypatch, microbatches_per_epoch, epochs, stop_at, expect_steps,
    expect_epoch_end, expect_visits,
):
    # A should_epoch_stop callback shrinks the remaining budget from a CONCEPTUAL
    # horizon of total micro-batches. That horizon truncated num_train_epochs
    # (int(1.5) == 1), so a fractional run that stopped inside its first epoch saw
    # "one pass" as the whole run: the remaining budget went to zero and training
    # ended at the stop instead of continuing into the epochs num_train_epochs had
    # authorized. transformers sizes its epoch loop with
    # num_train_epochs = ceil(args.num_train_epochs) (set_initial_training_values)
    # and stops on max_steps = ceil(args.num_train_epochs * num_update_steps_per_
    # epoch), so the horizon must ceil and the shrunk budget must be clamped.
    # Golden values measured by running transformers.Trainer (5.14.1 and 4.57.6)
    # on the matching row counts.
    _patch_value_and_grad_with_aux(monkeypatch)

    class StopEpochAtStep:
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == stop_at:
                control.should_epoch_stop = True
            return control

    spy = _EpochScheduleSpy()
    trainer, batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=microbatches_per_epoch,
        grad_accum=2, epochs=epochs, callbacks=[spy, StopEpochAtStep()],
    )
    trainer.train()

    assert trainer._global_step == expect_steps
    assert spy.epoch_end == expect_epoch_end
    assert batches.visits == expect_visits


def test_non_padding_input_token_mode_skips_padded_positions(monkeypatch):
    # include_num_input_tokens_seen="non_padding" is a distinct COUNTING MODE, not
    # just another way to enable counting: HF counts the attention mask (falling
    # back to a pad-token comparison, then to every position) instead of the padded
    # tensor's numel. The MLX loop honored the gate but always counted numel, so a
    # "non_padding" run overcounted by the whole padded fraction, and any callback
    # enforcing a token budget or reporting throughput off
    # state.num_input_tokens_seen inherited that error.
    import inspect
    import tempfile

    from transformers import Trainer, TrainerCallback

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import FiniteTextBatchPlan, _FiniteTextRow

    # 1. Pin the expectation to the installed transformers, whose "non_padding"
    #    branch sums the attention mask. Same code in 4.57.x (inline in
    #    _inner_training_loop) and 5.x (Trainer._track_num_input_tokens).
    hf_owner = getattr(Trainer, "_track_num_input_tokens", None)
    hf_src = " ".join(
        inspect.getsource(
            hf_owner if hf_owner is not None else Trainer._inner_training_loop
        ).split()
    )
    branch_at = hf_src.index('include_num_input_tokens_seen == "non_padding"')
    branch = hf_src[branch_at:branch_at + 700]
    assert (
        'if "attention_mask" in inputs: input_tokens = inputs["attention_mask"].sum()'
        in branch
    )
    assert "pad_token_id" in branch
    assert "numel()" in branch

    # 2. Two micro-batches, each two rows of DIFFERENT length, so the plan pads:
    #    widths 10 and 9, so numel is 2*10 + 2*9 = 38 but only 10+4 + 9+3 = 26
    #    positions are real. Padding is unavoidable in any real batched run.
    row_lengths = (10, 4, 9, 3)
    schedule = ((0, 1), (2, 3))
    padded_numel = 2 * 10 + 2 * 9
    non_padding = sum(row_lengths)
    assert (padded_numel, non_padding) == (38, 26)

    _patch_value_and_grad_with_aux(monkeypatch)

    class TokenSpy(TrainerCallback):
        def __init__(self):
            self.per_step = []

        def on_step_end(self, args, state, control, **kwargs):
            self.per_step.append(int(state.num_input_tokens_seen))
            return control

    def run(flag):
        spy = TokenSpy()
        args = MLXTrainingConfig(
            max_steps=len(schedule),
            gradient_accumulation_steps=1,
            logging_steps=len(schedule),
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            cast_norm_output_to_input_dtype=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            disable_memory_limits=True,
            output_dir=tempfile.mkdtemp(),
        )
        args.include_num_input_tokens_seen = flag
        trainer = MLXTrainer(
            _tiny_lm_for_loop_tests(),
            types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
            [],
            args=args,
            callbacks=[spy],
        )
        trainer._batches = FiniteTextBatchPlan(
            tuple(
                _FiniteTextRow(
                    tuple(range(1, length + 1)),
                    offset=1,
                    labels=tuple(range(1, length + 1)),
                )
                for length in row_lengths
            ),
            schedule,
            max_seq_length=64,
            pad_id=99,
        )
        trainer._build_optimizer = _frozen_optimizer()
        trainer.save_model = lambda *_a, **_kw: None
        trainer.train()
        return spy.per_step, int(trainer.state.num_input_tokens_seen)

    # "all"/True keep counting every forwarded position (unchanged behavior).
    for every in ("all", True):
        assert run(every) == ([20, 38], padded_numel), every
    # "non_padding" drops the padded positions, matching real transformers.Trainer
    # on the equivalent torch batches.
    assert run("non_padding") == ([14, 26], non_padding)


def test_mlx_batch_input_token_count_non_padding_ladder():
    # The helper mirrors HF's "non_padding" ladder: attention mask first, then the
    # text tuple batch's lengths column (its exclusive real-token end, i.e. that
    # batch's attention mask), then a pad-token comparison, then every position.
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import _mlx_batch_input_token_count

    ids = mx.array([[1, 2, 3, 99], [4, 5, 99, 99]], dtype=mx.int32)
    mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=mx.int32)
    lengths = mx.array([[1, 3], [1, 2]], dtype=mx.int32)

    # VLM dict batch: the processor's attention mask wins.
    dict_batch = {"input_ids": ids, "attention_mask": mask}
    assert _mlx_batch_input_token_count(dict_batch) == 8
    assert _mlx_batch_input_token_count(dict_batch, mode="all") == 8
    assert _mlx_batch_input_token_count(dict_batch, mode="non_padding") == 5

    # Text tuple batch: no attention mask, so lengths[:, 1] is summed.
    tuple_batch = (ids, lengths, mx.zeros((2, 4), dtype=mx.int32))
    assert _mlx_batch_input_token_count(tuple_batch) == 8
    assert _mlx_batch_input_token_count(tuple_batch, mode="non_padding") == 5

    # Neither: the pad id is used when known, else every position (HF's warning
    # path). 99 appears only as padding here, so both rungs agree with the mask.
    bare = {"input_ids": ids}
    assert _mlx_batch_input_token_count(
        bare, mode="non_padding", pad_token_id=99,
    ) == 5
    assert _mlx_batch_input_token_count(bare, mode="non_padding") == 8

    # A tuple whose second element is not a (B, 2) lengths array falls through.
    assert _mlx_batch_input_token_count(
        (ids, None, None), mode="non_padding", pad_token_id=99,
    ) == 5
    assert _mlx_batch_input_token_count((ids,), mode="non_padding") == 8

    # No input ids at all still degrades to 0 rather than raising, in every mode.
    assert _mlx_batch_input_token_count(None, mode="non_padding") == 0
    assert _mlx_batch_input_token_count(
        {"pixel_values": mx.zeros((2, 3))}, mode="non_padding",
    ) == 0


@pytest.mark.parametrize("interrupt", [SystemExit, KeyboardInterrupt])
def test_callback_interrupt_joins_the_ddp_failure_consensus(monkeypatch, interrupt):
    # Regression for "Route callback interrupts through DDP failure consensus".
    # _fire captured only `Exception`, so a callback raising a non-Exception
    # BaseException (KeyboardInterrupt from a Ctrl-C landing inside a callback,
    # or SystemExit) unwound that rank WITHOUT entering the failure consensus,
    # while every peer entered it and blocked there forever: MLX collectives have
    # no timeout and block in C holding the GIL, so the peers cannot even be
    # signalled out. Verified on a real two-rank `mlx.launch --backend ring` run:
    # pre-fix the peer was still parked in _distributed_status_mask 25 s later and
    # needed SIGKILL. Every other consensus call site in this file (batch fetch,
    # evaluation, optimizer step, checkpoint, best-model restore) already captures
    # BaseException, and _raise_distributed_failure_from_any already re-raises a
    # non-Exception unwrapped without mutating trainer state -- that branch was
    # simply unreachable from _fire.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    # Homogeneous world of 2: the all-sum doubles the local value, so
    # _distributed_any_flag(True) aborts and _distributed_any_flag(False) does not.
    monkeypatch.setattr(
        trainer_mod.mx.distributed, "all_sum",
        lambda value, group=None, stream=None: value * mx.array(2, dtype=value.dtype),
    )
    monkeypatch.setattr(
        trainer_mod.nn, "average_gradients", lambda grad, group=None, **kw: grad,
    )

    class InterruptAtSecondStep:
        def __init__(self, raising):
            self.raising = raising
            self.calls = 0

        def on_step_end(self, args, state, control, **kwargs):
            self.calls += 1
            if self.raising and self.calls == 2:
                raise interrupt("callback interrupt")
            return control

    def run(raising):
        """Return (consensus contexts joined, outcome, stop_requested)."""
        contexts = []
        original = MLXTrainer._raise_distributed_failure

        def recording(self, failed, context, exc=None):
            contexts.append(context)
            return original(self, failed, context, exc)

        monkeypatch.setattr(MLXTrainer, "_raise_distributed_failure", recording)

        def _pinned_ensure_distributed(self):
            self._distributed_world = object()
            self._distributed_rank = 0
            self._distributed_world_size = 2
            self._distributed_is_main_process = True
            self._distributed_initialized = True
            return self._distributed_world

        monkeypatch.setattr(
            MLXTrainer, "_ensure_distributed", _pinned_ensure_distributed,
        )
        args = MLXTrainingConfig(
            max_steps=4,
            gradient_accumulation_steps=1,
            logging_steps=2,
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
            callbacks=[InterruptAtSecondStep(raising)],
        )
        trainer._batches = _make_shape_guard_text_plan((8, 8, 8, 8))
        trainer.save_model = lambda *_a, **_kw: None
        trainer._build_optimizer = _frozen_optimizer()
        outcome = None
        try:
            trainer.train()
        except BaseException as exc:  # noqa: BLE001
            outcome = exc
        return contexts, outcome, trainer.stop_requested

    peer_contexts, peer_outcome, _ = run(raising=False)
    raiser_contexts, raiser_outcome, raiser_stop = run(raising=True)

    # The peer runs to completion and reports no failure of its own.
    assert peer_outcome is None
    # The interrupt still reaches the caller unwrapped, exactly as HF's
    # callback_handler.call_event lets it propagate (transformers
    # trainer_callback.py wraps nothing).
    assert type(raiser_outcome) is interrupt
    assert str(raiser_outcome) == "callback interrupt"
    # The interrupt branch must not latch a stop into a reusable trainer.
    assert raiser_stop is False

    # The lockstep invariant the collective actually requires: up to the abort,
    # the interrupted rank took part in EVERY consensus its peer took part in,
    # ending with the on_step_end dispatch that raised. Pre-fix the raiser
    # skipped that last one, and the peer blocked in it with nobody to meet.
    assert raiser_contexts, "the interrupted rank joined no consensus at all"
    assert raiser_contexts[-1] == "on_step_end callback"
    assert raiser_contexts == peer_contexts[:len(raiser_contexts)]
    assert len(peer_contexts) > len(raiser_contexts)


def test_declared_length_streaming_dispatches_callback_epochs(monkeypatch):
    # Regression for "Use known streaming lengths for callback epoch boundaries".
    # A streaming num_train_epochs run whose iterable declares a reliable __len__
    # is supported: _prepare_data resolves _streaming_epoch_batch_count and derives
    # total_steps from it. The loop, though, built epoch metadata only for
    # max_steps streams, so batches_per_epoch stayed None: on_epoch_begin /
    # on_epoch_end never fired and state.epoch stayed None for the whole run, so
    # epoch-based logging/eval/checkpoint callbacks were silently skipped and
    # WandbCallback.on_save in checkpoint mode raised
    # "TypeError: type NoneType doesn't define __round__ method" on
    # round(state.epoch, 2).
    #
    # The expected trace below is what real transformers produces (verified on
    # 5.14.1 AND 4.57.6) for the equivalent run -- a torch IterableDataset with
    # __len__ == 6, per_device_train_batch_size=2, num_train_epochs=2,
    # save_steps=2: len(train_dataloader) == 3, state.max_steps == 6,
    # on_epoch_begin/on_epoch_end twice each, state.epoch walking
    # 0 -> 1/3 -> 2/3 -> 1.0 -> 1.0 -> 4/3 -> 5/3 -> 2.0, and checkpoint aliases
    # epoch_0.67 / epoch_1.33 / epoch_2.0. HF gets there because
    # steps_in_epoch = len(epoch_dataloader) (transformers trainer.py) and an
    # IterableDataset with __len__ gives its dataloader a length.
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class EpochSpy:
        def __init__(self):
            self.events = []

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.events.append(("on_epoch_begin", state.global_step, state.epoch))
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            self.events.append(("on_epoch_end", state.global_step, state.epoch))
            return control

        def on_save(self, args, state, control, **kwargs):
            # Verbatim WandbCallback.on_save with log_model="checkpoint".
            self.events.append(
                ("on_save", state.global_step, f"epoch_{round(state.epoch, 2)}"),
            )
            return control

    rows = [{"text": f"{value} {value + 10} {value + 20}"} for value in range(1, 7)]
    args = MLXTrainingConfig(
        streaming=True,
        max_steps=0,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        completion_only_loss=False,
        dataset_order="sequential",
        logging_steps=1000,
        save_steps=2,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=tempfile.mkdtemp(),
    )
    spy = EpochSpy()
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        _streaming_text_tokenizer(),
        _DeclaredTextRows(rows),
        args=args,
        callbacks=[spy],
    )
    trainer.save_model = lambda *_a, **_kw: None
    trainer._save_checkpoint = lambda *_a, **_kw: None
    trainer._build_optimizer = _frozen_optimizer()
    trainer.train()

    # The stream really is the length-declaring kind this branch is meant to serve,
    # and the run length is unchanged -- only the callback lifecycle was missing.
    assert trainer._streaming_epoch_batch_count == 3
    assert trainer.state.max_steps == 6
    assert trainer.state.num_train_epochs == 2
    # state.epoch is numeric and complete, so round(state.epoch, 2) cannot raise.
    assert trainer.state.epoch == 2.0
    assert trainer.train_dataset.epochs == [0, 1]

    assert spy.events == [
        ("on_epoch_begin", 0, 0.0),
        ("on_save", 2, "epoch_0.67"),
        ("on_epoch_end", 3, 1.0),
        ("on_epoch_begin", 3, 1.0),
        ("on_save", 4, "epoch_1.33"),
        ("on_save", 6, "epoch_2.0"),
        ("on_epoch_end", 6, 2.0),
    ]


def test_declared_length_streaming_honors_should_epoch_stop(monkeypatch):
    # A declared-length streaming run carries a real epoch lifecycle (the test
    # above), so a callback can raise control.should_epoch_stop from it -- but the
    # honoring was gated on `batch_iter is None`, i.e. on materialized batches. The
    # request was therefore dropped: the producer ran the epoch's whole remainder
    # and the next on_epoch_begin silently cleared the flag, so the run was
    # indistinguishable from one that never asked to stop.
    #
    # HF breaks its inner AND outer step loops on should_epoch_stop for any
    # dataloader (transformers _inner_training_loop: the
    # `if self.control.should_epoch_stop or self.control.should_training_stop:
    # break` pair), then rebuilds iter(epoch_dataloader) for the next epoch, so the
    # abandoned tail of the pass is skipped. Draining the producer to the next pass
    # boundary reaches the same place.
    #
    # Golden values from real transformers (5.14.1 AND 4.57.6) on the equivalent
    # run -- IterableDataset with __len__ == 6, per_device_train_batch_size=2
    # (3 micro-batches per pass), gradient_accumulation_steps=1,
    # num_train_epochs=3, save_steps=2, stopping the epoch at global_step 1:
    # state.max_steps 9, 7 optimizer steps, on_epoch_end at
    # (1, 1/3), (4, 2.0), (7, 3.0), and on_save at (2, 4/3), (4, 2.0), (6, 8/3).
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class StopFirstEpochSpy:
        def __init__(self):
            self.epoch_end = []
            self.step_end = []
            self.saves = []

        def on_step_end(self, args, state, control, **kwargs):
            self.step_end.append((state.global_step, round(float(state.epoch), 6)))
            if state.global_step == 1:
                control.should_epoch_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            self.epoch_end.append((state.global_step, round(float(state.epoch), 6)))
            return control

        def on_save(self, args, state, control, **kwargs):
            self.saves.append((state.global_step, round(float(state.epoch), 6)))
            return control

    rows = [{"text": f"{value} {value + 10} {value + 20}"} for value in range(1, 7)]
    args = MLXTrainingConfig(
        streaming=True,
        max_steps=0,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        completion_only_loss=False,
        dataset_order="sequential",
        logging_steps=1000,
        save_steps=2,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=tempfile.mkdtemp(),
    )
    spy = StopFirstEpochSpy()
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        _streaming_text_tokenizer(),
        _DeclaredTextRows(rows),
        args=args,
        callbacks=[spy],
    )
    trainer.save_model = lambda *_a, **_kw: None
    trainer._save_checkpoint = lambda *_a, **_kw: None
    trainer._build_optimizer = _frozen_optimizer()
    trainer.train()

    assert trainer._streaming_epoch_batch_count == 3
    assert trainer.state.max_steps == 9
    # Epoch 1 gave up its last two micro-batches, so the run is one whole epoch's
    # worth of steps shorter than the untruncated 9 -- it did not silently make
    # them up out of the following passes.
    assert trainer._global_step == 7
    assert spy.epoch_end == [(1, 0.333333), (4, 2.0), (7, 3.0)]
    assert spy.step_end == [
        (1, 0.333333), (2, 1.333333), (3, 1.666667), (4, 2.0),
        (5, 2.333333), (6, 2.666667), (7, 3.0),
    ]
    assert spy.saves == [(2, 1.333333), (4, 2.0), (6, 2.666667)]
    # Every epoch after the truncated one begins on a fresh pass: the producer was
    # drained to the pass boundary, so set_epoch still advances once per pass and
    # no epoch starts part way through the source.
    assert trainer.train_dataset.epochs == [0, 1, 2]


def _run_unsized_streaming_epoch_probe(monkeypatch, spy, **strategy_overrides):
    """Drive a length-less streaming max_steps run and return the trainer.

    strategy_overrides are hand-set on trainer.args after construction, the
    route _sync_synthesized_arg preserves for a caller-supplied strategy.
    """
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    rows = [{"text": f"{value} {value + 10} {value + 20}"} for value in range(1, 7)]
    args = MLXTrainingConfig(
        streaming=True,
        max_steps=3,
        num_train_epochs=0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        completion_only_loss=False,
        dataset_order="sequential",
        logging_steps=1000,
        save_steps=1000,
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
        _streaming_text_tokenizer(),
        _CountingTextRows(rows, infinite=True),
        args=args,
        callbacks=[spy],
    )
    for name, value in strategy_overrides.items():
        setattr(trainer.args, name, value)
    trainer.save_model = lambda *_a, **_kw: None
    trainer._save_checkpoint = lambda *_a, **_kw: None
    trainer._build_optimizer = _frozen_optimizer()
    trainer.train()
    return trainer


class _UnsizedStreamEpochSpy:
    def __init__(self):
        self.epoch_begin = []
        self.epoch_end = []
        self.epochs = []
        self.saves = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_begin.append((state.global_step, round(state.epoch, 2)))
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_end.append((state.global_step, round(state.epoch, 2)))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self.epochs.append(round(state.epoch, 2))
        return control

    def on_save(self, args, state, control, **kwargs):
        self.saves.append((state.global_step, round(state.epoch, 2)))
        return control


def test_unsized_streaming_dispatches_one_synthetic_epoch(monkeypatch):
    # Regression: a length-less stream has no dataset boundaries, so the loop
    # kept batches_per_epoch None and fired NO epoch events at all. HF still runs
    # one conceptual epoch over the synthetic horizon
    # steps_in_epoch = max_steps * grad_accum: num_train_epochs = sys.maxsize only
    # means "re-iterate as needed", and the step budget is exhausted inside the
    # first pass of `for epoch in range(...)`, so exactly one on_epoch_begin and
    # one on_epoch_end fire. Suppressing them left every on_epoch_* callback dead
    # and made logging_strategy/eval_strategy/save_strategy="epoch" silently do
    # nothing for the whole run.
    # Goldens measured by running a real transformers.Trainer over an unsized
    # IterableDataset at max_steps=6, grad_accum=1 -- identical on 4.57.6 and
    # 5.14.1: on_epoch_begin (0, 0.0), on_epoch_end (6, 1.0), num_train_epochs
    # sys.maxsize, final state.epoch 1.0.
    spy = _UnsizedStreamEpochSpy()
    trainer = _run_unsized_streaming_epoch_probe(
        monkeypatch, spy, save_strategy="no",
    )

    assert trainer._streaming_epoch_batch_count is None
    assert spy.epoch_begin == [(0, 0.0)], spy.epoch_begin
    assert spy.epoch_end == [(3, 1.0)], spy.epoch_end
    # state.epoch is unchanged, and stays numeric so round(state.epoch, 2) in
    # WandbCallback.on_save still works.
    assert spy.epochs == [0.33, 0.67, 1.0]


def test_unsized_streaming_epoch_strategy_acts_on_the_synthetic_boundary(monkeypatch):
    # The point of dispatching the lifecycle: an "epoch" strategy now gets its
    # boundary action, matching the real transformers.Trainer, which fires
    # on_save and on_evaluate at the close of the same synthetic epoch.
    # "no" is the control -- the interval is far past the run and the strategy
    # is not "steps", so nothing else can produce a save here.
    for strategy, expected in (("no", []), ("epoch", [(3, 1.0)])):
        spy = _UnsizedStreamEpochSpy()
        _run_unsized_streaming_epoch_probe(
            monkeypatch, spy, save_strategy=strategy,
        )
        assert spy.saves == expected, (strategy, spy.saves)


def test_unsized_streaming_epoch_events_do_not_enable_the_producer_drain(monkeypatch):
    # The synthetic horizon drives the epoch LIFECYCLE only. Honoring
    # control.should_epoch_stop skips to the next boundary, and for a length-less
    # stream _honor_epoch_stop_skip has no index to fast-forward: it drains
    # micro-batches out of a producer that cannot replay them, where HF simply
    # rebuilds its iterator. So the two gates stay on batches_per_epoch, and the
    # rows pulled from the source must not change.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert src.count("if batches_per_epoch and _sync_epoch_stop():") == 2, src.count(
        "if batches_per_epoch and _sync_epoch_stop():"
    )
    assert "epoch_event_microbatches and _sync_epoch_stop()" not in src

    class _StopEpochAtStep1(_UnsizedStreamEpochSpy):
        def on_step_end(self, args, state, control, **kwargs):
            super().on_step_end(args, state, control, **kwargs)
            if state.global_step == 1:
                control.should_epoch_stop = True
            return control

    spy = _StopEpochAtStep1()
    trainer = _run_unsized_streaming_epoch_probe(monkeypatch, spy)
    # The callback's request is simply not honorable here, exactly as before, so
    # the source is read straight through and nothing is silently discarded.
    assert trainer.train_dataset.pulls == 6, trainer.train_dataset.pulls
    assert trainer._global_step == 3


def test_no_consensus_site_captures_bare_exception():
    # A rank that captures only Exception skips _raise_distributed_failure on an
    # interrupt while its peers enter and block in it, with no way out: the
    # collective holds the GIL in C so Ctrl-C never reaches Python. Every site
    # feeding a consensus must therefore capture BaseException.
    import inspect
    import re

    import unsloth_zoo.mlx.trainer as trainer_mod

    src = inspect.getsource(trainer_mod).splitlines()
    offenders = []
    for i, line in enumerate(src):
        if "_raise_distributed_failure" not in line or line.lstrip().startswith("def "):
            continue
        for j in range(i, max(0, i - 40), -1):
            m = re.search(r"except (BaseException|Exception)\b", src[j])
            if m:
                if m.group(1) == "Exception":
                    offenders.append((j + 1, src[j].strip(), i + 1))
                break
    assert offenders == [], offenders


def _hf_flow_args(
    *, eval_strategy="no", eval_steps=10 ** 6, eval_delay=0,
    logging_strategy="steps", logging_steps=10 ** 6, logging_first_step=False,
    save_strategy="no", save_steps=10 ** 6,
):
    """Stand in for TrainingArguments when driving DefaultFlowCallback.

    A real TrainingArguments cannot be built on Apple Silicon: pyproject omits
    accelerate there, and the constructor hard-requires accelerate>=1.1.0
    whenever torch is importable. Since that is the platform MLX actually runs
    on, building one would make this file uncollectable for its own users.
    Only the attributes DefaultFlowCallback reads matter, and
    IntervalStrategy/SaveStrategy are str enums, so plain strings compare equal
    to them. test_hf_flow_args_stand_in_matches_real_training_arguments pins
    the equivalence wherever a real one can be constructed.
    """
    return types.SimpleNamespace(
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        eval_delay=eval_delay,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        logging_first_step=logging_first_step,
        save_strategy=save_strategy,
        save_steps=save_steps,
    )


def test_hf_flow_args_stand_in_matches_real_training_arguments():
    # The stand-in above only holds while it agrees with the real object on
    # every field DefaultFlowCallback reads, including the enum normalization
    # TrainingArguments applies in __post_init__. Skipped on the platform that
    # forced the stand-in, which is exactly where it cannot be checked.
    import tempfile

    from transformers.utils import is_accelerate_available

    # The same predicate the constructor itself raises on, so this skips for a
    # too-old accelerate as well as for a missing one.
    if not is_accelerate_available():
        pytest.skip("TrainingArguments needs accelerate, omitted on arm64 macOS")
    from transformers import TrainingArguments

    read_by_flow = (
        "eval_strategy", "eval_delay", "logging_strategy",
        "logging_first_step", "save_strategy",
    )
    # Every strategy on every axis, so the stand-in is pinned for the log and
    # save cadences as well as the eval one.
    for strategy, steps, delay in (
        ("no", 2, 0), ("steps", 2, 0), ("epoch", 2, 0), ("steps", 2, 3),
    ):
        fields = dict(
            eval_strategy=strategy, eval_steps=steps, eval_delay=delay,
            logging_strategy=strategy, logging_steps=steps,
            save_strategy=strategy, save_steps=steps,
        )
        real = TrainingArguments(
            output_dir=tempfile.mkdtemp(), report_to=[], **fields,
        )
        stub = _hf_flow_args(**fields)
        for field in read_by_flow:
            assert getattr(stub, field) == getattr(real, field), (
                strategy, delay, field,
            )
        # Equal as strings is not enough: the flow compares against enum
        # members, so the stub must land on the same side of those tests.
        for action in ("evaluate", "log", "save"):
            assert (
                _flow_fires(
                    stub, total_steps=6, steps_per_epoch=3, action=action,
                )
                == _flow_fires(
                    real, total_steps=6, steps_per_epoch=3, action=action,
                )
            ), (strategy, delay, action)


def _flow_fires(args, *, total_steps, steps_per_epoch, action="evaluate"):
    """Steps DefaultFlowCallback asks to log/evaluate/save, for an args object."""
    from transformers.trainer_callback import (
        DefaultFlowCallback,
        TrainerControl,
        TrainerState,
    )

    flag = f"should_{action}"
    state = TrainerState(
        max_steps=total_steps, eval_steps=args.eval_steps,
        logging_steps=args.logging_steps, save_steps=args.save_steps,
    )
    flow = DefaultFlowCallback()
    fires = []
    for step in range(1, total_steps + 1):
        state.global_step = step
        state.epoch = step / steps_per_epoch
        control = flow.on_step_end(args, state, TrainerControl())
        # HF clears the flag when it runs the action, so a step can only run it
        # once however many hooks asked for it.
        if not getattr(control, flag) and step % steps_per_epoch == 0:
            control = flow.on_epoch_end(args, state, TrainerControl())
        if getattr(control, flag):
            fires.append(step)
    return fires


def _hf_eval_steps_from_default_flow(
    *, total_steps, steps_per_epoch, eval_strategy, eval_steps, eval_delay=0,
):
    """Ask the installed transformers DefaultFlowCallback which steps evaluate.

    Derived from the shipped implementation rather than hardcoded, so the
    expectation tracks the 4.x/5.x differences in DefaultFlowCallback (5.x adds
    a final-step evaluation for the steps strategy) instead of pinning one.
    """
    return _flow_fires(
        _hf_flow_args(
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            eval_delay=eval_delay,
        ),
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
    )


def _run_eval_cadence_probe(monkeypatch, *, eval_steps, **arg_overrides):
    """Run the real loop for 4 steps / 2 epochs and report the eval steps."""
    import tempfile

    from transformers.trainer_callback import DefaultFlowCallback

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    args = MLXTrainingConfig(
        max_steps=4,
        gradient_accumulation_steps=1,
        logging_steps=10 ** 6,
        eval_steps=eval_steps,
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
        eval_dataset=[{"input_ids": [1, 2, 3, 4]}],
        args=args,
        callbacks=[DefaultFlowCallback()],
    )
    # Eval is live, so the bridge synthesized eval_strategy="steps" already; the
    # override below is therefore a genuine caller-supplied strategy, exactly
    # what _sync_synthesized_arg preserves for a real TrainingArguments/
    # SFTConfig or a hand-set override.
    assert trainer.args.eval_strategy == "steps"
    for name, value in arg_overrides.items():
        setattr(trainer.args, name, value)

    trainer._batches = _make_shape_guard_text_plan((10,) * 4)
    trainer._callback_batches_per_epoch = lambda _batches: 2
    trainer._eval_batches_labeled = ["batch-0"]

    evaluated = []

    def _fake_evaluate(batches, loss_fn, is_vlm=False):
        evaluated.append(trainer.state.global_step)
        trainer._last_eval_metrics = {"eval_loss": 1.25, "eval_perplexity": 3.5}
        return 1.25, 3.5

    trainer._evaluate = _fake_evaluate
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None
    trainer.train()
    return evaluated


def test_static_eval_cadence_honors_caller_supplied_eval_strategy(monkeypatch):
    # Regression: the loop's static cadence evaluated purely from eval_steps, so
    # a caller-supplied eval_strategy -- which _sync_synthesized_arg explicitly
    # preserves -- was ignored. eval_strategy="no" still evaluated on the step
    # cadence, and eval_strategy="epoch" evaluated on BOTH the step cadence and
    # DefaultFlowCallback's epoch end, double-counting every epoch for
    # early-stopping and best-model tracking.
    for strategy in ("no", "epoch"):
        expected = _hf_eval_steps_from_default_flow(
            total_steps=4, steps_per_epoch=2,
            eval_strategy=strategy, eval_steps=2,
        )
        got = _run_eval_cadence_probe(
            monkeypatch, eval_steps=2, eval_strategy=strategy,
        )
        assert got == expected, (strategy, got, expected)
    # The steps strategy is unchanged, and is still deduplicated against the
    # identical request DefaultFlowCallback raises on the same step.
    expected = _hf_eval_steps_from_default_flow(
        total_steps=4, steps_per_epoch=2, eval_strategy="steps", eval_steps=2,
    )
    assert expected == [2, 4]
    assert _run_eval_cadence_probe(
        monkeypatch, eval_steps=2, eval_strategy="steps",
    ) == expected


def test_static_eval_cadence_honors_eval_delay(monkeypatch):
    # DefaultFlowCallback gates its step evaluation on
    # `args.eval_delay <= state.global_step`; the loop's own cadence bypassed it
    # entirely and evaluated from step 1.
    expected = _hf_eval_steps_from_default_flow(
        total_steps=4, steps_per_epoch=2,
        eval_strategy="steps", eval_steps=2, eval_delay=3,
    )
    assert expected == [4]
    assert _run_eval_cadence_probe(
        monkeypatch, eval_steps=2, eval_strategy="steps", eval_delay=3,
    ) == expected


def test_static_eval_cadence_keeps_legacy_behaviour_without_a_strategy():
    # An args object that never went through _ensure_callback_args_compat has no
    # eval_strategy at all; the cadence must not silently disable itself.
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = types.SimpleNamespace()
    assert trainer._static_eval_cadence_enabled() is True
    assert trainer._eval_delay_satisfied(1) is True
    # IntervalStrategy is a str Enum, so both spellings normalize.
    from transformers.trainer_utils import IntervalStrategy

    trainer.args = types.SimpleNamespace(eval_strategy=IntervalStrategy.STEPS)
    assert trainer._static_eval_cadence_enabled() is True
    trainer.args = types.SimpleNamespace(eval_strategy=IntervalStrategy.EPOCH)
    assert trainer._static_eval_cadence_enabled() is False
    trainer.args = types.SimpleNamespace(eval_strategy="steps", eval_delay="oops")
    assert trainer._eval_delay_satisfied(0) is True


def test_static_log_and_save_cadences_keep_legacy_behaviour_without_a_strategy():
    # Same contract as the eval cadence for the log and save axes: a missing
    # field means the args object never went through _ensure_callback_args_compat
    # and must keep the interval-only cadence rather than going silent.
    from transformers.trainer_utils import IntervalStrategy, SaveStrategy

    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = types.SimpleNamespace()
    assert trainer._static_log_cadence_enabled() is True
    assert trainer._static_save_cadence_enabled() is True

    for member, expected in (
        (IntervalStrategy.STEPS, True),
        (IntervalStrategy.EPOCH, False),
        (IntervalStrategy.NO, False),
    ):
        trainer.args = types.SimpleNamespace(logging_strategy=member)
        assert trainer._static_log_cadence_enabled() is expected, member
    for member, expected in (
        (SaveStrategy.STEPS, True),
        (SaveStrategy.EPOCH, False),
        (SaveStrategy.NO, False),
    ):
        trainer.args = types.SimpleNamespace(save_strategy=member)
        assert trainer._static_save_cadence_enabled() is expected, member


def _run_log_save_cadence_probe(
    monkeypatch, *, logging_steps, save_steps, total_steps=6,
    steps_per_epoch=3, eval_steps=None, with_flow=True, extra_callbacks=(),
    eval_losses=None, **arg_overrides,
):
    """Run the real loop and report the steps that logged, saved and evaluated.

    6 steps over 2 epochs of 3, so an interval of 2 lands off the epoch
    boundaries and the steps/epoch/no strategies are told apart.

    Passing eval_steps attaches an eval dataset and a stub _evaluate, so the
    third cadence axis runs on the same 6-step/2-epoch geometry as the other
    two; leaving it None keeps the run eval-free.

    with_flow=False drops DefaultFlowCallback from the callback list.
    MLXTrainer does not install one itself (transformers' Trainer always does),
    so that is the configuration a caller who passes their own callbacks
    actually gets, and each cadence must hold in both.
    """
    import tempfile

    from transformers.trainer_callback import (
        DefaultFlowCallback,
        ProgressCallback,
    )

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    out_dir = tempfile.mkdtemp()
    args = MLXTrainingConfig(
        max_steps=total_steps,
        gradient_accumulation_steps=1,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_steps is not None else 10 ** 6,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=out_dir,
    )
    logged, saved, summaries, evaluated = [], [], [], []

    class _Spy:
        def on_log(self, args, state, control, logs=None, **kwargs):
            payload = logs or {}
            # The run summary train() dispatches before on_train_end is not a
            # cadence log, and an evaluation dispatches its metrics through
            # on_log too, so keep all three apart.
            if "train_runtime" in payload:
                summaries.append((state.global_step, dict(payload)))
            elif "eval_loss" in payload:
                pass
            else:
                logged.append(state.global_step)
            return control

        def on_save(self, args, state, control, **kwargs):
            saved.append(state.global_step)
            return control

        def on_evaluate(self, args, state, control, **kwargs):
            evaluated.append(state.global_step)
            return control

    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [],
        args=args,
        eval_dataset=(
            [{"input_ids": [1, 2, 3, 4]}] if eval_steps is not None else None
        ),
        # ProgressCallback is a real stock consumer of on_log, so the payloads
        # below are exercised through shipped transformers code, not just a spy.
        callbacks=(
            ([DefaultFlowCallback()] if with_flow else [])
            + [ProgressCallback(), _Spy()] + list(extra_callbacks)
        ),
    )
    # Both strategies were synthesized from the positive MLX intervals, so an
    # override below is a genuine caller-supplied strategy -- exactly what
    # _sync_synthesized_arg preserves for a hand-set field.
    assert trainer.args.logging_strategy == "steps"
    assert trainer.args.save_strategy == "steps"
    if eval_steps is not None:
        assert trainer.args.eval_strategy == "steps"
    for name, value in arg_overrides.items():
        setattr(trainer.args, name, value)

    trainer._batches = _make_shape_guard_text_plan((10,) * total_steps)
    trainer._callback_batches_per_epoch = lambda _batches: steps_per_epoch
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None
    if eval_steps is not None:
        trainer._eval_batches_labeled = ["batch-0"]
        # eval_losses drives a controlled improve/worsen sequence, which is what
        # the "best" save strategy keys on; the default is a flat metric.
        losses = list(eval_losses or ())

        def _fake_evaluate(batches, loss_fn, is_vlm=False):
            loss = losses.pop(0) if losses else 1.25
            trainer._last_eval_metrics = {
                "eval_loss": loss, "eval_perplexity": 3.5,
            }
            return loss, 3.5

        trainer._evaluate = _fake_evaluate
    result = trainer.train()
    checkpoints = sorted(
        int(name.split("-")[1])
        for name in os.listdir(out_dir) if name.startswith("checkpoint-")
    )
    return {
        "logged": logged,
        "saved": saved,
        "evaluated": evaluated,
        "checkpoints": checkpoints,
        "summaries": summaries,
        "result": result,
        "state": trainer.state,
    }


def _hf_flow_steps(action, *, total_steps=6, steps_per_epoch=3, **fields):
    """Ask the installed DefaultFlowCallback which steps run `action`."""
    return _flow_fires(
        _hf_flow_args(**fields),
        total_steps=total_steps, steps_per_epoch=steps_per_epoch,
        action=action,
    )


def test_static_save_cadence_honors_caller_supplied_save_strategy(monkeypatch):
    # Regression: the loop checkpointed purely from save_steps, so a
    # caller-supplied save_strategy -- which _sync_synthesized_arg explicitly
    # preserves -- was ignored. save_strategy="no" still wrote checkpoint-N and
    # dispatched on_save on the step cadence (with HF's default save_steps=500
    # that is a surprise checkpoint every 500 steps), and save_strategy="epoch"
    # wrote the step checkpoints on top of the epoch ones, doubling on_save at
    # every boundary and advertising checkpoints the strategy never asked for.
    for strategy in ("no", "epoch", "steps"):
        expected = _hf_flow_steps(
            "save", save_strategy=strategy, save_steps=2,
        )
        probe = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=2,
            save_strategy=strategy,
        )
        assert probe["saved"] == expected, (strategy, probe["saved"], expected)
        # on_save must advertise exactly the checkpoints that exist on disk.
        assert probe["checkpoints"] == sorted(set(expected)), strategy
    # The three strategies really are distinguishable at these parameters, so
    # the assertions above cannot pass by coincidence.
    assert _hf_flow_steps("save", save_strategy="no", save_steps=2) == []
    assert _hf_flow_steps("save", save_strategy="epoch", save_steps=2) == [3, 6]
    assert _hf_flow_steps("save", save_strategy="steps", save_steps=2) == [2, 4, 6]


def test_static_log_cadence_honors_caller_supplied_logging_strategy(monkeypatch):
    # Regression: the loop logged purely from logging_steps, so a
    # caller-supplied logging_strategy was ignored. logging_strategy="no" still
    # emitted a log every logging_steps, and "epoch" logged on both the step
    # cadence and DefaultFlowCallback's epoch end.
    for strategy in ("no", "epoch", "steps"):
        expected = _hf_flow_steps(
            "log", logging_strategy=strategy, logging_steps=2,
        )
        # The loop always flushes the run's last window at the final step so the
        # returned train_loss covers it; HF folds the same trailing window into
        # its returned train_loss silently. That flush is the one cadence entry
        # MLX adds, and it lands on the final step whatever the strategy.
        expected = sorted(set(expected) | {6})
        probe = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=2, save_steps=10 ** 6,
            logging_strategy=strategy,
        )
        assert probe["logged"] == expected, (
            strategy, probe["logged"], expected,
        )
    assert _hf_flow_steps("log", logging_strategy="no", logging_steps=2) == []
    assert _hf_flow_steps("log", logging_strategy="epoch", logging_steps=2) == [3, 6]
    assert _hf_flow_steps("log", logging_strategy="steps", logging_steps=2) == [2, 4, 6]


# The three cadence axes, each as (DefaultFlowCallback action name, the args
# field that selects its strategy, the probe key recording where it fired).
_CADENCE_AXES = (
    ("log", "logging_strategy", "logged"),
    ("save", "save_strategy", "saved"),
    ("evaluate", "eval_strategy", "evaluated"),
)


_CADENCE_INTERVAL_FIELD = {
    "log": "logging_steps", "save": "save_steps", "evaluate": "eval_steps",
}


def _run_one_axis_cadence_probe(
    monkeypatch, action, strategy, *, with_flow, interval=2, **extra,
):
    """Probe one cadence axis at one strategy, with/without the HF flow."""
    field = dict((a, f) for a, f, _ in _CADENCE_AXES)[action]
    key = dict((a, k) for a, _, k in _CADENCE_AXES)[action]
    # The probed axis gets `interval` on a 6-step/3-per-epoch run, so "steps"
    # and "epoch" (3, 6) cannot be confused. The other two axes keep an interval
    # far past the run so they never interfere.
    # Both intervals are swept: 2 divides 6, 4 does not, and
    # DefaultFlowCallback.on_step_end forces a final-step save (4.x and 5.x)
    # and, on 5.x, a final-step evaluation once state.global_step reaches
    # state.max_steps. The divisible interval hides that tail entirely, which is
    # why the loop's missing copy of it survived a matrix that only swept 2.
    kwargs = dict(logging_steps=10 ** 6, save_steps=10 ** 6, with_flow=with_flow)
    kwargs[_CADENCE_INTERVAL_FIELD[action]] = interval
    kwargs[field] = strategy
    kwargs.update(extra)
    return _run_log_save_cadence_probe(monkeypatch, **kwargs)[key]


def _hf_cadence_reference(action, strategy, interval=2, **fields):
    """HF's cadence for one axis, from the installed DefaultFlowCallback."""
    field = dict((a, f) for a, f, _ in _CADENCE_AXES)[action]
    fields = {
        field: strategy, _CADENCE_INTERVAL_FIELD[action]: interval, **fields,
    }
    expected = _hf_flow_steps(action, **fields)
    if action == "log":
        # The loop always flushes the run's last window at the final step so the
        # returned train_loss covers it, whatever the strategy (see
        # test_static_log_cadence_honors_caller_supplied_logging_strategy).
        expected = sorted(set(expected) | {6})
    return expected


def test_epoch_cadence_fires_without_a_default_flow_callback(monkeypatch):
    # Regression: gating the loop's static log/save/eval interval on the
    # caller-supplied strategy is only half of DefaultFlowCallback.on_step_end.
    # The other half is on_epoch_end, and MLXTrainer -- unlike transformers'
    # Trainer, which always installs DefaultFlowCallback -- installs no flow
    # callback of its own. So a caller who hand-sets "epoch" and passes their own
    # callbacks got the step cadence switched off with nothing put in its place:
    # no periodic checkpoint, no periodic log and no periodic evaluation for the
    # whole run, which is strictly worse than the wrong-cadence bug being fixed.
    # The loop must raise the epoch action itself, so all three axes match HF
    # whether or not a flow callback happens to be installed.
    # Collected rather than asserted per cell so a failure names every cell that
    # drifted, not just the first.
    # Both a step interval that divides the 6-step budget and one that does not
    # (4), because DefaultFlowCallback's final-step force only shows up in the
    # second -- see test_step_cadence_forces_the_final_step_action.
    mismatches = []
    for action, _field, _key in _CADENCE_AXES:
        for strategy in ("steps", "epoch", "no"):
            for interval in (2, 4):
                expected = _hf_cadence_reference(action, strategy, interval)
                for with_flow in (True, False):
                    got = _run_one_axis_cadence_probe(
                        monkeypatch, action, strategy,
                        with_flow=with_flow, interval=interval,
                    )
                    if got != expected:
                        mismatches.append(
                            (action, strategy, interval,
                             "with_flow" if with_flow else "no_flow",
                             got, expected)
                        )
    assert mismatches == [], mismatches
    # The strategies really are distinguishable at these parameters on every
    # axis, so the assertions above cannot pass by coincidence.
    for action in ("log", "save", "evaluate"):
        assert _hf_cadence_reference(action, "steps") == [2, 4, 6], action
        assert _hf_cadence_reference(action, "epoch") == [3, 6], action
        # Only the log axis carries the loop's extra final-window flush.
        assert _hf_cadence_reference(action, "no") == (
            [6] if action == "log" else []
        ), action
        # And the non-divisible interval keeps the epoch cadence distinct from
        # the steps one, so the added cells test something the old ones did not.
        assert _hf_cadence_reference(action, "epoch", 4) == [3, 6], action
        assert 4 in _hf_cadence_reference(action, "steps", 4), action


def test_step_cadence_forces_the_final_step_action(monkeypatch):
    # Regression: DefaultFlowCallback.on_step_end forces a save once
    # state.global_step reaches state.max_steps whenever save_strategy is
    # "steps" (4.x and 5.x), and 5.x forces an evaluation too when the interval
    # did not already land there. MLXTrainer installs no flow callback, so with
    # an interval that does not divide the budget the run wrote no
    # checkpoint-<max_steps> -- the LAST resumable checkpoint, and the only one
    # carrying optimizer/trainer state, since the unconditional final
    # save_model() writes adapters alone -- and dispatched no on_save for it. On
    # 5.x the final evaluation went missing too, so load_best_model_at_end could
    # restore a stale earlier model and early-stopping/reporting callbacks never
    # saw the run's last metrics. An interval of 4 over 6 steps is exactly that
    # gap; the divisible interval the older matrix swept hid it.
    mismatches = []
    for action in ("save", "evaluate"):
        expected = _hf_cadence_reference(action, "steps", 4)
        for with_flow in (True, False):
            got = _run_one_axis_cadence_probe(
                monkeypatch, action, "steps", with_flow=with_flow, interval=4,
            )
            if got != expected:
                mismatches.append(
                    (action, "with_flow" if with_flow else "no_flow",
                     got, expected)
                )
    assert mismatches == [], mismatches
    # on_save must advertise exactly the checkpoints that exist on disk, and the
    # final one is the point of the fix.
    for with_flow in (True, False):
        probe = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=4,
            save_strategy="steps", with_flow=with_flow,
        )
        assert probe["saved"] == [4, 6], (with_flow, probe["saved"])
        assert probe["checkpoints"] == [4, 6], (with_flow, probe["checkpoints"])
    # The save axis forces the final step on every supported transformers, so
    # this cell pins the shape of the fix and cannot pass by coincidence.
    assert _hf_cadence_reference("save", "steps", 4) == [4, 6]
    # The eval axis is the 4.x/5.x split, so it stays derived: 5.x forces the
    # final evaluation, 4.x has no such block. The loop's own answer must be the
    # installed callback's answer.
    from unsloth_zoo.mlx.trainer import _default_flow_evaluates_final_step

    eval_reference = _hf_cadence_reference("evaluate", "steps", 4)
    assert eval_reference in ([4], [4, 6]), eval_reference
    assert (eval_reference == [4, 6]) is _default_flow_evaluates_final_step()


def test_step_cadence_does_not_force_a_budget_the_run_never_reached(monkeypatch):
    # The forced action is keyed on state.max_steps -- the same fixed field HF's
    # flow tests -- not on "the last step this run happened to execute". A run
    # that a callback stops early, or one whose budget _honor_epoch_stop_skip
    # shrank (epoch-count runs only), ends below max_steps and gets no forced
    # save from HF, because the flow's block never fires. Keying the loop's copy
    # on the live budget would save there anyway and put the with-flow and
    # without-flow cadences back out of step.
    class _StopAtStep4:
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 4:
                control.should_training_stop = True
            return control

    runs = {}
    for with_flow in (True, False):
        runs[with_flow] = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
            save_strategy="steps", with_flow=with_flow,
            extra_callbacks=[_StopAtStep4()],
        )
        # No interval lands inside the run, so any checkpoint here could only be
        # the forced one -- and the truncated end must not force it.
        assert runs[with_flow]["saved"] == [], (
            with_flow, runs[with_flow]["saved"],
        )
        assert runs[with_flow]["checkpoints"] == [], with_flow
        # The run really did stop short of the budget, so the guard is live.
        state = runs[with_flow]["state"]
        assert state.global_step == 4 < state.max_steps, (
            with_flow, state.global_step, state.max_steps,
        )
    # And the same run carried to completion does force it, so the assertions
    # above are about the truncation and not about the fix being inert.
    assert _run_log_save_cadence_probe(
        monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
        save_strategy="steps", with_flow=False,
    )["saved"] == [6]


def test_step_cadence_honors_logging_first_step(monkeypatch):
    # Regression: DefaultFlowCallback.on_step_end raises should_log at
    # state.global_step == 1 when args.logging_first_step, BEFORE it tests
    # logging_strategy -- so "no" and "epoch" log step 1 as well.
    # _ensure_callback_args_compat exposes and preserves the flag, but the
    # loop's own cadence only logged interval multiples plus its final-window
    # flush, so step 1 was silently dropped whenever logging_steps > 1.
    mismatches = []
    for strategy in ("steps", "no", "epoch"):
        expected = _hf_cadence_reference(
            "log", strategy, 4, logging_first_step=True,
        )
        for with_flow in (True, False):
            got = _run_one_axis_cadence_probe(
                monkeypatch, "log", strategy, with_flow=with_flow, interval=4,
                logging_first_step=True,
            )
            if got != expected:
                mismatches.append(
                    (strategy, "with_flow" if with_flow else "no_flow",
                     got, expected)
                )
    assert mismatches == [], mismatches
    # First-step logging is not gated on the strategy, the three strategies stay
    # distinguishable, and the flag is off by default -- so none of the cells
    # above can pass by coincidence. The trailing 6 is the loop's own
    # final-window flush.
    assert _hf_cadence_reference(
        "log", "steps", 4, logging_first_step=True) == [1, 4, 6]
    assert _hf_cadence_reference(
        "log", "no", 4, logging_first_step=True) == [1, 6]
    assert _hf_cadence_reference(
        "log", "epoch", 4, logging_first_step=True) == [1, 3, 6]
    assert _hf_cadence_reference("log", "steps", 4) == [4, 6]


def test_default_flow_final_step_eval_probe_matches_the_installed_flow():
    # The loop asks the shipped DefaultFlowCallback whether it forces a
    # final-step evaluation instead of pinning a transformers version, so the
    # probe has to agree with what that callback actually does. A future release
    # that reads an argument the probe's stand-in lacks falls back to "no forced
    # evaluation"; this is what makes that fallback visible.
    from unsloth_zoo.mlx import trainer as trainer_mod

    trainer_mod._DEFAULT_FLOW_FINAL_STEP_EVAL = None
    probed = trainer_mod._default_flow_evaluates_final_step()
    # 4 does not divide 6, so a fire at 6 can only come from the forced block.
    reference = _hf_flow_steps("evaluate", eval_strategy="steps", eval_steps=4)
    assert reference[:1] == [4], reference
    assert probed is (reference == [4, 6]), (probed, reference)
    # Probed once per process: the loop reads it on every final step.
    assert trainer_mod._DEFAULT_FLOW_FINAL_STEP_EVAL is probed


def test_step_cadence_request_reads_the_same_fields_as_the_static_cadence():
    # Unit contract for the helper the loop calls before every on_step_end.
    from transformers.trainer_callback import TrainerControl

    from unsloth_zoo.mlx.trainer import MLXTrainer

    def _requested(*, global_step, max_steps, eval_steps=0, **args_fields):
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.args = types.SimpleNamespace(**args_fields)
        trainer.state = types.SimpleNamespace(
            global_step=global_step, max_steps=max_steps, eval_steps=eval_steps,
        )
        trainer.control = TrainerControl()
        trainer._request_step_cadence_actions()
        return (
            trainer.control.should_log,
            trainer.control.should_evaluate,
            trainer.control.should_save,
        )

    # Away from step 1 and the final step the helper asks for nothing.
    assert _requested(global_step=3, max_steps=6) == (False, False, False)
    # An unknown budget has no final step to force.
    assert _requested(
        global_step=3, max_steps=0, save_strategy="steps",
    ) == (False, False, False)
    # logging_first_step fires at step 1 whatever the logging strategy, and only
    # when the caller asked for it.
    assert _requested(
        global_step=1, max_steps=6,
        logging_first_step=True, logging_strategy="no",
    )[0] is True
    assert _requested(global_step=1, max_steps=6, logging_strategy="no")[0] is False
    # The final save follows save_strategy, and only "steps" -- HF leaves the
    # epoch strategy to on_epoch_end and "no" to nobody.
    for strategy, expected in (("steps", True), ("epoch", False), ("no", False)):
        assert _requested(
            global_step=6, max_steps=6, save_strategy=strategy,
        )[2] is expected, strategy
    # A 0 eval interval means "never" for the loop's own cadence, so it must not
    # reach HF's unguarded modulo.
    assert _requested(
        global_step=6, max_steps=6, eval_strategy="steps", eval_steps=0,
    )[1] is False
    # An args object that never went through _ensure_callback_args_compat keeps
    # the legacy answer on BOTH halves of the same rule: _static_save_cadence_
    # enabled() says the interval cadence applies, so the final step it belongs
    # to does too. train() always populates the field, so this is unit-only.
    assert _requested(global_step=6, max_steps=6)[2] is True


def test_save_strategy_best_is_decided_in_the_trainer_core_not_the_flow():
    # Derived pin for the two facts the loop's copy of the rule depends on:
    # DefaultFlowCallback never raises SaveStrategy.BEST (it acts on STEPS and
    # EPOCH only), and HF's Trainer core ASSIGNS the decision rather than ORing
    # it, so a non-improving evaluation also clears a save requested elsewhere.
    # If either changes upstream, the loop has to change with it.
    import inspect

    from transformers import Trainer
    from transformers.trainer_callback import DefaultFlowCallback
    from transformers.trainer_utils import SaveStrategy

    assert "best" in [member.value for member in SaveStrategy]
    core = inspect.getsource(Trainer._maybe_log_save_evaluate)
    assert "SaveStrategy.BEST" in core, core
    assert "self.control.should_save = is_new_best_metric" in core, core
    assert "BEST" not in inspect.getsource(DefaultFlowCallback)


def test_save_strategy_best_checkpoints_every_improving_evaluation(monkeypatch):
    # Regression: gating the static save cadence on save_strategy left "best"
    # -- the third SaveStrategy member, shipped since transformers 4.47 -- with
    # no cadence at all, because it is neither "steps" nor "epoch" and
    # DefaultFlowCallback never raises it. Before the gate the run at least
    # checkpointed on the save_steps interval; after it, a caller who hand-set
    # "best" got no resumable checkpoint and no on_save for the whole run, while
    # the adapters-only final save_model() carries no optimizer or trainer state.
    # HF writes a normal checkpoint-<global_step> at every improving evaluation
    # (Trainer._maybe_log_save_evaluate), so mirror that.
    # Evaluations land on steps 2/4/6; losses improve, worsen, improve.
    for with_flow in (True, False):
        probe = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
            eval_steps=2, save_strategy="best",
            metric_for_best_model="eval_loss", greater_is_better=False,
            eval_losses=[1.0, 2.0, 0.5],
        )
        assert probe["evaluated"] == [2, 4, 6], (with_flow, probe["evaluated"])
        assert probe["saved"] == [2, 6], (with_flow, probe["saved"])
        assert probe["checkpoints"] == [2, 6], (with_flow, probe["checkpoints"])
        assert probe["state"].best_metric == 0.5
        assert probe["state"].best_global_step == 6
    # The other order pins that it is the improvement and not the step index
    # doing the work.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
        eval_steps=2, save_strategy="best", with_flow=False,
        metric_for_best_model="eval_loss", greater_is_better=False,
        eval_losses=[1.0, 0.5, 2.0],
    )
    assert probe["saved"] == [2, 4], probe["saved"]
    # And greater_is_better inverts it, so the comparison is not hardcoded.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
        eval_steps=2, save_strategy="best", with_flow=False,
        metric_for_best_model="eval_loss", greater_is_better=True,
        eval_losses=[1.0, 0.5, 2.0],
    )
    assert probe["saved"] == [2, 6], probe["saved"]


def test_save_strategy_best_clears_a_save_requested_at_an_eval_step(monkeypatch):
    # HF ASSIGNS `control.should_save = is_new_best_metric` under "best", so an
    # evaluation that did not improve also CLEARS a save another source
    # requested for that same step -- an OR would checkpoint a worse model and
    # advertise it through on_save. A step with no evaluation is untouched,
    # because HF's assignment sits inside its `if control.should_evaluate`.
    class _ForceSaveAt:
        def __init__(self, *steps):
            self.steps = set(steps)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in self.steps:
                control.should_save = True
            return control

    # Step 4 evaluates and gets worse: the request is cleared.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
        eval_steps=2, save_strategy="best", with_flow=False,
        metric_for_best_model="eval_loss", greater_is_better=False,
        eval_losses=[1.0, 2.0, 3.0], extra_callbacks=[_ForceSaveAt(4)],
    )
    assert probe["saved"] == [2], probe["saved"]
    # Step 3 does not evaluate, so the request stands.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
        eval_steps=2, save_strategy="best", with_flow=False,
        metric_for_best_model="eval_loss", greater_is_better=False,
        eval_losses=[1.0, 2.0, 3.0], extra_callbacks=[_ForceSaveAt(3)],
    )
    assert probe["saved"] == [2, 3], probe["saved"]


def test_save_strategy_best_without_a_usable_metric_writes_nothing(monkeypatch):
    # An unresolvable metric_for_best_model means _update_callback_best_metric
    # reports no improvement, which is what HF's _determine_best_metric returns
    # too (its whole body is under `if args.metric_for_best_model is not None`),
    # so nothing is saved rather than everything. HF's Trainer.__init__ rejects
    # "best" without a metric outright; MLXTrainer must at least not checkpoint
    # indiscriminately.
    for metric in (None, "does_not_exist"):
        probe = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
            eval_steps=2, save_strategy="best", with_flow=False,
            metric_for_best_model=metric, eval_losses=[1.0, 0.5, 0.25],
        )
        assert probe["evaluated"] == [2, 4, 6], (metric, probe["evaluated"])
        assert probe["saved"] == [], (metric, probe["saved"])
        assert probe["checkpoints"] == [], (metric, probe["checkpoints"])
    # MLXTrainingConfig defaults metric_for_best_model to "eval_loss", so the
    # strategy does work out of the box -- the empty results above are about the
    # metric being unusable, not about "best" being inert.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
        eval_steps=2, save_strategy="best", with_flow=False,
        eval_losses=[1.0, 0.5, 0.25],
    )
    assert probe["saved"] == [2, 4, 6], probe["saved"]


def test_best_save_strategy_helper_normalizes_like_the_other_cadences():
    # Same str-Enum normalization contract as _static_save_cadence_enabled, and
    # a missing field is not "best".
    from transformers.trainer_utils import SaveStrategy

    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.args = types.SimpleNamespace()
    assert trainer._best_save_strategy_enabled() is False
    for member, expected in (
        (SaveStrategy.BEST, True), ("best", True), ("BEST", True),
        (SaveStrategy.STEPS, False), (SaveStrategy.EPOCH, False),
        (SaveStrategy.NO, False),
    ):
        trainer.args = types.SimpleNamespace(save_strategy=member)
        assert trainer._best_save_strategy_enabled() is expected, member
    # And "best" must not switch on either of the other two cadences.
    trainer.args = types.SimpleNamespace(save_strategy="best")
    assert trainer._static_save_cadence_enabled() is False
    assert trainer._epoch_cadence_enabled("save_strategy") is False


def test_epoch_cadence_request_is_deduplicated_against_default_flow(monkeypatch):
    # The loop's epoch request and DefaultFlowCallback's on_epoch_end ask for the
    # same action on the same step, so a second mechanism (running the action
    # directly at the boundary) would double-fire it: two on_save events and two
    # writes of the same checkpoint-N, a duplicate eval_loss in log_history that
    # EarlyStoppingCallback would count twice. Both go through the one
    # control.should_* request the loop already clears when it runs the action,
    # which makes the pair idempotent.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=2, save_steps=2, eval_steps=2,
        logging_strategy="epoch", save_strategy="epoch", eval_strategy="epoch",
    )
    assert probe["saved"] == [3, 6], probe["saved"]
    assert probe["checkpoints"] == [3, 6], probe["checkpoints"]
    assert probe["evaluated"] == [3, 6], probe["evaluated"]
    assert probe["logged"] == [3, 6], probe["logged"]
    # One evaluation record per boundary in the history the callbacks read.
    history = list(probe["state"].log_history)
    assert [
        entry["step"] for entry in history if "eval_loss" in entry
    ] == [3, 6], history


def test_epoch_cadence_closes_a_truncated_final_epoch(monkeypatch):
    # HF fires on_epoch_end for a truncated final epoch after its inner step
    # loop breaks and feeds the result straight into _maybe_log_save_evaluate,
    # so an "epoch" strategy still gets that boundary's action. 5 steps over
    # epochs of 3 leaves exactly that shape: a natural boundary at 3 and a
    # truncated close at 5. Raising the request only at natural boundaries would
    # drop the run's LAST checkpoint, the one a caller most wants.
    for action, field, key, interval in (
        ("log", "logging_strategy", "logged", "logging_steps"),
        ("save", "save_strategy", "saved", "save_steps"),
        ("evaluate", "eval_strategy", "evaluated", "eval_steps"),
    ):
        for with_flow in (True, False):
            kwargs = dict(
                logging_steps=10 ** 6, save_steps=10 ** 6,
                total_steps=5, steps_per_epoch=3, with_flow=with_flow,
            )
            kwargs[interval] = 2
            if action == "evaluate":
                kwargs["eval_steps"] = 2
            kwargs[field] = "epoch"
            probe = _run_log_save_cadence_probe(monkeypatch, **kwargs)
            assert probe[key] == [3, 5], (action, with_flow, probe[key])
            if action == "save":
                assert probe["checkpoints"] == [3, 5], with_flow


def test_epoch_cadence_closes_a_callback_stopped_epoch(monkeypatch):
    # The third on_epoch_end site: a callback that sets control.should_epoch_stop
    # mid-epoch. HF breaks its inner step loop, fires on_epoch_end and feeds
    # _maybe_log_save_evaluate, so its flow raises the epoch action for the
    # epoch the callback just closed -- at the fractional epoch, not a snapped
    # one. The loop must raise it on the same terms, or the checkpoint an
    # early-stopping integration expects at the cut is silently missing.
    class _EndEpochAtStep2:
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_epoch_stop = True
            return control

    for with_flow in (True, False):
        probe = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
            save_strategy="epoch", with_flow=with_flow,
            extra_callbacks=[_EndEpochAtStep2()],
        )
        # Step 2 is the truncated first epoch (2 of 3 micro-batches), and the
        # skip lands the loop on the next boundary.
        assert probe["saved"][0] == 2, (with_flow, probe["saved"])
        assert probe["checkpoints"][0] == 2, (with_flow, probe["checkpoints"])
    # And the two configurations agree exactly, which is the whole point.
    runs = [
        _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
            save_strategy="epoch", with_flow=flow,
            extra_callbacks=[_EndEpochAtStep2()],
        )["saved"]
        for flow in (True, False)
    ]
    assert runs[0] == runs[1], runs


def test_epoch_eval_cadence_without_an_eval_dataset_is_inert(monkeypatch):
    # A hand-set eval_strategy="epoch" on a run with no eval dataset must not
    # raise and must not dispatch a phantom on_evaluate. _run_eval already
    # returns early on empty eval batches (identically on every rank), which is
    # the same thing that happens today when DefaultFlowCallback raises the
    # request, so the request must not gain a divergent guard of its own.
    for with_flow in (True, False):
        probe = _run_log_save_cadence_probe(
            monkeypatch, logging_steps=10 ** 6, save_steps=10 ** 6,
            eval_strategy="epoch", with_flow=with_flow,
        )
        assert probe["evaluated"] == [], (with_flow, probe["evaluated"])


def test_epoch_cadence_request_precedes_every_on_epoch_end_dispatch():
    # Placement is what makes the request both HF-faithful and DDP-safe.
    # HF's DefaultFlowCallback sits at index 0 of the callback list, so it raises
    # the epoch action before any other on_epoch_end callback observes control;
    # requesting before the fire puts the loop in the same position. And every
    # on_epoch_end dispatch is followed by _distributed_sync_control_actions, so
    # the request lands on the near side of the all-reduce that makes log/eval/
    # save rank-consistent -- no rank can enter a collective its peers skip.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    fires = [
        idx for idx in range(len(src))
        if src.startswith('_fire("on_epoch_end")', idx)
    ]
    assert len(fires) == 3, fires
    for fire_at in fires:
        window = src[:fire_at]
        request_at = window.rindex("_request_epoch_cadence_actions()")
        # Nothing but whitespace/comments between the request and the fire.
        between = src[request_at:fire_at]
        assert "_fire(" not in between[len("_request_epoch_cadence_actions()"):], (
            between
        )
        sync_at = src.index("_distributed_sync_control_actions()", fire_at)
        assert fire_at < sync_at
    # The request itself is pure local reads of args/state, both rank-consistent,
    # so it adds no collective of its own and cannot desynchronize the ranks.
    import ast
    import textwrap

    request_src = inspect.getsource(MLXTrainer._request_epoch_cadence_actions)
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(textwrap.dedent(request_src)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not any(name.startswith("_distributed") for name in called), called


def test_epoch_cadence_helpers_keep_legacy_behaviour_without_a_strategy():
    # Same contract as the static cadences: an args object that never went
    # through _ensure_callback_args_compat has no strategy field at all, and must
    # keep the legacy interval-only behaviour rather than gaining an epoch
    # cadence it never asked for.
    from transformers.trainer_callback import TrainerControl
    from transformers.trainer_utils import IntervalStrategy, SaveStrategy

    from unsloth_zoo.mlx.trainer import MLXTrainer

    def _requested(args, epoch=1.0):
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.args = args
        trainer.state = types.SimpleNamespace(epoch=epoch)
        trainer.control = TrainerControl()
        trainer._request_epoch_cadence_actions()
        return (
            trainer.control.should_log,
            trainer.control.should_evaluate,
            trainer.control.should_save,
        )

    assert _requested(types.SimpleNamespace()) == (False, False, False)
    # The enum members and their plain-string spellings normalize the same way.
    for member in (IntervalStrategy.EPOCH, "epoch"):
        assert _requested(types.SimpleNamespace(
            logging_strategy=member, eval_strategy=member,
        )) == (True, True, False), member
    for member in (SaveStrategy.EPOCH, "epoch"):
        assert _requested(
            types.SimpleNamespace(save_strategy=member)
        ) == (False, False, True), member
    for member in (IntervalStrategy.STEPS, IntervalStrategy.NO, "steps", "no"):
        assert _requested(types.SimpleNamespace(
            logging_strategy=member, eval_strategy=member,
            save_strategy=member,
        )) == (False, False, False), member
    # DefaultFlowCallback.on_epoch_end gates its epoch evaluation on
    # `args.eval_delay <= state.epoch` -- the EPOCH, not the step, unlike its
    # on_step_end gate. An unparseable delay falls back to "no delay" so a bad
    # override cannot disable evaluation outright.
    delayed = types.SimpleNamespace(eval_strategy="epoch", eval_delay=2)
    assert _requested(delayed, epoch=1.0) == (False, False, False)
    assert _requested(delayed, epoch=2.0) == (False, True, False)
    assert _requested(
        types.SimpleNamespace(eval_strategy="epoch", eval_delay="oops"),
        epoch=1.0,
    ) == (False, True, False)
    # A state with no epoch yet cannot be compared against the delay at all, so
    # the request is raised rather than dropped. The three loop call sites all
    # set state.epoch first, so this only guards helper paths.
    assert _requested(delayed, epoch=None) == (False, True, False)


def test_final_training_metrics_are_dispatched_before_on_train_end(monkeypatch):
    # Regression: the aggregate train_loss / train_runtime / throughput /
    # completed-step metrics were only assembled into the returned
    # MLXTrainOutput. HF logs the same summary through Trainer.log immediately
    # before on_train_end (trainer.py _finalize_training on 5.x, the tail of
    # _inner_training_loop on 4.x), which is how it reaches state.log_history
    # and on_log -- and so how WandbCallback.on_log promotes train_runtime /
    # train_loss into the run summary. Without it no integration ever saw them.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=2, save_steps=10 ** 6,
    )
    assert len(probe["summaries"]) == 1, probe["summaries"]
    step, payload = probe["summaries"][0]
    assert step == 6
    assert set(payload) == {
        "train_loss", "train_runtime", "train_steps", "total_train_steps",
        "trained_tokens", "train_samples_per_second", "epoch",
    }, sorted(payload)
    # The dispatched payload and the returned output must not drift apart.
    result = probe["result"]
    for key, value in payload.items():
        if key == "epoch":
            continue
        assert result[key] == value, key
    # It lands in state.log_history too, as the tail entry, exactly like HF.
    history = list(probe["state"].log_history)
    assert "train_runtime" in history[-1]
    assert history[-1]["step"] == 6
    assert sum("train_runtime" in entry for entry in history) == 1


def test_final_training_metrics_dispatch_survives_logging_strategy_no(monkeypatch):
    # The summary is not a DefaultFlowCallback cadence, so gating the periodic
    # interval on logging_strategy must not suppress it: HF calls
    # Trainer.log(metrics) unconditionally, and logging_strategy="no" is exactly
    # the case where that is the run's only log.
    probe = _run_log_save_cadence_probe(
        monkeypatch, logging_steps=2, save_steps=10 ** 6,
        logging_strategy="no",
    )
    assert len(probe["summaries"]) == 1, probe["summaries"]
    assert probe["summaries"][0][0] == 6
    assert "train_loss" in probe["summaries"][0][1]


def test_on_prediction_step_fires_once_per_eval_batch():
    # HF dispatches on_prediction_step after every evaluation batch from
    # Trainer.evaluation_loop. Stock ProgressCallback advances its evaluation
    # bar from it, and per-batch evaluation instrumentation hooks into it, so an
    # evaluation that only emits on_log/on_evaluate silently drops both.
    import inspect

    import mlx.core as mx
    from transformers import Trainer

    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _MLXCallbackHandler,
    )

    assert "callback_handler.on_prediction_step" in inspect.getsource(
        Trainer.evaluation_loop
    )

    class PredictionSpy:
        def __init__(self):
            self.seen = []

        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kw):
            self.seen.append(eval_dataloader)
            return control

    trainer = MLXTrainer(
        _MinimalTextModel(), _streaming_text_tokenizer(),
        _CountingTextRows(({"text": "10 1"},)),
        args=MLXTrainingConfig(streaming=True, max_steps=1, max_seq_length=8),
    )
    spy = PredictionSpy()
    trainer.callback_handler = _MLXCallbackHandler(
        [spy], model=None, processing_class=None, optimizer=None, lr_scheduler=None,
    )
    batches = ["batch-0", "batch-1", "batch-2"]
    trainer.callback_handler.eval_dataloader = batches

    def _loss_fn(_model, _batch, _lengths, _labels):
        return mx.array(1.0), mx.array(4)

    def _unpack(batch):
        return batch, None, None

    _, ntokens = trainer._evaluate_batch_totals(
        [_unpack(b) for b in batches], _loss_fn,
    )
    assert int(ntokens.item()) == 12
    # One dispatch per evaluation batch, carrying the handler's eval_dataloader
    # so ProgressCallback's has_length(eval_dataloader) guard sees a real total.
    assert spy.seen == [batches, batches, batches]


def test_on_prediction_step_failure_uses_the_eval_consensus_path():
    # A callback that raises on one rank must join the same eval-status
    # consensus as a failed batch, or the peers block in the next collective.
    import inspect

    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._evaluate_batch_totals)
    dispatch = src.index("self._fire_prediction_step()")
    guard = src.index("except BaseException as exc:", src.index("if not failed"))
    status = src.index("self._distributed_eval_status(failed)")
    assert dispatch < guard < status, "dispatch must feed the eval consensus"


def test_eval_dataloader_tracks_the_split_being_evaluated():
    # HF rebuilds its eval_dataloader per split, so on_prediction_step reports
    # the split being consumed; the dict itself has len == split count, which
    # would give ProgressCallback a nonsense bar total.
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _MLXCallbackHandler,
    )

    class PredictionSpy:
        def __init__(self):
            self.seen = []

        def on_prediction_step(self, args, state, control, eval_dataloader=None, **kw):
            self.seen.append(list(eval_dataloader))
            return control

    trainer = MLXTrainer(
        _MinimalTextModel(), _streaming_text_tokenizer(),
        _CountingTextRows(({"text": "10 1"},)),
        args=MLXTrainingConfig(streaming=True, max_steps=1, max_seq_length=8),
    )
    spy = PredictionSpy()
    trainer.callback_handler = _MLXCallbackHandler(
        [spy], model=None, processing_class=None, optimizer=None, lr_scheduler=None,
    )
    splits = {
        "a": [("a0", None, None)],
        "b": [("b0", None, None), ("b1", None, None)],
    }
    trainer.callback_handler.eval_dataloader = splits
    trainer.model = types.SimpleNamespace(
        eval=lambda: None, train=lambda: None,
    )

    def _loss_fn(_model, _batch, _lengths, _labels):
        return mx.array(1.0), mx.array(4)

    trainer._evaluate(splits, _loss_fn)
    assert spy.seen == [splits["a"], splits["b"], splits["b"]]
    # Restored afterwards, so a later single-split eval is not left pointing at
    # the last split.
    assert trainer.callback_handler.eval_dataloader is splits


def test_dict_eval_rebuilds_the_prediction_bar_per_split(monkeypatch):
    # Stock ProgressCallback sizes its evaluation bar from the first
    # on_prediction_step's eval_dataloader and only tears it down in
    # on_evaluate, which MLX fires once for the whole dict. The first split's
    # bar therefore kept counting every later split's batch past its own total:
    # 2/2 climbed to 9/2 across splits of 2, 3 and 4 batches. HF recurses
    # Trainer.evaluate per split, so it rebuilds the bar: 2/2, 3/3, 4/4.
    import mlx.core as mx
    from transformers.trainer_callback import ProgressCallback

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    _patch_value_and_grad_with_aux(monkeypatch)

    class RecordingProgress(ProgressCallback):
        def __init__(self):
            super().__init__()
            self.geometry = []
            self.closing = "<not called>"

        def on_prediction_step(self, args, state, control,
                               eval_dataloader=None, **kwargs):
            output = super().on_prediction_step(
                args, state, control, eval_dataloader=eval_dataloader, **kwargs,
            )
            self.geometry.append(
                (self.prediction_bar.total, self.prediction_bar.n)
            )
            return output

        def on_evaluate(self, args, state, control, **kwargs):
            self.closing = (
                None if self.prediction_bar is None
                else (self.prediction_bar.total, self.prediction_bar.n)
            )
            return super().on_evaluate(args, state, control, **kwargs)

    def make_batch(width):
        ids = mx.array([list(range(1, width + 1))], dtype=mx.int32)
        lengths = mx.array([[0, width - 1]], dtype=mx.int32)
        return (ids, lengths, None)

    out_dir = tempfile.mkdtemp()
    args = MLXTrainingConfig(
        max_steps=2,
        gradient_accumulation_steps=1,
        logging_steps=100,
        eval_steps=2,
        save_steps=0,
        use_cce=False,
        compile=False,
        gradient_checkpointing=False,
        cast_norm_output_to_input_dtype=False,
        max_grad_norm=0.0,
        max_grad_leaf_norm=0.0,
        disable_memory_limits=True,
        output_dir=out_dir,
    )
    bar = RecordingProgress()
    trainer = MLXTrainer(
        _tiny_lm_for_loop_tests(),
        types.SimpleNamespace(pad_token_id=99, eos_token_id=2),
        [{"text": f"row {i}"} for i in range(4)],
        args=args,
        callbacks=[bar],
    )
    trainer._prepare_data = lambda _is_vlm: ([make_batch(10) for _ in range(4)], None)
    sizes = {"a": 2, "b": 3, "c": 4}
    trainer.eval_dataset = {name: [{"input_ids": [1, 2, 3, 4]}] for name in sizes}
    trainer._eval_batches_labeled = {
        name: [make_batch(10) for _ in range(size)]
        for name, size in sizes.items()
    }
    trainer._build_optimizer = _frozen_optimizer()
    trainer.save_model = lambda *_a, **_kw: None
    trainer.train()

    assert bar.geometry == [
        (2, 1), (2, 2),
        (3, 1), (3, 2), (3, 3),
        (4, 1), (4, 2), (4, 3), (4, 4),
    ], bar.geometry
    # No bar ever runs past its own total.
    assert all(seen <= total for total, seen in bar.geometry), bar.geometry
    # The last split's bar is still torn down by on_evaluate, exactly as in HF.
    assert bar.closing == (4, 4), bar.closing
    assert bar.prediction_bar is None
