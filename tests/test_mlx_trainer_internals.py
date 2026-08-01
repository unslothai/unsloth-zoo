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


def test_ddp_vlm_exact_rank_keeps_local_plan_and_neutral_cap_share():
    """An exact rank must not export its catalog size as the shared cap
    (widening budget-bucketed peers) and keeps its local plan."""
    from unsloth_zoo.mlx import shape_guard as sg
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    events = [sg.TextShapeEvent(("vlm",), w, "none") for w in range(10, 50)]
    frontier = sg.build_text_shape_frontier(
        events, compile_scope=sg.FULL_STEP_SCOPE,
    )
    local_plan = sg.select_text_shape_padding_budget(
        frontier, exact_signature_threshold=sg.AUTOMATIC_TEXT_COMPILE_CEILING,
    )
    assert local_plan.report.action == "exact"
    contributed = []
    trainer = object.__new__(MLXTrainer)
    trainer._distributed_initialized = True
    trainer._distributed_world_size = 2
    trainer._distributed_any_flag = lambda _failed: False
    trainer._distributed_max_int = (
        lambda cap: contributed.append(cap) or 14  # peer's budget cap wins
    )

    plan, report, allowed = trainer._coordinate_text_shape_guard(
        local_plan, frontier, local_plan.report, True,
        build_compile_policy(args=MLXTrainingConfig()),
        automatic=True, keep_exact_local=True,
    )

    assert contributed == [1]  # neutral share, never the catalog size
    assert allowed is True and plan is local_plan
    assert report is local_plan.report and report.action == "exact"


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
        # VLM with no resolved compile decision: planning must not run before
        # qualification.
        (True, None, MLXTrainingConfig(), "vlm_compile_unqualified"),
        (False, iter(()), MLXTrainingConfig(), "streaming"),
        # Compiled global-norm clipping with accumulation is eligible, so the
        # guard plans instead of declining.
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

        def train(self, mode=True):
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

        def train(self, mode=True):
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


def _all_masked_vlm_plan():
    # Response-mask filtering drops all-masked rows, so an all-bad checked plan
    # only arises from rows that bypass it (pad slots, prompt/completion rows).
    # Model that directly through the class contract (checker_good=False rows).
    from unsloth_zoo.mlx.utils import FiniteVLMBatchPlan, _FiniteVLMRow

    return FiniteVLMBatchPlan(
        rows=[_FiniteVLMRow({"text": "0"}, False)],
        schedule=((0,),),
        empty_masks=None,
        processor=object(),
        config={},
        max_seq_length=8,
        image_size=None,
    )


def test_check_vlm_all_masked_reduces_counts_across_ranks(monkeypatch):
    # VLM mirror of the text-path mask check: a fully-masked local shard must not
    # raise alone in DDP, since supervision counts are all-summed first.
    import mlx.core as mx

    import unsloth_zoo.mlx.trainer as trainer_mod
    from unsloth_zoo.mlx.trainer import _check_vlm_all_masked

    def fake_all_sum(value, group=None, stream=None):
        return value + mx.array([0, 5], dtype=mx.int32)

    monkeypatch.setattr(trainer_mod.mx.distributed, "all_sum", fake_all_sum)

    _check_vlm_all_masked(
        _all_masked_vlm_plan(), comm_group=object(), world_size=2,
    )


def test_check_vlm_all_masked_single_process_still_raises():
    from unsloth_zoo.mlx.trainer import _check_vlm_all_masked

    with pytest.raises(ZeroDivisionError):
        _check_vlm_all_masked(_all_masked_vlm_plan())


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
    # The marker is set by the position-recording prepare phase.
    prepare_source = inspect.getsource(mlx_utils._vlm_positions_for_compile)
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


def test_qwen3_visual_window_preserves_batched_row_ownership():
    import mlx.core as mx
    import unsloth_zoo.mlx.compile as mc

    masks = mx.array([[0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=mx.bool_)
    embeds = [mx.array([[10], [11], [12], [20], [21], [22]])]

    padded_cache = [types.SimpleNamespace(offset=mx.array([1, -2]), left_padding=mx.array([3, 2]))]
    assert mc._qwen3_cache_offsets(padded_cache, batch_size=2) == ((1, -2), (4, 0))

    packed, positions = mc._pack_qwen3_visual_state(masks, embeds)
    assert packed.shape == (2, 3, 1, 1)
    assert positions.tolist() == [[1, 2, 4], [0, 2, 3]]
    visual_row = dict(
        inputs_embeds=mx.zeros((1, 6, 1)),
        visual_pos_masks=masks[:1],
        _unsloth_qwen3_visual_state=packed[:1],
        _unsloth_qwen3_visual_positions=positions[:1],
    )
    text_row = dict(inputs_embeds=mx.zeros((1, 9, 1)))
    cold = mc._qwen3_prompt_merge_adapter(lambda rows, _ids: rows)
    aligned = cold([visual_row, text_row], [range(6), range(9)])
    assert aligned[0]["_unsloth_qwen3_visual_state"].tolist() == packed[:1].tolist()
    assert aligned[0]["_unsloth_qwen3_visual_positions"].tolist() == [[4, 5, 7]]
    assert aligned[0]["_unsloth_qwen3_visual_width"].tolist() == [9]
    assert aligned[1]["_unsloth_qwen3_visual_state"].shape == (1, 3, 1, 1)
    assert not aligned[1]["_unsloth_qwen3_visual_state"].any().item()
    assert aligned[1]["_unsloth_qwen3_visual_positions"].tolist() == [[-1, -1, -1]]
    assert aligned[1]["visual_pos_masks"].shape == (1, 9)
    assert not aligned[1]["visual_pos_masks"].any().item()
    apc_batch = types.SimpleNamespace(
        _prompt_kwargs={
            "visual_pos_masks": mx.zeros((2, 9), dtype=mx.bool_),
            "_unsloth_qwen3_visual_width": mx.array([6, 9], dtype=mx.int32),
        },
    )
    apc = mc._qwen3_mixed_prompt_batch_adapter(
        lambda _self, _sequences: apc_batch
    )
    assert apc(
        object(),
        [(None, None, None, visual_row), (None, None, None, text_row)],
    )._prompt_kwargs["_unsloth_qwen3_visual_width"].tolist() == [9, 9]
    apc_masks, apc_embeds = mc._qwen3_visual_window(
        mx.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=mx.bool_),
        mx.array([[[[10]], [[0]]], [[[20]], [[21]]]]),
        mx.array([[1, -1], [4, 5]]),
        mask_offsets=(4, 4),
        position_widths=(9, 9),
        window=4,
    )
    assert apc_masks.tolist() == [[0, 0, 0, 0], [1, 1, 0, 0]]
    assert apc_embeds[0].tolist() == [[20], [21]]
    cached_row = dict(
        inputs_embeds=visual_row["inputs_embeds"],
        visual_pos_masks=masks[:1],
    )
    cached = cold([cached_row, visual_row], [range(6)] * 2)[0]
    assert not cached["_unsloth_qwen3_visual_state"].any().item()
    assert cached["_unsloth_qwen3_visual_positions"].tolist() == [[1, 2, 4]]
    assert mc._pad_qwen3_prompt_rows([]) == []
    assert mc._pad_qwen3_prompt_rows([text_row])[0] is text_row
    window_masks, window_embeds = mc._qwen3_visual_window(
        masks, packed, positions, mask_offsets=(1, 1), window=2
    )
    assert window_masks.tolist() == [[1, 1], [0, 1]]
    assert window_embeds[0].tolist() == [[10], [11], [21]]

    def drops_deepstack(self):
        deepstack_visual_embeds = mx.eval([])
        return deepstack_visual_embeds

    assert mc._qwen3_drops_deepstack_after_eval(drops_deepstack)

    language_calls = []

    def native_language(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        if visual_pos_masks.shape[-1] != inputs.shape[-1]:
            start = int(cache[0].offset)
            window = inputs.shape[1]
            n_before = int(visual_pos_masks[:, :start].sum().item())
            n_window = int(
                visual_pos_masks[:, start : start + window].sum().item()
            )
            deepstack_visual_embeds = [
                layer[n_before : n_before + n_window]
                for layer in deepstack_visual_embeds
            ]
            visual_pos_masks = visual_pos_masks[:, start : start + window]
        language_calls.append((visual_pos_masks, deepstack_visual_embeds))

    language_adapter = mc._qwen3_visual_state_adapter(native_language)
    language_adapter(
        object(),
        mx.zeros((1, 4)),
        cache=[types.SimpleNamespace(offset=1, left_padding=3)],
        visual_pos_masks=mx.pad(masks[:1], [(0, 0), (3, 0)])[:, 4:8],
        deepstack_visual_embeds=None,
        _unsloth_qwen3_visual_state=packed[:1],
        _unsloth_qwen3_visual_positions=aligned[0]["_unsloth_qwen3_visual_positions"],
        _unsloth_qwen3_visual_width=aligned[0]["_unsloth_qwen3_visual_width"],
    )
    assert language_calls[0][0].tolist() == [[1, 1, 0, 1]]
    assert language_calls[0][1][0].tolist() == [[10], [11], [12]]


def test_qwen_video_tensor_adapter_is_exact_and_composable():
    from functools import wraps

    import mlx.core as mx
    import unsloth_zoo.mlx.compile as mc

    calls = []

    def legacy(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        if pixel_values is None:
            return ("text", grid_thw, mask)
        calls.append((input_ids, pixel_values, kwargs))
        return pixel_values

    video, image = object(), object()
    adapted = mc._qwen_video_tensor_adapter(legacy)
    assert adapted(object(), 1, pixel_values_videos=video, video_grid_thw=2) is video
    assert adapted(
        object(), 1, image, pixel_values_videos=video, video_grid_thw=2
    ) is image
    expected_kwargs = {"pixel_values_videos": video, "video_grid_thw": 2}
    assert calls == [(1, video, expected_kwargs), (1, image, expected_kwargs)]
    assert adapted.__wrapped__ is legacy
    assert mc._qwen_video_tensor_adapter(adapted) is adapted

    def fixed(self, input_ids=None, pixel_values=None, **kwargs):
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        return pixel_values, video_grid_thw

    assert mc._qwen_video_tensor_adapter(fixed) is fixed

    def unknown(self, input_ids=None, pixel_values=None, **kwargs):
        video_grid_thw = kwargs.get("video_grid_thw", None)
        if pixel_values is None:
            kwargs.pop("pixel_values_videos", None)
            return ("text", video_grid_thw)

    assert mc._qwen_video_tensor_adapter(unknown) is unknown

    def unreachable(self, input_ids=None, pixel_values=None, **kwargs):
        video_grid_thw = kwargs.get("video_grid_thw", None)
        if pixel_values is None:
            return video_grid_thw
            pixel_values = kwargs.get("pixel_values_videos", None)

    assert not mc._qwen_video_contract_matches(unreachable, "fixed")

    def unreachable_guard(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        return grid_thw, mask
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

    assert not mc._qwen_video_contract_matches(unreachable_guard, "fixed")
    assert not mc._qwen_video_contract_matches(unreachable_guard, "legacy")

    @wraps(legacy)
    def keyword_outer(self, *, input_ids=None, pixel_values=None, **kwargs):
        return legacy(
            self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

    composed = mc._qwen_video_tensor_adapter(keyword_outer)
    assert composed(object(), input_ids=3, pixel_values_videos=video) is video


    def staged_legacy(self, input_ids=None, pixel_values=None, **kwargs):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        if pixel_values is None:
            return grid_thw, mask, mx.eval([])

    staged = mc._qwen3_batch_embedding_adapter(
        staged_legacy, replace_visual_inference=True
    )
    assert mc._qwen_video_tensor_adapter(staged) is not staged
    replacing = mc._qwen3_batch_embedding_adapter(
        staged_legacy, fixed, replace_visual_inference=True
    )
    assert mc._qwen_video_tensor_adapter(replacing) is replacing


def test_qwen_installers_bind_exact_release_owners(monkeypatch):
    import unsloth_zoo.mlx.compile as mc

    expected = (
        "mlx_vlm.models.qwen2_vl.qwen2_vl",
        "mlx_vlm.models.qwen2_5_vl.qwen2_5_vl",
        "mlx_vlm.models.qwen3_5.qwen3_5",
        "mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe",
    )
    assert mc._QWEN_VIDEO_TENSOR_OWNER_MODULES == expected
    owners = {
        name: type(name, (), {"get_input_embeddings": lambda self: None})
        for name in expected
    }
    patched, imported = [], []

    def import_module(module_name):
        imported.append(module_name)
        return types.SimpleNamespace(Model=owners[module_name])

    monkeypatch.setattr(mc.importlib, "import_module", import_module)
    monkeypatch.setattr(mc, "_patch_qwen_video_tensor", patched.append)
    mc._install_qwen_video_tensor_patches()
    assert imported == list(expected)
    assert patched == [owners[module_name] for module_name in expected]

    method_names = (
        "__call__", "_deepstack_process", "get_input_embeddings",
        "merge_input_ids_with_image_features", "rot_pos_emb",
        "fast_pos_embed_interpolate", "_build_mixed_prompt_batch",
    )

    def owner(name):
        methods = {
            key: lambda *_args, _name=f"{name}.{key}", **_kwargs: _name
            for key in method_names
        }
        methods["merge_input_ids_with_image_features"] = staticmethod(
            methods["merge_input_ids_with_image_features"]
        )
        return type(name, (), methods)

    dense_model, dense_language = owner("DenseModel"), owner("DenseLanguage")
    moe_model, moe_language = owner("MoeModel"), owner("MoeLanguage")
    cold_batch, apc_batch = owner("ColdBatch"), owner("ApcBatch")
    vision = types.SimpleNamespace(
        Attention=owner("Attention"),
        Qwen3VLMoEVisionBlock=owner("VisionBlock"),
        VisionModel=owner("VisionModel"),
    )
    namespace = types.SimpleNamespace
    generate = namespace(
        _merge_prefill_prompt_kwargs=cold_batch.get_input_embeddings,
        BatchGenerator=cold_batch,
    )
    generate_ar = namespace(
        _merge_prefill_prompt_kwargs=apc_batch.get_input_embeddings,
        BatchGenerator=apc_batch,
    )
    modules = {
        "mlx_vlm.models.qwen3_vl.qwen3_vl": namespace(Model=dense_model),
        "mlx_vlm.models.qwen3_vl.vision": vision,
        "mlx_vlm.models.qwen3_5.qwen3_5": namespace(Model=owner("Qwen35")),
        "mlx_vlm.models.qwen3_vl.language": namespace(
            Qwen3VLModel=owner("DenseBackbone"), LanguageModel=dense_language),
        "mlx_vlm.models.qwen3_vl_moe.qwen3_vl_moe": namespace(Model=moe_model),
        "mlx_vlm.models.qwen3_vl_moe.vision": vision,
        "mlx_vlm.models.qwen3_vl_moe.language": namespace(
            Qwen3VLMoEModel=owner("MoeBackbone"), LanguageModel=moe_language),
        "mlx_vlm.generate": generate,
        "mlx_vlm.generate.ar": generate_ar,
    }
    bindings = (
        (dense_model, "get_input_embeddings", "_unsloth_qwen3_batch_visual_state"),
        (moe_model, "get_input_embeddings", "_unsloth_qwen3_batch_visual_state"),
        (dense_language, "__call__", "_unsloth_qwen3_visual_state"),
        (moe_language, "__call__", "_unsloth_qwen3_visual_state"),
        (cold_batch, "_build_mixed_prompt_batch", "_unsloth_qwen3_mixed_prompt_batch"),
        (apc_batch, "_build_mixed_prompt_batch", "_unsloth_qwen3_mixed_prompt_batch"),
        (generate, "_merge_prefill_prompt_kwargs", "_unsloth_qwen3_prompt_merge"),
        (generate_ar, "_merge_prefill_prompt_kwargs", "_unsloth_qwen3_prompt_merge"),
    )
    originals = [getattr(owner, name) for owner, name, _marker in bindings]
    monkeypatch.setattr(mc.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(mc, "_PATCHED_ARCHES", set())
    monkeypatch.setattr(mc, "_PATCH_BINDINGS", set())
    mc._install_qwen3_family_compile_patches()
    for (owner, name, marker), original in zip(bindings, originals):
        wrapped = getattr(owner, name)
        assert getattr(wrapped, marker)
        assert wrapped.__wrapped__ is original
    called = []
    monkeypatch.setattr(mc, "_PATCHES_INSTALLED", False)
    monkeypatch.setattr(mc, "_PATCHED_PATTERN_BUNDLES", set())
    monkeypatch.setattr(mc, "_install_safe_fused_sdpa_mask_patches", lambda: None)
    monkeypatch.setattr(mc, "list_compile_pattern_bundles", lambda: ())
    monkeypatch.setattr(mc, "_install_qwen_video_tensor_patches", lambda: called.append(True))
    monkeypatch.setattr(mc, "_invalidate_qualification_cache", lambda: None)
    monkeypatch.setattr(mc, "build_compile_qualifications", lambda: {})
    assert mc.install_mlx_compile_patches() == {} and called == [True]


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


def test_legacy_llava_prefill_adapter_is_signature_gated(monkeypatch):
    import unsloth_zoo.mlx.compile as mc

    calls = []

    def strict(self, inputs, cache=None):
        calls.append((inputs, cache))
        return "strict"

    class LegacyLanguage:
        __call__ = strict

    module = types.SimpleNamespace(LanguageModel=LegacyLanguage)
    monkeypatch.setattr(mc.importlib, "import_module", lambda name: module)
    mc._install_legacy_llava_prefill_patch()
    inputs = object()
    assert LegacyLanguage()(inputs, cache="cache", n_to_process=4) == "strict"
    assert calls == [(inputs, "cache")]
    assert LegacyLanguage.__call__.__wrapped__ is strict
    assert mc._legacy_vlm_prefill_kwargs_adapter(LegacyLanguage.__call__) is LegacyLanguage.__call__
    def explicit(self, inputs, n_to_process=None):
        return n_to_process

    @mc.wraps(strict)
    def permissive(self, *args, **kwargs):
        return kwargs
    assert mc._legacy_vlm_prefill_kwargs_adapter(explicit) is explicit
    assert mc._legacy_vlm_prefill_kwargs_adapter(permissive) is permissive
    monkeypatch.setattr(mc.importlib, "import_module", lambda name: (_ for _ in ()).throw(RuntimeError))
    mc._install_legacy_llava_prefill_patch()

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


def test_wandb_watch_is_neutralized_for_the_on_train_begin_dispatch():
    # transformers' WandbCallback.setup runs on the first on_train_begin and,
    # when WANDB_WATCH is gradients/parameters/all, calls wandb.watch(model),
    # which raises TypeError("Expected a pytorch model (torch.nn.Module)") on an
    # mlx Module. The opt-in therefore aborts training during callback setup.
    # Upstream reads the environment variable, so that is the lever.
    import inspect

    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXTrainingConfig,
        _MLXCallbackHandler,
    )

    class WandbCallback:  # the class NAME is what the bridge matches on
        def __init__(self):
            self.saw = "unset"

        def on_train_begin(self, args, state, control, **kwargs):
            self.saw = os.environ.get("WANDB_WATCH")

    def _trainer(callbacks):
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.args = MLXTrainingConfig(max_steps=1)
        trainer.callback_handler = _MLXCallbackHandler(
            callbacks, model=object(), processing_class=None,
            optimizer=None, lr_scheduler=None,
        )
        return trainer

    previous = os.environ.get("WANDB_WATCH")
    try:
        os.environ["WANDB_WATCH"] = "gradients"
        wandb_cb = WandbCallback()
        trainer = _trainer([wandb_cb])
        mode = trainer._suppress_torch_only_wandb_watch()
        assert mode == "gradients"
        trainer.callback_handler.call_event(
            "on_train_begin", trainer.args, object(), object(),
        )
        assert wandb_cb.saw == "false"
        trainer._restore_wandb_watch(mode)
        assert os.environ["WANDB_WATCH"] == "gradients"

        # No WandbCallback in the handler: the environment is left alone.
        assert _trainer([object()])._suppress_torch_only_wandb_watch() is None
        assert os.environ["WANDB_WATCH"] == "gradients"

        # An opt-out is not something to suppress.
        os.environ["WANDB_WATCH"] = "false"
        assert _trainer([WandbCallback()])._suppress_torch_only_wandb_watch() is None
    finally:
        if previous is None:
            os.environ.pop("WANDB_WATCH", None)
        else:
            os.environ["WANDB_WATCH"] = previous

    # The loop wires it around the real dispatch, restoring even if a callback
    # raises.
    source = inspect.getsource(MLXTrainer._train_inner)
    assert "_suppress_torch_only_wandb_watch()" in source
    assert "finally:\n            self._restore_wandb_watch(" in source


def test_best_checkpoint_survives_save_total_limit_rotation(tmp_path):
    # HF's _rotate_checkpoints never deletes best_model_checkpoint. Ours pruned
    # by age alone, so save_total_limit=1 with a best result at step 1 and a worse
    # save at step 2 deleted checkpoint-1 and left callbacks a dangling path.
    from unsloth_zoo.mlx.trainer import _prune_stale_checkpoints

    for step in (1, 2, 3):
        (tmp_path / f"checkpoint-{step}").mkdir()

    # limit 1 keeps the best AND the latest, as HF does by raising its own limit.
    _prune_stale_checkpoints(str(tmp_path), 1, keep_step=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint-1", "checkpoint-3",
    ]
    # No protected step: unchanged behaviour, newest only.
    _prune_stale_checkpoints(str(tmp_path), 1)
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint-3"]

    # Protecting an old best must not stop the limit binding: excluding it from
    # the stale slice instead retained save_total_limit + 1 from then on.
    for step in (1, 2, 3, 4):
        (tmp_path / f"checkpoint-{step}").mkdir(exist_ok=True)
    _prune_stale_checkpoints(str(tmp_path), 2, keep_step=1)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint-1", "checkpoint-4",
    ]


def test_final_metrics_separate_sample_and_token_throughput(monkeypatch):
    # train_samples_per_second is HF's standard sample-throughput key and this
    # payload now reaches integration callbacks, so publishing tokens/s under it
    # recorded a sequence-length-dependent number as samples/s.
    _patch_value_and_grad_with_aux(monkeypatch)

    trainer, _batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=4, grad_accum=2, epochs=1,
    )
    result = trainer.train()
    metrics = result.metrics if hasattr(result, "metrics") else dict(result)

    assert metrics["train_tokens_per_second"] > 0
    # 4 rows over one epoch, not the token count.
    assert metrics["train_samples_per_second"] == pytest.approx(
        4 / metrics["train_runtime"], rel=1e-6,
    )
    assert metrics["train_samples_per_second"] != metrics["train_tokens_per_second"]


def test_pre_loop_stop_still_reports_a_numeric_epoch(monkeypatch):
    # A callback that stops the run from on_train_begin exits before the loop
    # sets state.epoch. HF's TrainerState defaults it to 0, and stock callbacks
    # rely on that: NotebookProgressCallback.on_train_end does int(state.epoch),
    # so a zero-step cancellation raised TypeError instead of returning.
    _patch_value_and_grad_with_aux(monkeypatch)

    class _StopAtBegin:
        def __init__(self):
            self.epochs = []

        def on_train_begin(self, args, state, control, **kwargs):
            control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.epochs.append(int(state.epoch))
            return control

    spy = _StopAtBegin()
    trainer, _batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=4, grad_accum=2, epochs=1,
        callbacks=[spy],
    )
    trainer.train()

    assert trainer._global_step == 0
    assert spy.epochs == [0]


def test_best_checkpoint_path_is_recorded_for_callbacks(monkeypatch):
    # HF's _save_checkpoint records checkpoint-<best_global_step> on state so
    # on_save and on_train_end callbacks can find the best checkpoint. Ours left
    # the field None for the whole run whenever best tracking is callback-side,
    # so integrations could not locate a directory that was on disk.
    _patch_value_and_grad_with_aux(monkeypatch)

    class _SaveSpy:
        def __init__(self):
            self.paths = []

        def on_step_end(self, args, state, control, **kwargs):
            # Stands in for an improving evaluation: HF's _determine_best_metric
            # advances best_global_step and nothing else.
            state.best_global_step = state.best_global_step or state.global_step
            return control

        def on_save(self, args, state, control, **kwargs):
            self.paths.append(state.best_model_checkpoint)
            return control

    spy = _SaveSpy()
    out_dir = tempfile.mkdtemp()
    trainer, _batches = _epoch_flush_loop_trainer(
        out_dir, microbatches_per_epoch=4, grad_accum=2, epochs=1, save_steps=1,
        callbacks=[spy],
    )
    trainer.train()

    expected = os.path.join(out_dir, "checkpoint-1")
    assert os.path.isdir(expected)
    # Visible to on_save at the time it fires, not only afterwards.
    assert spy.paths[-1] == expected
    assert trainer.state.best_model_checkpoint == expected


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

        def train(self, mode=True):
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
    # epoch: HF only reads trainer_state.json when resume_from_checkpoint is
    # given. It opens at TrainerState's own default instead, not None, so a run
    # cancelled before the loop still hands callbacks a number.
    reused.train()
    assert noop.begin[1] == pytest.approx(0.0), noop.begin


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
def test_vlm_checker_consumes_plan_metadata_only():
    """Call-path contract after the eager internals were deleted: the
    all-masked checker reads construction-time plan supervision counts and
    no longer iterates materialized batch dicts."""
    import inspect

    from unsloth_zoo.mlx.trainer import _check_vlm_all_masked

    source = inspect.getsource(_check_vlm_all_masked)
    assert "supervision_counts" in source
    assert "for batch_dict in batches" not in source
    assert "tolist" not in source


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


def _epoch_flush_loop_trainer(
    out_dir, *, microbatches_per_epoch, grad_accum, epochs, save_steps=0,
    callbacks=(), cycle_length=None,
):
    """Trainer wired to run the real loop over recorded micro-batch visits.

    cycle_length makes the plan a prebuilt multi-epoch schedule (the
    torch_randperm path): microbatches_per_epoch is then the whole schedule and
    cycle_length is one pass through the dataset.
    """
    import mlx.core as mx

    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    class RecordingBatches(list):
        cycle_length = None

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
    if cycle_length is not None:
        RecordingBatches.cycle_length = int(cycle_length)
        # _batches set so train() keeps the prebuilt-epoch flag it clears on the
        # freshly-prepared path.
        trainer._batches = batches
        trainer._prepared_batches_include_epochs = True
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


def test_fractional_prebuilt_schedule_keeps_every_pass_boundary(monkeypatch):
    # The torch_randperm path prebuilds every epoch, so 1.5 epochs of a
    # 5-micro-batch pass is ONE plan spanning two passes and int(num_train_epochs)
    # does not divide it. Reading the pass length as len(plan) // 1 made the whole
    # plan a single epoch: the boundary at micro-batch 5 vanished and the
    # accumulation window never restarted there. The plan's own cycle_length is
    # the pass. Shape measured from a real transformers Trainer: 5 optimizer
    # steps over 9 micro-batches.
    _patch_value_and_grad_with_aux(monkeypatch)

    spy = _EpochScheduleSpy()
    trainer, batches = _epoch_flush_loop_trainer(
        tempfile.mkdtemp(), microbatches_per_epoch=9, grad_accum=2, epochs=1.5,
        cycle_length=5, callbacks=[spy],
    )
    trainer.train()

    assert trainer.state.max_steps == 5 == trainer._global_step
    # The pass boundary at micro-batch 5 forces its own update (step 3).
    assert spy.epoch_end == [(3, 1.0), (5, 1.8)]
    assert batches.visits == [0, 1, 2, 3, 4, 5, 6, 7, 8]


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


def test_restored_callback_states_are_visible_to_callbacks_after_resume():
    """A checkpoint's stateful_callbacks must reach TrainerState on resume.

    The export side was covered but the restore side was not, so dropping the
    assignment entirely went unnoticed.
    """
    from unsloth_zoo.mlx.trainer import MLXTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    trainer.state = types.SimpleNamespace(stateful_callbacks={})

    saved = {"EarlyStoppingCallback": {"early_stopping_patience_counter": 2}}
    trainer._restore_callback_states(saved)
    assert trainer.state.stateful_callbacks == saved
    # Mirrored, not aliased: mutating the checkpoint payload afterwards must
    # not reach into live state.
    saved["EarlyStoppingCallback"] = {"early_stopping_patience_counter": 99}
    assert trainer.state.stateful_callbacks["EarlyStoppingCallback"] == {
        "early_stopping_patience_counter": 2
    }

    # An absent or empty section leaves whatever the run already had.
    for empty in (None, {}, "not-a-dict"):
        trainer._restore_callback_states(empty)
        assert trainer.state.stateful_callbacks == {
            "EarlyStoppingCallback": {"early_stopping_patience_counter": 2}
        }
def test_qwen3_visual_window_keeps_features_when_mask_spans_whole_window():
    """A reused prompt cache must not shift the feature window off the mask."""
    import mlx.core as mx
    import unsloth_zoo.mlx.compile as mc

    masks = mx.array([[0, 0, 1, 1, 0, 0]], dtype=mx.bool_)
    embeds = [mx.array([[1], [2]])]
    packed, positions = mc._pack_qwen3_visual_state(masks, embeds)

    # The mask already covers only the new turn, so an offset carried over from
    # an earlier turn must not be applied to it a second time.
    for carried_offset in (0, 6, 20):
        window_masks, window_embeds = mc._qwen3_visual_window(
            masks,
            packed,
            positions,
            mask_offsets=(carried_offset,),
            window=6,
        )
        assert window_masks.tolist() == [[0, 0, 1, 1, 0, 0]]
        assert window_embeds[0].tolist() == [[1], [2]]


def test_qwen3_prompt_rows_defer_to_the_native_deepstack_path():
    """A row carrying mlx-vlm's own deepstack features keeps them.

    Synthesizing empty compact state for such a row hands the language call
    zeros in place of its real visual features.
    """
    import mlx.core as mx
    import unsloth_zoo.mlx.compile as mc

    masks = mx.array([[0, 1, 1, 0]], dtype=mx.bool_)
    state, positions = mc._pack_qwen3_visual_state(masks, [mx.array([[1.0], [2.0]])])
    compact_row = {
        "inputs_embeds": mx.zeros((1, 4, 1)),
        "visual_pos_masks": masks,
        mc._QWEN3_VISUAL_STATE_KEY: state,
        mc._QWEN3_VISUAL_POSITIONS_KEY: positions,
    }
    native_row = {
        "inputs_embeds": mx.zeros((1, 4, 1)),
        "visual_pos_masks": masks,
        # A bare array is what _pack_qwen3_visual_state declines to compact.
        "deepstack_visual_embeds": mx.array([[7.0], [8.0]]),
    }

    rows = [native_row, compact_row]
    assert mc._pad_qwen3_prompt_rows(rows) is rows

    seen = []

    def language(self, inputs, inputs_embeds=None, mask=None, cache=None,
                 visual_pos_masks=None, deepstack_visual_embeds=None, **kwargs):
        seen.append(deepstack_visual_embeds)

    padded = mc._pad_qwen3_prompt_rows(rows)[0]
    call_kwargs = dict(padded)
    call_kwargs.pop("inputs_embeds")
    mc._qwen3_visual_state_adapter(language)(
        object(), mx.zeros((1, 4)), cache=None, **call_kwargs
    )
    assert seen[0].tolist() == [[7.0], [8.0]]

    # A mask-only row owns no features, so it still gets an empty state to keep
    # its slot in the batch.
    mask_only = {"inputs_embeds": mx.zeros((1, 4, 1)), "visual_pos_masks": masks}
    filled = mc._pad_qwen3_prompt_rows([mask_only, compact_row])[0]
    assert not filled[mc._QWEN3_VISUAL_STATE_KEY].any().item()
