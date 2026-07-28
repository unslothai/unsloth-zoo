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


def test_preference_configs_preserve_explicit_default_warmup_steps():
    # The preference subclasses (ORPO/DPO/GRPO) must keep MLXTrainingConfig's
    # custom __init__ so an explicitly-passed warmup_steps that equals the
    # default (5) is still recorded as explicit and wins over a positive
    # warmup_ratio (HF get_warmup_steps parity: warmup_steps > 0 overrides the
    # ratio). A dataclass-generated __init__ would skip
    # _unsloth_mlx_warmup_steps_explicit and silently switch to the ratio,
    # changing the LR schedule.
    from unsloth_zoo.mlx.trainer import (
        MLXTrainer,
        MLXORPOConfig,
        MLXDPOConfig,
        MLXGRPOConfig,
    )

    for config_cls in (MLXORPOConfig, MLXDPOConfig, MLXGRPOConfig):
        cfg = config_cls(
            learning_rate=5e-5,
            lr_scheduler_type="linear",
            warmup_steps=5,
            warmup_ratio=0.1,
        )
        assert cfg._unsloth_mlx_warmup_steps_explicit is True, config_cls.__name__
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.args = cfg
        # Explicit warmup_steps=5 must win over the ratio (0.1 * 8 -> 1).
        assert trainer._resolve_warmup_steps(total_steps=8) == 5, config_cls.__name__

    # The GRPO subclass must still register and honour its extra fields through
    # the inherited __init__ (init=False keeps them in fields()).
    grpo = MLXGRPOConfig(num_generations=8, warmup_ratio=0.2)
    assert grpo.num_generations == 8
    assert grpo.loss_type == "grpo"
    assert grpo._unsloth_mlx_warmup_steps_explicit is False
    # An unset warmup_steps still lets the ratio drive the schedule.
    grpo_trainer = MLXTrainer.__new__(MLXTrainer)
    grpo_trainer.args = grpo
    assert grpo_trainer._resolve_warmup_steps(total_steps=100) == 20


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


def test_response_mask_tokenizer_rejects_encode_only_tokenizer():
    from unsloth_zoo.mlx.trainer import _resolve_response_mask_tokenizer

    class EncodeOnlyTokenizer:
        def encode(self, text):
            return [1, 2, 3]

        def convert_tokens_to_ids(self, token):
            return 1

    with pytest.raises(TypeError, match="requires a callable"):
        _resolve_response_mask_tokenizer(EncodeOnlyTokenizer())


# ---------------------------------------------------------------------------
# GRPO / DPO preference correctness (PR #832 review fixes).
# ---------------------------------------------------------------------------

def test_hf_encoding_tokenizer_keeps_plain_hf_and_unwraps_mlx_wrapper():
    from unsloth_zoo.mlx.utils import _hf_encoding_tokenizer
    from transformers import PreTrainedTokenizerBase

    class PlainHF(PreTrainedTokenizerBase):
        pass

    class RustLike:
        # The low-level Rust tokenizer: encode returns Encoding, no eos_token_id.
        def encode(self, text):
            return object()

    class Wrapper:
        # mlx-lm TokenizerWrapper: real HF tokenizer stored under _tokenizer.
        def __init__(self, inner):
            self._tokenizer = inner

    hf = PlainHF()
    assert _hf_encoding_tokenizer(Wrapper(hf)) is hf

    # A plain HF fast tokenizer ALSO exposes _tokenizer (its Rust backend); it
    # must be returned as-is rather than unwrapped to the Rust object.
    plain = PlainHF()
    plain._tokenizer = RustLike()
    assert _hf_encoding_tokenizer(plain) is plain


def test_grpo_and_preference_use_hf_aware_tokenizer_unwrap():
    import inspect
    from unsloth_zoo.mlx import trainer as trainer_mod
    from unsloth_zoo.mlx import utils as utils_mod

    grpo_src = inspect.getsource(trainer_mod.MLXGRPOTrainer._grpo_rollout_generator)
    assert "_hf_encoding_tokenizer(self.tokenizer)" in grpo_src
    assert 'getattr(self.tokenizer, "_tokenizer"' not in grpo_src

    pref_src = inspect.getsource(utils_mod.create_preference_batches)
    assert "_hf_encoding_tokenizer(tokenizer)" in pref_src


def test_preference_batches_preserve_completion_when_prompt_exceeds_budget():
    # A preference row whose prompt+answer exceeds max_seq_length must keep the
    # chosen/rejected answer tokens (the [pe, seq_end) span the DPO/ORPO loss
    # scores), not right-truncate them away. Prompt=6 tokens (value 1), answers=4
    # tokens each (values 2/3); prompt+answer=10 > max_seq_length=8, and the
    # prompt alone (6) leaves only 2 slots. A plain [:8] would keep 6 prompt + 2
    # answer tokens (dropping 2 of the 4 scored answer tokens); the completion-
    # preserving truncation must drop 2 PROMPT tokens so all 4 answer tokens
    # survive in the scored span.
    from unsloth_zoo.mlx.utils import create_preference_batches

    class _Tok:
        eos_token_id = 7  # distinct from prompt (1) / answer (2, 3) tokens
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            return [int(part) for part in text.split()]

    dataset = [{
        "prompt": "1 1 1 1 1 1",
        "chosen": " 2 2 2 2",
        "rejected": " 3 3 3 3",
    }]
    batch, lengths, _ = create_preference_batches(
        dataset, _Tok(), batch_size=1, max_seq_length=8,
    )[0]
    lengths = lengths.tolist()
    rows = batch.tolist()
    chosen_start, chosen_end = lengths[0]
    rejected_start, rejected_end = lengths[1]

    # The 4 answer tokens plus the appended EOS (5 total) all remain in the scored
    # [start, end) completion span; the completion-preserving truncation dropped
    # PROMPT tokens (not answer tokens) to fit max_seq_length=8.
    assert chosen_end - chosen_start == 5
    assert rejected_end - rejected_start == 5
    # Every scored token is an answer token (2 / 3) followed by the EOS (7),
    # never a prompt token (1).
    assert rows[0][chosen_start:chosen_end] == [2, 2, 2, 2, 7]
    assert rows[1][rejected_start:rejected_end] == [3, 3, 3, 3, 7]


def test_preference_batches_append_eos_to_completions():
    # TRL appends the EOS id to each DPO/ORPO completion so the model learns to
    # stop. The scored [response_start, seq_end) span must therefore end with the
    # tokenizer's eos_token_id (guarded against a double EOS). No truncation here
    # (well under max_seq_length), isolating the EOS-append behavior.
    from unsloth_zoo.mlx.utils import create_preference_batches

    class _Tok:
        eos_token_id = 7  # distinct from prompt (1) / answer (2, 3) tokens
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            return [int(part) for part in text.split()]

    dataset = [{"prompt": "1 1", "chosen": " 2 2", "rejected": " 3 3 3"}]
    batch, lengths, _ = create_preference_batches(
        dataset, _Tok(), batch_size=1, max_seq_length=64,
    )[0]
    lengths = lengths.tolist()
    rows = batch.tolist()
    cs, ce = lengths[0]
    rs, re_ = lengths[1]
    # Chosen completion "2 2" -> [2, 2, 7]; rejected "3 3 3" -> [3, 3, 3, 7].
    assert rows[0][cs:ce] == [2, 2, 7]
    assert rows[1][rs:re_] == [3, 3, 3, 7]
    # The EOS is the last scored token of each completion.
    assert rows[0][ce - 1] == 7 and rows[1][re_ - 1] == 7


def test_preference_batches_clamp_padding_to_max_seq_length():
    # When max_seq_length is not a multiple of pad_to_multiple, the per-batch
    # pad_to_multiple round-up must be clamped back to max_seq_length, mirroring
    # the SFT/text path (_create_labeled_batches: padded_len = min(padded_len,
    # max_seq_length)). Here max_seq_length=40 with the default pad_to_multiple=32
    # would round a full row up to 64 without the clamp, running the preference
    # forward over 24 extra padded positions and potentially overrunning a tight
    # context cap. Each row is already truncated to max_seq_length, so the batch
    # width must equal max_seq_length exactly, never the rounded-up 64.
    from unsloth_zoo.mlx.utils import create_preference_batches

    class _Tok:
        eos_token_id = 7
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            return [int(part) for part in text.split()]

    # prompt=1 token, chosen/rejected each 40 answer tokens: prompt+answer(+EOS)
    # exceeds max_seq_length=40, so each row is truncated to exactly 40 tokens.
    dataset = [{
        "prompt": "1",
        "chosen": " " + " ".join(["2"] * 40),
        "rejected": " " + " ".join(["3"] * 40),
    }]
    batch, lengths, _ = create_preference_batches(
        dataset, _Tok(), batch_size=1, max_seq_length=40,
    )[0]
    rows = batch.tolist()
    # Batch width is clamped to max_seq_length (40), not rounded up to 64.
    assert len(rows[0]) == 40
    assert len(rows[1]) == 40
    # The scored completion span still fits within the clamped width.
    for start, end in lengths.tolist():
        assert 0 <= start <= end <= 40


def test_preference_batches_common_prefix_boundary_under_token_merge():
    # When the tokenizer MERGES the boundary token -- the last prompt token fuses
    # with the first chosen/rejected token inside encode(prompt+answer) -- the
    # standalone prompt is no longer a true prefix of the concatenation. A
    # length-only min(len(p_ids), len(c_full), len(r_full)) boundary would then
    # (a) treat the merged token (which carries part of the response) as prompt and
    # mask it out of the scored [response_start, seq_end) span, and (b) land on a
    # position where the chosen and rejected "prompt" prefixes DIFFER, so the pair
    # no longer shares a single response_start. _common_prefix_len stops at the
    # first divergent id, keeping the merged token in the response and a single
    # response_start both rows agree on.
    from unsloth_zoo.mlx.utils import (
        create_preference_batches,
        _common_prefix_len,
    )

    class _MergeTok:
        eos_token_id = 7  # distinct from every content token below
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            # Standalone prompt "P" ends with a boundary token (9). Concatenating
            # the answer FUSES that 9 with the first answer token into a NEW id
            # (5 for chosen, 6 for rejected), so p_ids is not a prefix of either
            # full encoding and the two prompt prefixes diverge at index 2.
            table = {
                "P": [1, 1, 9],
                "PC": [1, 1, 5, 2, 2],
                "PR": [1, 1, 6, 3, 3],
            }
            return list(table[text])

    # The shared prefix across the standalone prompt and both concatenations is 2
    # (the divergence index), NOT min(len(...)) == 3.
    assert _common_prefix_len(
        [1, 1, 9], [1, 1, 5, 2, 2, 7], [1, 1, 6, 3, 3, 7]
    ) == 2
    # Sanity: an ordinary true-prefix pair is a no-op (returns len(p_ids)).
    assert _common_prefix_len([1, 1], [1, 1, 2, 2], [1, 1, 3, 3, 3]) == 2

    dataset = [{"prompt": "P", "chosen": "C", "rejected": "R"}]
    batch, lengths, _ = create_preference_batches(
        dataset, _MergeTok(), batch_size=1, max_seq_length=64,
    )[0]
    rows = batch.tolist()
    lengths = lengths.tolist()
    chosen_start, chosen_end = lengths[0]
    rejected_start, rejected_end = lengths[1]

    # Response boundary is the shared-prefix length (2), NOT min(len(p_ids)) == 3.
    assert chosen_start == 2
    assert rejected_start == 2
    # Both rows keep the SAME prompt prefix up to the boundary; a min()-based
    # boundary at 3 would put the diverging merged tokens (5 vs 6) inside the
    # "prompt" region, so the pair would no longer share a response_start.
    assert rows[0][:chosen_start] == [1, 1]
    assert rows[1][:rejected_start] == [1, 1]
    # The merged boundary token stays in the scored response span (5 for chosen,
    # 6 for rejected), followed by the answer tokens and the appended EOS -- a
    # min() boundary would have masked it out of the loss.
    assert rows[0][chosen_start:chosen_end] == [5, 2, 2, 7]
    assert rows[1][rejected_start:rejected_end] == [6, 3, 3, 7]


def test_grpo_prepare_data_normalizes_tokenizer_chat_template():
    # GRPO must apply the configured chat_template override to the rollout
    # tokenizer, mirroring the SFT/DPO data path (MLXTrainer._prepare_data).
    # Otherwise _grpo_render_prompt renders chat prompts on the raw tokenizer:
    # a MLXTrainingConfig(chat_template=...) override is silently ignored and a
    # base tokenizer without a built-in template raises, even though the same
    # config trains fine under SFT.
    import inspect
    import types
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer
    from unsloth_zoo.mlx.utils import _hf_encoding_tokenizer

    # Structural parity: the GRPO data path normalizes like the SFT/DPO one.
    src = inspect.getsource(MLXGRPOTrainer._prepare_data)
    assert "normalize_mlx_chat_template" in src

    class _Tok:
        eos_token_id = 0
        chat_template = None  # base checkpoint: no built-in template

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = types.SimpleNamespace(_config={}, _hf_repo=None)
    trainer.tokenizer = _Tok()
    trainer.train_dataset = [{"prompt": [{"role": "user", "content": "hi"}]}]
    raw_template = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
    trainer.args = types.SimpleNamespace(chat_template=raw_template)

    batches, batch_iter = trainer._prepare_data(is_vlm=False)
    # GRPO drives the loop with a rollout generator, not static batches.
    assert batches is None
    assert batch_iter is not None
    # The tokenizer used to render rollout prompts now carries the override, so
    # apply_chat_template renders (and no longer raises) on this base tokenizer.
    hf = _hf_encoding_tokenizer(trainer.tokenizer)
    assert getattr(hf, "chat_template", None) == raw_template


def test_grpo_trainer_fills_defaults_from_base_config(monkeypatch):
    from unsloth_zoo.mlx.trainer import (
        MLXGRPOTrainer,
        MLXGRPOConfig,
        MLXTrainer,
        MLXTrainingConfig,
    )

    # Exercise only the config coercion; skip the heavy base __init__.
    monkeypatch.setattr(MLXTrainer, "__init__", lambda self, *a, **k: None)

    base = MLXTrainingConfig(max_steps=3)
    assert not hasattr(base, "num_generations")
    assert not hasattr(base, "max_completion_length")

    reward = lambda completions, **kw: [0.0] * len(completions)
    MLXGRPOTrainer(object(), object(), [], reward, args=base)

    defaults = MLXGRPOConfig()
    assert base.loss_type == "grpo"
    assert base.num_generations == defaults.num_generations
    assert base.max_completion_length == defaults.max_completion_length
    assert base.grpo_beta == defaults.grpo_beta


def test_base_trainer_rejects_grpo_loss_type():
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXGRPOConfig

    inst = MLXTrainer.__new__(MLXTrainer)
    inst.args = MLXGRPOConfig()
    with pytest.raises(ValueError, match="MLXGRPOTrainer"):
        MLXTrainer._prepare_data(inst, is_vlm=False)


def test_grpo_eval_is_skipped_like_dpo():
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    # GRPO must be inside the unsupported-eval skip guard; otherwise SFT eval
    # batches feed labels as advantages into make_grpo_loss_fn and crash on
    # advantages.reshape.
    src = inspect.getsource(MLXTrainer._train_inner)
    assert 'in ("orpo", "dpo", "grpo")' in src


def test_lora_reference_discovery_covers_non_loralinear_adapters():
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.utils import iter_mlx_lora_modules

    # A LoRA adapter that is NOT an mlx-lm LoRALinear (e.g. LoRAEmbedding /
    # LoRASwitchLinear) still exposes lora_a/lora_b and must be discovered so
    # the DPO/GRPO reference forward disables every adapter, not just linears.
    class FakeEmbedAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.zeros((4, 2))
            self.lora_b = mx.zeros((2, 4))
            self.scale = 1.0

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = FakeEmbedAdapter()

    found = [m for _, m in iter_mlx_lora_modules(Root())]
    assert any(isinstance(m, FakeEmbedAdapter) for m in found)

    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer
    src = inspect.getsource(MLXTrainer._train_inner)
    assert src.count("iter_mlx_lora_modules(model)") >= 2
    assert "is_leaf=lambda x: isinstance(x, LoRALinear)" not in src


def test_disable_lora_dropout_neutralizes_every_adapter_dropout():
    # TRL DPO/ORPO default disable_dropout=True; the MLX preference loss runs in
    # train() mode inside the compiled step (per-step eval() unreliable), so the
    # setup must neutralize each LoRA/DoRA module's dropout by replacing it with
    # identity, covering every adapter iter_mlx_lora_modules finds.
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import _mlx_disable_lora_dropout, _mlx_identity

    class FakeLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.zeros((4, 2))
            self.lora_b = mx.zeros((2, 4))
            self.scale = 1.0
            # A stand-in "dropout" that is NOT identity (zeros its input).
            self.dropout = lambda x: x * 0.0

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = FakeLoRA()
            self.k = FakeLoRA()

    root = Root()
    n = _mlx_disable_lora_dropout(root)
    assert n == 2
    assert root.q.dropout is _mlx_identity
    assert root.k.dropout is _mlx_identity
    x = mx.ones((2, 4))
    assert _mlx_identity(x).tolist() == x.tolist()


def test_dpo_orpo_setup_disables_lora_dropout_but_grpo_does_not():
    # DPO and ORPO must disable adapter dropout before building their loss (TRL
    # default disable_dropout=True); GRPO must NOT (TRL default False), so keeping
    # the prior-round GRPO-dropout rejection consistent.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    call = "_mlx_disable_lora_dropout(model)"
    assert src.count(call) == 2
    orpo_b = src.index("make_orpo_loss_fn")
    dpo_b = src.index("make_dpo_loss_fn")
    grpo_b = src.index("make_grpo_loss_fn")
    # ORPO disables before its loss builder ...
    assert src.index(call) < orpo_b
    # ... DPO disables before its loss builder (after the ORPO builder) ...
    assert orpo_b < src.index(call, orpo_b) < dpo_b
    # ... and the GRPO branch has no disable call before its builder.
    assert call not in src[dpo_b:grpo_b]


def test_grpo_reward_aggregation_skips_none_values():
    import inspect
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    # The rollout reward loop must skip None (TRL multi-task reward pattern)
    # rather than crash on float(None).
    src = inspect.getsource(MLXGRPOTrainer._grpo_rollout_generator)
    assert "if v is not None" in src

    # Behavioral check on the aggregation logic the generator runs: None from
    # one reward func is ignored, the applicable func still contributes.
    N = 3
    reward_funcs = [
        lambda **kw: [1.0, None, 2.0],
        lambda **kw: [None, 0.5, None],
    ]
    total = [0.0] * N
    for rf in reward_funcs:
        for i, v in enumerate(rf()):
            if v is not None:
                total[i] += float(v)
    assert total == [1.0, 0.5, 2.0]


# ---------------------------------------------------------------------------
# PR #832 round-2 review fixes: GRPO group-size / reward-length validation and
# preference-reference non-LoRA-trainable guard.
# ---------------------------------------------------------------------------

def _make_grpo_trainer_for_rollout(monkeypatch, num_generations, reward_funcs,
                                   batch_texts=None):
    """Minimal MLXGRPOTrainer wired just enough to drive one rollout."""
    import types
    import mlx_lm
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    class _Tok:
        eos_token_id = 0
        def encode(self, s):
            return [1, 2, 3]

    if batch_texts is None:
        batch_texts = ["gen"] * num_generations
    resp = types.SimpleNamespace(texts=batch_texts)
    monkeypatch.setattr(mlx_lm, "batch_generate",
                        lambda *a, **k: resp, raising=False)

    class _Model:
        # Minimal stand-in for an mlx nn.Module: tracks training mode so the
        # rollout's eval()/train() dropout guard has something to toggle.
        def __init__(self):
            self.training = True
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = _Model()
    trainer.tokenizer = _Tok()
    trainer.train_dataset = [{"prompt": "hello", "answer": "x"}]
    trainer.reward_funcs = list(reward_funcs)
    args = types.SimpleNamespace(
        num_generations=num_generations,
        temperature=1.0,
        max_completion_length=4,
        max_seq_length=32,
    )
    trainer.args = args
    return trainer


def test_grpo_rollout_rejects_single_generation(monkeypatch):
    # num_generations=1 gives a one-element group: mean == reward and std == 0,
    # so every group-relative advantage is exactly 0 and the policy receives no
    # reward gradient (a silent no-op GRPO objective). Fail fast instead.
    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=1,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.0]],
    )
    gen = trainer._grpo_rollout_generator()
    with pytest.raises(ValueError, match="num_generations >= 2"):
        next(gen)


def test_grpo_rollout_rejects_empty_dataset(monkeypatch):
    # An empty (or fully filtered) prompt dataset must fail fast with the same
    # clear "No GRPO prompts found" message on the default max_steps>0 path, not
    # crash with an opaque ZeroDivisionError from "_order[idx % len(prompts)]"
    # once the rollout loop starts. The epoch path already validates this in
    # _prepare_data; the generator guard extends it to every entry path.
    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.0, 0.0]],
    )
    trainer.train_dataset = []
    gen = trainer._grpo_rollout_generator()
    with pytest.raises(ValueError, match="No GRPO prompts found"):
        next(gen)


def test_grpo_rollout_rejects_wrong_length_reward_vector(monkeypatch):
    # A reward func returning fewer scores than completions would leave the
    # unfilled slots at 0.0 (silently fabricated rewards -> corrupted
    # advantages); a longer one would index out of range. Fail fast.
    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=3,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [1.0]],
        batch_texts=["a", "b", "c"],
    )
    gen = trainer._grpo_rollout_generator()
    with pytest.raises(ValueError, match="returned 1 scores for 3 completions"):
        next(gen)


def test_grpo_rollout_accepts_matching_length_reward_vector(monkeypatch):
    # The length guard must not false-positive on a correct reward func: one
    # score per completion advances past both guards and yields a rollout tuple.
    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=3,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2, 0.3]],
        batch_texts=["a", "b", "c"],
    )
    gen = trainer._grpo_rollout_generator()
    batch, lengths, advantages = next(gen)
    assert batch.shape[0] == 3
    assert advantages.shape[0] == 3


def test_grpo_rollout_generates_with_dropout_disabled(monkeypatch):
    # GRPO rollouts are autoregressive sampling used to score completions. The
    # training loop puts the model in train() mode before the rollout runs, and
    # mlx-lm's batch_generate never toggles eval, so LoRA/DoRA dropout would be
    # active during generation: the sampled tokens (and the log-probs the loss
    # later scores them with) would come from a randomly masked sub-network,
    # corrupting the advantages. The rollout must generate in eval() and restore
    # the prior training mode afterwards.
    import types
    import mlx_lm

    class _Model:
        def __init__(self):
            self.training = True
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    model = _Model()
    trainer.model = model

    seen = {}

    def _capture_generate(m, tok, **kwargs):
        seen["training_during_generate"] = m.training
        return types.SimpleNamespace(texts=["a", "b"])

    monkeypatch.setattr(mlx_lm, "batch_generate", _capture_generate, raising=False)

    gen = trainer._grpo_rollout_generator()
    next(gen)

    # Dropout must be inert while sampling the rollout ...
    assert seen["training_during_generate"] is False
    # ... and the prior training mode must be restored for the grad step.
    assert model.training is True


# ---------------------------------------------------------------------------
# PR #832 round-4 review fixes: GRPO gradient-accumulation weight (rollout rows,
# not tokens) and the GRPO rollout double-BOS guard.
# ---------------------------------------------------------------------------

def test_grpo_loss_returns_rollout_row_count_not_token_count(monkeypatch):
    # The GRPO loss is a masked per-row mean over completion tokens, then a mean
    # over the rollout group (TRL loss_type="grpo"). The trainer accumulates
    # micro-batch grads weighted by this returned count, so it must be the
    # number of rollout ROWS; returning the completion-token total would let
    # micro-batches with longer completions dominate under
    # gradient_accumulation_steps > 1.
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.utils import make_grpo_loss_fn

    # Real MLX cross_entropy defaults to reduction="none" (per token), which the
    # loss relies on; the torch shim defaults to "mean". Force the faithful
    # per-token reduction so the code path runs exactly as on MLX.
    _orig_ce = nn.losses.cross_entropy

    def _ce_none(logits, targets, **kw):
        # Real MLX returns a per-element tensor shaped like ``targets``; the
        # shim flattens, so reshape back to keep the (rows, tokens) layout.
        out = _orig_ce(logits, targets, **{**kw, "reduction": "none"})
        return out.reshape(*targets.shape)

    monkeypatch.setattr(nn.losses, "cross_entropy", _ce_none)
    # Real MLX mx.maximum broadcasts a Python scalar; the torch shim requires a
    # tensor. Coerce so the loss's tok_per_row = mx.maximum(mask.sum(-1), 1.0)
    # runs as it does on MLX.
    _orig_max = mx.maximum
    monkeypatch.setattr(
        mx, "maximum",
        lambda a, b, **kw: _orig_max(
            a, mx.array(b) if isinstance(b, (int, float)) else b, **kw
        ),
    )

    V = 5
    # Two rollout rows with DIFFERENT completion lengths so the row count (2)
    # and the completion-token total (2 + 5 = 7) are distinct.
    batch = mx.array(
        [[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 4, 4]], dtype=mx.int32
    )
    lengths = mx.array([[1, 3], [1, 6]], dtype=mx.int32)
    advantages = mx.array([0.5, -0.5])

    def model(inp):
        return mx.zeros((inp.shape[0], inp.shape[1], V))

    loss_fn = make_grpo_loss_fn(beta=0.0, reference_free=True)
    _loss, weight = loss_fn(model, batch, lengths, advantages)
    assert int(weight) == batch.shape[0] == 2


# ---------------------------------------------------------------------------
# PR #832 round-7 review fixes: ORPO/DPO gradient-accumulation weight must be
# the PAIR count (not the response-token count), the GRPO rollout must score the
# generated token IDs (not a decode->re-encode roundtrip), and preference
# truncation under max_steps must not keep only the shortest pairs.
# ---------------------------------------------------------------------------

def _ce_none_monkeypatch(monkeypatch):
    """Force MLX's per-token cross_entropy reduction under the torch shim."""
    import mlx.core as mx
    import mlx.nn as nn
    _orig_ce = nn.losses.cross_entropy

    def _ce_none(logits, targets, **kw):
        out = _orig_ce(logits, targets, **{**kw, "reduction": "none"})
        return out.reshape(*targets.shape)

    monkeypatch.setattr(nn.losses, "cross_entropy", _ce_none)
    _orig_max = mx.maximum
    monkeypatch.setattr(
        mx, "maximum",
        lambda a, b, **kw: _orig_max(
            a, mx.array(b) if isinstance(b, (int, float)) else b, **kw
        ),
    )
    # The torch shim does not implement nn.log_sigmoid (no existing MLX test
    # exercises it); route it to torch for the preference losses under test.
    monkeypatch.setattr(
        nn, "log_sigmoid",
        lambda x: torch.nn.functional.logsigmoid(x),
        raising=False,
    )


def test_orpo_loss_returns_pair_count_not_token_count(monkeypatch):
    # The ORPO loss reduces as a token-mean SFT term plus a pair-mean odds-ratio
    # term. The trainer weights each accumulated micro-batch gradient by this
    # returned count, so it must be the number of preference PAIRS; returning the
    # response-token total would let micro-batches with longer chosen/rejected
    # responses dominate under gradient_accumulation_steps > 1.
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import make_orpo_loss_fn

    _ce_none_monkeypatch(monkeypatch)
    V = 5
    # 2 pairs -> 4 rows [chosen0, chosen1, rejected0, rejected1] with DIFFERENT
    # response lengths so the pair count (2) and the token total are distinct.
    batch = mx.array(
        [
            [1, 2, 3, 4, 0, 0],
            [1, 2, 3, 4, 4, 4],
            [1, 2, 3, 0, 0, 0],
            [1, 2, 3, 4, 4, 0],
        ],
        dtype=mx.int32,
    )
    lengths = mx.array([[1, 4], [1, 6], [1, 3], [1, 5]], dtype=mx.int32)

    def model(inp):
        return mx.zeros((inp.shape[0], inp.shape[1], V))

    loss_fn = make_orpo_loss_fn(beta=0.1)
    _loss, weight = loss_fn(model, batch, lengths, None)
    assert int(weight) == batch.shape[0] // 2 == 2


def test_dpo_loss_returns_pair_count_not_token_count(monkeypatch):
    # DPO loss is a mean over preference pairs, so the accumulate-then-normalize
    # weight must be the pair count, not the response-token total.
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import make_dpo_loss_fn

    _ce_none_monkeypatch(monkeypatch)
    V = 5
    batch = mx.array(
        [
            [1, 2, 3, 4, 0, 0],
            [1, 2, 3, 4, 4, 4],
            [1, 2, 3, 0, 0, 0],
            [1, 2, 3, 4, 4, 0],
        ],
        dtype=mx.int32,
    )
    lengths = mx.array([[1, 4], [1, 6], [1, 3], [1, 5]], dtype=mx.int32)

    def model(inp):
        return mx.zeros((inp.shape[0], inp.shape[1], V))

    loss_fn = make_dpo_loss_fn(beta=0.1, reference_free=True)
    _loss, weight = loss_fn(model, batch, lengths, None)
    assert int(weight) == batch.shape[0] // 2 == 2


def test_grpo_rollout_scores_generated_token_ids_not_reencoded_text(monkeypatch):
    # When mlx-lm surfaces the generated token IDs, the rollout must score those
    # exact ids (prompt ids + sampled completion ids), not a decode->re-encode
    # roundtrip of the completion text. Here the decoded text 'gen' re-encodes to
    # [1, 2, 3], distinct from the sampled ids, so building rows from the ids
    # proves the roundtrip is bypassed and the loss runs on the sampled tokens.
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["gen", "gen"],
    )

    def _fake_generate(model, tokenizer, prompts=None, max_tokens=None,
                       sampler=None, verbose=False, return_token_ids=False):
        # The trainer detects support via this explicit parameter and must
        # request the ids.
        assert return_token_ids is True
        return types.SimpleNamespace(
            texts=["gen", "gen"], token_ids=[[7, 8, 9, 9], [7, 8]]
        )

    monkeypatch.setattr(mlx_lm, "batch_generate", _fake_generate, raising=False)

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _advantages = next(gen)
    rows = batch.tolist()
    lens = lengths.tolist()
    # pids = encode('hello') = [1, 2, 3]; response boundary pe = 3.
    pe = lens[0][0]
    assert pe == 3
    assert rows[0][:pe] == [1, 2, 3]
    # Row 0 filled max_completion_length (4) ids -> truncated on length, no EOS
    # was emitted, so the ids are scored as-is.
    assert rows[0][pe:lens[0][1]] == [7, 8, 9, 9]
    # Row 1 stopped early (2 < 4 ids): mlx-lm stripped the terminal EOS, which
    # the rollout re-appends (eos_token_id == 0) so the stop action is scored.
    assert rows[1][pe:lens[1][1]] == [7, 8, 0]


def test_grpo_rollout_appends_sampled_eos_on_normal_stop(monkeypatch):
    # mlx-lm's batch_generate strips the terminal EOS from the returned ids for a
    # normally-terminating row (finish_reason == "stop") and keeps all
    # max_completion_length ids only when it truncated on length. TRL's GRPO
    # completion mask is inclusive of that EOS (sequence_indices <= eos_idx), so
    # the model's probability of stopping gets advantage-weighted gradient/KL.
    # The rollout must therefore re-append the EOS for a stopped row (ids shorter
    # than the cap) and leave a truncated row (ids == cap) untouched.
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=3,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2, 0.3]],
        batch_texts=["a", "b", "c"],
    )
    # eos_token_id == 0 (see _Tok); max_completion_length == 4.
    def _gen(model, tokenizer, prompts=None, max_tokens=None,
             sampler=None, verbose=False, return_token_ids=False):
        assert return_token_ids is True
        return types.SimpleNamespace(
            texts=["a", "b", "c"],
            # row0: stopped early (1 < 4) -> EOS re-appended
            # row1: stopped early (3 < 4) -> EOS re-appended
            # row2: filled the cap (4 == 4) -> truncated, no EOS
            token_ids=[[7], [7, 8, 9], [7, 8, 9, 9]],
        )

    monkeypatch.setattr(mlx_lm, "batch_generate", _gen, raising=False)

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _adv = next(gen)
    rows = batch.tolist()
    lens = lengths.tolist()
    pe = lens[0][0]
    assert rows[0][pe:lens[0][1]] == [7, 0]
    assert rows[1][pe:lens[1][1]] == [7, 8, 9, 0]
    assert rows[2][pe:lens[2][1]] == [7, 8, 9, 9]


def test_grpo_rollout_appends_no_eos_when_tokenizer_lacks_eos_id(monkeypatch):
    # Guard the EOS re-append against tokenizers with no eos_token_id: a None id
    # must be skipped (no crash, no bogus token) rather than appended.
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    trainer.tokenizer.eos_token_id = None

    def _gen(model, tokenizer, prompts=None, max_tokens=None,
             sampler=None, verbose=False, return_token_ids=False):
        return types.SimpleNamespace(texts=["a", "b"], token_ids=[[7], [7, 8]])

    monkeypatch.setattr(mlx_lm, "batch_generate", _gen, raising=False)

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _adv = next(gen)
    rows = batch.tolist()
    lens = lengths.tolist()
    pe = lens[0][0]
    assert rows[0][pe:lens[0][1]] == [7]
    assert rows[1][pe:lens[1][1]] == [7, 8]


def test_grpo_rollout_caps_generation_to_trainable_remainder(monkeypatch):
    # The reward funcs score the FULL generated text, so generation must be
    # bounded so prompt + completion fits max_seq_length; otherwise a post-hoc
    # (pids+comp)[:max_seq_length] cut would drop reward-scored completion tokens
    # from the loss/KL span (silent reward/score misattribution). When the prompt
    # is long relative to max_seq_length, max_tokens must be capped to
    # max_seq_length - len(pids), below max_completion_length.
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    # Prompt encodes to 6 ids; tiny max_seq_length leaves only 2 for completion.
    trainer.tokenizer.encode = lambda s: [1, 2, 3, 4, 5, 6]
    trainer.args.max_completion_length = 20
    trainer.args.max_seq_length = 8

    seen = {}
    def _gen(model, tokenizer, prompts=None, max_tokens=None,
             sampler=None, verbose=False, return_token_ids=False):
        seen["max_tokens"] = max_tokens
        return types.SimpleNamespace(texts=["a", "b"], token_ids=[[7, 8], [7, 8]])

    monkeypatch.setattr(mlx_lm, "batch_generate", _gen, raising=False)

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _adv = next(gen)
    # max_seq_length(8) - len(pids)(6) = 2, below max_completion_length(20).
    assert seen["max_tokens"] == 2
    # Every built row fits within max_seq_length (no reward-scored token dropped).
    for row_len in [l[1] for l in lengths.tolist()]:
        assert row_len <= 8


def test_grpo_rollout_rejects_prompt_that_fills_max_seq_length(monkeypatch):
    # A single prompt that already consumes max_seq_length leaves no trainable
    # completion; the rollout must fail fast rather than emit an empty/degenerate
    # loss span whose reward still folds into the group advantages.
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    trainer.tokenizer.encode = lambda s: list(range(40))  # len 40
    trainer.args.max_seq_length = 32

    monkeypatch.setattr(
        mlx_lm, "batch_generate",
        lambda *a, **k: types.SimpleNamespace(texts=["a", "b"], token_ids=[[7], [7]]),
        raising=False,
    )
    gen = trainer._grpo_rollout_generator()
    with pytest.raises(ValueError, match="no room for a completion"):
        next(gen)


def test_grpo_advantages_use_unbiased_group_std(monkeypatch):
    # TRL standardizes grouped rewards with the sample (Bessel-corrected) std
    # (torch.std default unbiased=True); the population std (mx.std default) is
    # smaller and inflates advantages by sqrt(N/(N-1)). For a 2-element group with
    # rewards {0, 1}: mean 0.5, unbiased std sqrt(0.5) ~= 0.7071 -> |adv| ~= 0.707,
    # whereas the population std 0.5 would give |adv| ~= 1.0. Assert the unbiased
    # scale so a regression back to population std is caught.
    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.0, 1.0]],
        batch_texts=["a", "b"],
    )
    gen = trainer._grpo_rollout_generator()
    _batch, _lengths, advantages = next(gen)
    adv = [float(x) for x in advantages.tolist()]
    assert abs(abs(adv[0]) - 0.7071) < 0.01, adv
    assert abs(abs(adv[1]) - 0.7071) < 0.01, adv
    # Sign: the below-mean sample is penalized, the above-mean sample rewarded.
    assert adv[0] < 0 < adv[1]


def test_grpo_rollout_falls_back_to_reencode_without_token_ids(monkeypatch):
    # On older mlx-lm whose batch_generate lacks return_token_ids, the rollout
    # must still build rows by re-encoding the completion text (no crash).
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["gen", "gen"],
    )

    def _old_generate(model, tokenizer, prompts=None, max_tokens=None,
                      sampler=None, verbose=False):
        return types.SimpleNamespace(texts=["gen", "gen"])

    monkeypatch.setattr(mlx_lm, "batch_generate", _old_generate, raising=False)

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _advantages = next(gen)
    # Fallback re-encodes rendered + c; encode() returns [1, 2, 3] here.
    assert batch.shape[0] == 2
    assert lengths.tolist()[0][0] == 3


def test_grpo_rollout_reencode_fallback_uses_common_prefix_boundary(monkeypatch):
    # On older mlx-lm (no return_token_ids) the rollout re-encodes rendered + c.
    # Encoding the concatenation is NOT a plain concat of encode(rendered) and
    # encode(c): a BPE/SentencePiece tokenizer can MERGE the boundary token (the
    # last prompt token fuses with the first completion token), so the standalone
    # prompt ids stop being a true prefix of the re-encoded row. Reusing
    # pe = len(pids) would then mask the wrong span (train a merged part-response
    # token as prompt / drop the first real completion token) and forward the
    # wrong completion_ids to reward funcs. The fallback must recompute the
    # boundary as the shared prefix of the prompt ids and the re-encoded row (the
    # same _common_prefix_len fix the preference path uses), not len(pids).
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["gen", "gen"],
    )
    # eos_token_id None so the EOS re-append does not fire, isolating the boundary.
    trainer.tokenizer.eos_token_id = None

    def _merge_encode(text, add_special_tokens=True):
        # Standalone prompt 'hello' -> [1, 2, 3]. The concatenated 'hello' + 'gen'
        # MERGES the boundary token 3 with the first completion token into a
        # single id 99: [1, 2, 99, 100]. The shared prefix is [1, 2] (length 2),
        # so the true response boundary is 2, NOT len(pids) == 3.
        if text == "hello":
            return [1, 2, 3]
        return [1, 2, 99, 100]

    trainer.tokenizer.encode = _merge_encode

    def _old_generate(model, tokenizer, prompts=None, max_tokens=None,
                      sampler=None, verbose=False):
        return types.SimpleNamespace(texts=["gen", "gen"])

    monkeypatch.setattr(mlx_lm, "batch_generate", _old_generate, raising=False)

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _adv = next(gen)
    lens = lengths.tolist()
    rows = batch.tolist()
    # Boundary is the shared-prefix length (2), NOT len(pids) == 3 (the merge bug).
    assert lens[0][0] == 2
    # The scored response span is the merged/real completion tokens after the seam.
    assert rows[0][lens[0][0]:lens[0][1]] == [99, 100]


def test_grpo_rollout_shuffles_prompt_order_by_default(monkeypatch):
    # TRL's GRPO samples prompts with a shuffling sampler by DEFAULT
    # (GRPOTrainer._get_train_sampler -> RepeatSampler(shuffle=shuffle_dataset),
    # and GRPOConfig.shuffle_dataset defaults to True), so a run capped by
    # max_steps below the dataset size still samples ACROSS the whole dataset. The
    # MLX rollout must likewise visit prompts in a seeded permutation by default
    # (a plain 0..N cycle would train only the ordered prefix under such a cap),
    # and honor an explicit sequential-order request.
    import types
    import mlx_lm
    import numpy as np
    from unsloth_zoo.mlx.utils import _normalize_seed

    class _Tok:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    monkeypatch.setattr(
        mlx_lm, "batch_generate",
        lambda *a, **k: types.SimpleNamespace(texts=["a", "b"]),
        raising=False,
    )

    seen = []

    def _reward(completions=None, prompts=None, **kw):
        seen.append(prompts[0])
        return [0.1, 0.2]

    N = 8
    seed = 123
    dataset = [{"prompt": f"p{i}", "answer": str(i)} for i in range(N)]

    # Default (shuffled): the visitation order is the seeded permutation.
    seen.clear()
    trainer = _make_ddp_grpo_trainer(0, 1, dataset, [_reward], _Tok())
    trainer.args.seed = seed
    gen = trainer._grpo_rollout_generator()
    for _ in range(N):
        next(gen)
    perm = np.random.RandomState(_normalize_seed(seed)).permutation(N)
    assert seen == [f"p{int(i)}" for i in perm]
    # A seeded permutation of 8 rows is not the identity, so the prefix-only bug
    # (visiting p0, p1, ... in order) is ruled out.
    assert seen != [f"p{i}" for i in range(N)]

    # Explicit sequential order is honored (byte-for-byte the old 0..N cycle).
    seen.clear()
    trainer = _make_ddp_grpo_trainer(0, 1, dataset, [_reward], _Tok())
    trainer.args.preserve_dataset_order = True
    gen = trainer._grpo_rollout_generator()
    for _ in range(N):
        next(gen)
    assert seen == [f"p{i}" for i in range(N)]


def test_create_preference_batches_samples_across_lengths_when_truncating():
    # With max_steps (num_batches) capping the run below the dataset size,
    # length-sorting then keeping the first chunks would train only on the
    # shortest pairs. A random subset must be taken before the sort instead.
    import numpy as np
    from unsloth_zoo.mlx.utils import (
        create_preference_batches,
        _normalize_seed,
    )

    class Tok:
        eos_token_id = 0

        def encode(self, s, add_special_tokens=True):
            return [1] * len(s)

    # 10 pairs; pair i has i+1 response tokens. Keep 2 of 10.
    N = 10
    dataset = [
        {"prompt": "", "chosen": "a" * (i + 1), "rejected": "a" * (i + 1)}
        for i in range(N)
    ]
    batches = create_preference_batches(
        dataset, Tok(), batch_size=1, max_seq_length=64,
        pad_to_multiple=0, num_batches=2, seed=42,
    )
    assert len(batches) == 2
    # Recover kept pair indices from the unpadded chosen length. Pair i has i+1
    # response tokens plus the appended EOS, so seq_end == i+2.
    kept = sorted(int(l.tolist()[0][1]) - 2 for _, l, _ in batches)
    order = np.random.RandomState(_normalize_seed(42)).permutation(N)
    expected = sorted(int(i) for i in order[:2])
    assert kept == expected
    # The length-sort-then-truncate bug would keep only the two shortest pairs.
    assert kept != [0, 1]


def test_create_preference_batches_guards_double_bos_on_chat_prompts():
    # A preference row whose prompt was already rendered with the chat template's
    # leading BOS (apply_chat_template(tokenize=False)) must be tokenized with
    # add_special_tokens=False, matching the SFT/GRPO encode_mlx_text guard. Raw
    # hf.encode would prepend a SECOND BOS, so DPO/ORPO would optimize
    # duplicate-BOS sequences (policy + reference, chosen + rejected) on every row.
    from unsloth_zoo.mlx.utils import create_preference_batches

    class Tok:
        eos_token_id = 0
        bos_token = "<s>"

        def __init__(self):
            self.add_special_tokens_seen = []

        def encode(self, text, add_special_tokens=True):
            self.add_special_tokens_seen.append(add_special_tokens)
            ids = [5] if add_special_tokens else []  # 5 stands in for BOS
            return ids + [ord(ch) for ch in text]

    tok = Tok()
    # Prompt already carries the rendered BOS; chosen/rejected are appended text.
    dataset = [{"prompt": "<s>hi", "chosen": "ok", "rejected": "no"}]
    create_preference_batches(
        dataset, tok, batch_size=1, max_seq_length=64,
        pad_to_multiple=0, num_batches=None, seed=0,
    )
    # All three encodes (prompt, prompt+chosen, prompt+rejected) run on
    # BOS-prefixed text, so none may re-add a second BOS.
    assert tok.add_special_tokens_seen  # encode actually ran
    assert all(flag is False for flag in tok.add_special_tokens_seen)


def test_preference_batches_render_conversational_rows_through_chat_template():
    # TRL DPO/ORPO accept conversational (message-list) preference rows and
    # normalize them via maybe_apply_chat_template (trl/data_utils.py:
    # apply_chat_template renders prompt / prompt+chosen / prompt+rejected as
    # strings, add_generation_prompt when the last prompt role is 'user'). The
    # MLX builder must render message-list prompt/chosen/rejected through the chat
    # template before encoding; string-concatenating and encoding the raw lists
    # would crash (a list has no .startswith, and encode rejects a list of dicts),
    # so common conversational preference datasets would fail before training.
    from unsloth_zoo.mlx.utils import create_preference_batches

    class _Tok:
        eos_token_id = 0
        bos_token = None

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False,
                                continue_final_message=False):
            parts = []
            for m in messages:
                tag = "U:" if m["role"] == "user" else "A:"
                parts.append(tag + m["content"])
            s = "".join(parts)
            if add_generation_prompt:
                s += "A:"
            return s

        def encode(self, text, add_special_tokens=True):
            # An unrendered message list would raise here (ord on a dict), so
            # reaching this with a plain string proves the chat render happened.
            return [ord(ch) for ch in text]

    dataset = [{
        "prompt": [{"role": "user", "content": "hi"}],
        "chosen": [{"role": "assistant", "content": "ok"}],
        "rejected": [{"role": "assistant", "content": "no"}],
    }]
    batch, lengths, _ = create_preference_batches(
        dataset, _Tok(), batch_size=1, max_seq_length=64,
        pad_to_multiple=0, num_batches=None, seed=0,
    )[0]
    lens = lengths.tolist()
    rows = batch.tolist()
    # prompt renders to 'U:hiA:' (6 chars); prompt+chosen to 'U:hiA:ok'. The
    # response boundary is the shared-prefix length 6, and the chosen/rejected
    # completions ('ok' / 'no') + appended EOS follow it.
    assert lens[0][0] == len("U:hiA:") == 6
    assert rows[0][lens[0][0]:lens[0][1]] == [ord("o"), ord("k"), 0]
    assert rows[1][lens[1][0]:lens[1][1]] == [ord("n"), ord("o"), 0]


def test_train_on_responses_only_rejects_preference_loss_types():
    # train_on_responses_only builds SFT-shaped batches on trainer._batches, which
    # _prepare_data returns before the DPO/ORPO branch; the preference loss would
    # then mis-split unrelated SFT rows into [chosen; rejected] pairs (NaN at
    # batch size 1). It must fail fast for a preference loss_type.
    import types
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    for lt in ("dpo", "orpo"):
        trainer = types.SimpleNamespace(args=types.SimpleNamespace(loss_type=lt))
        with pytest.raises(ValueError, match="incompatible with preference losses"):
            train_on_responses_only(trainer)

    # SFT is unaffected: the guard does not fire, so execution proceeds past it
    # (failing later on the missing tokenizer, a different error).
    sft = types.SimpleNamespace(
        args=types.SimpleNamespace(loss_type="sft"), tokenizer=None,
    )
    with pytest.raises(ValueError, match="tokenizer must be provided"):
        train_on_responses_only(sft)


def test_prepare_data_short_circuit_guards_preference_prebuilt_batches():
    # Defense in depth: prebuilt SFT batches (_batches) must never be handed to
    # the DPO/ORPO preference losses; the short-circuit must raise for those
    # loss_types before returning them.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._prepare_data)
    sc = src.index("if self._batches is not None:")
    ret = src.index("return self._batches, None", sc)
    guard = src[sc:ret]
    assert 'loss_type", "sft") in ("orpo", "dpo")' in guard
    assert "raise ValueError" in guard


def test_grpo_rollout_guards_double_bos_on_chat_prompts(monkeypatch):
    # A chat template that already renders a leading BOS must be tokenized with
    # add_special_tokens=False on the GRPO rollout path (matching the SFT
    # encode_mlx_text guard), otherwise the rollout prompt and the training rows
    # begin with two BOS tokens and the prompt distribution is corrupted. Both
    # the prompt encode and the prompt+completion encode must take the guard.
    import types
    import mlx_lm
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    class _Tok:
        eos_token_id = 0
        bos_token = "<s>"

        def __init__(self):
            self.add_special_tokens_seen = []

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False,
                                continue_final_message=False):
            # Llama/Qwen/Gemma-style: template emits a leading BOS.
            return "<s>USER: " + messages[-1]["content"]

        def encode(self, text, add_special_tokens=True):
            self.add_special_tokens_seen.append(add_special_tokens)
            return [1, 2, 3]

    resp = types.SimpleNamespace(texts=["a", "b"])
    monkeypatch.setattr(mlx_lm, "batch_generate",
                        lambda *a, **k: resp, raising=False)

    class _Model:
        # Minimal mlx nn.Module stand-in so the rollout dropout guard can toggle
        # eval()/train() around generation.
        def __init__(self):
            self.training = True
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = _Model()
    tok = _Tok()
    trainer.tokenizer = tok
    trainer.train_dataset = [
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "x"}
    ]
    trainer.reward_funcs = [
        lambda completions=None, prompts=None, **kw: [0.1, 0.2]
    ]
    trainer.args = types.SimpleNamespace(
        num_generations=2, temperature=1.0,
        max_completion_length=4, max_seq_length=32,
    )

    gen = trainer._grpo_rollout_generator()
    next(gen)
    # The rendered chat prompt already carries a BOS, so every encode on this
    # rollout (1 prompt + 1 per completion) must suppress the extra special
    # token; no call may re-add a second BOS.
    assert tok.add_special_tokens_seen  # encode actually ran
    assert all(flag is False for flag in tok.add_special_tokens_seen)


def test_model_has_non_lora_trainable_params_detects_full_module():
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import model_has_non_lora_trainable_params

    class _LoRAMod:
        def __init__(self):
            self.lora_a = mx.zeros((4, 2))
            self.lora_b = mx.zeros((2, 4))
            self.scale = 1.0

    class _Model:
        def __init__(self, extra):
            self._lora = _LoRAMod()
            self._extra = extra  # "none" | "lm_head" | "wrapped_base"

        def named_modules(self):
            return [("", self), ("q_proj", self._lora)]

        def parameters(self):
            params = {"q_proj": {
                "lora_a": self._lora.lora_a, "lora_b": self._lora.lora_b,
            }}
            if self._extra == "lm_head":
                params["lm_head"] = {"weight": mx.zeros((8, 4))}
            elif self._extra == "wrapped_base":
                # A base weight INSIDE the LoRA module must be excluded.
                params["q_proj"]["weight"] = mx.zeros((4, 4))
            return params

        def trainable_parameters(self):
            return self.parameters()

    # A directly-trained non-LoRA module (lm_head) is detected: such a tensor
    # keeps moving so a LoRA-disable reference would not be the frozen policy.
    assert model_has_non_lora_trainable_params(_Model("lm_head")) is True
    # Pure LoRA (only adapter tensors trainable): reference is safe.
    assert model_has_non_lora_trainable_params(_Model("none")) is False
    # A wrapped base weight inside a LoRA module is not a non-LoRA trainable.
    assert model_has_non_lora_trainable_params(_Model("wrapped_base")) is False


def test_preference_reference_guards_reject_non_lora_trainables():
    # The DPO and GRPO loss setup must refuse a LoRA reference when non-LoRA
    # params are also trainable, otherwise the reference (LoRA-disable) silently
    # keeps the moving non-LoRA weights and the gradient is wrong.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert src.count("model_has_non_lora_trainable_params(model)") >= 2
    dpo_builder = src.index("make_dpo_loss_fn")
    grpo_builder = src.index("make_grpo_loss_fn")
    # The DPO guard precedes the DPO loss builder; the GRPO guard sits between
    # the DPO and GRPO loss builders (i.e. before the GRPO one).
    assert src.index("model_has_non_lora_trainable_params(model)") < dpo_builder
    assert src.index("model_has_non_lora_trainable_params(model)", dpo_builder) \
        < grpo_builder


def test_preference_reference_guards_reject_dora_adapters():
    # DPO/GRPO obtain the reference by zeroing the LoRA scale, but a DoRA layer
    # still applies its trainable magnitude m/||W|| (which drifts as m trains),
    # so the reference would no longer be the frozen base. The setup must reject
    # DoRA for referenced runs (before building the DPO/GRPO loss), gated so that
    # plain LoRA is untouched.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    guard = 'type(m).__name__.startswith("DoRA")'
    assert src.count(guard) >= 2
    dpo_builder = src.index("make_dpo_loss_fn")
    grpo_builder = src.index("make_grpo_loss_fn")
    # DoRA guard precedes the DPO loss builder ...
    assert src.index(guard) < dpo_builder
    # ... and a second DoRA guard sits before the GRPO loss builder.
    assert src.index(guard, dpo_builder) < grpo_builder


def test_grpo_rollout_skips_eos_reappend_for_multi_eos_tokenizer(monkeypatch):
    # mlx-lm stops on ANY id in the tokenizer's eos set and strips whichever one
    # stopped the row without surfacing which. For a multi-eos model, re-appending
    # the singular hf.eos_token_id could train the WRONG terminal token, so the
    # rollout must skip the re-append when the eos is ambiguous (a single-eos
    # tokenizer keeps re-appending, as the other tests assert).
    import types
    import mlx_lm

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    # eos_token_id is 0 (see _Tok), but the model can also stop on 99.
    trainer.tokenizer.eos_token_ids = [0, 99]

    def _gen(model, tokenizer, prompts=None, max_tokens=None,
             sampler=None, verbose=False, return_token_ids=False):
        # Both rows stopped early (len < max_completion_length 4).
        return types.SimpleNamespace(texts=["a", "b"], token_ids=[[7], [7, 8]])

    monkeypatch.setattr(mlx_lm, "batch_generate", _gen, raising=False)

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _adv = next(gen)
    rows = batch.tolist()
    lens = lengths.tolist()
    pe = lens[0][0]
    # No EOS re-appended for the ambiguous multi-eos tokenizer.
    assert rows[0][pe:lens[0][1]] == [7]
    assert rows[1][pe:lens[1][1]] == [7, 8]


def test_grpo_rollout_renders_chat_prompt_through_wrapper_not_inner(monkeypatch):
    # mlx-lm's TokenizerWrapper.apply_chat_template can honor a wrapper-level
    # custom template (chat_template_type / _chat_template from the model's
    # tokenizer config) that the inner HF tokenizer (_tokenizer) does not carry.
    # The GRPO rollout must render conversational prompts through the wrapper
    # (self.tokenizer) -- matching the SFT/DPO render path and the chat_template
    # override normalized in _prepare_data -- not through the unwrapped inner HF
    # tokenizer, which would render with the wrong template or raise. hf (the
    # inner tokenizer) is reserved for raw encoding only.
    import types
    import mlx_lm
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    encoded_texts = []

    class _InnerHF:
        # The unwrapped HF tokenizer: lacks the wrapper-level chat template.
        eos_token_id = 0
        bos_token = None

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError(
                "GRPO rendered through the inner HF tokenizer, which lacks the "
                "wrapper-level chat template"
            )

        def encode(self, text, add_special_tokens=True):
            encoded_texts.append(text)
            return [1, 2, 3]

    class _Wrapper:
        # mlx-lm TokenizerWrapper stand-in: a wrapper-level apply_chat_template
        # plus a distinct inner _tokenizer used only for raw encoding.
        eos_token_id = 0

        def __init__(self, inner):
            self._tokenizer = inner

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False,
                                continue_final_message=False):
            return "WRAP:" + messages[-1]["content"]

    resp = types.SimpleNamespace(texts=["a", "b"])
    monkeypatch.setattr(mlx_lm, "batch_generate",
                        lambda *a, **k: resp, raising=False)

    class _Model:
        def __init__(self):
            self.training = True
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    inner = _InnerHF()
    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = _Model()
    trainer.tokenizer = _Wrapper(inner)
    trainer.train_dataset = [
        {"prompt": [{"role": "user", "content": "hi"}], "answer": "x"}
    ]
    trainer.reward_funcs = [
        lambda completions=None, prompts=None, **kw: [0.1, 0.2]
    ]
    trainer.args = types.SimpleNamespace(
        num_generations=2, temperature=1.0,
        max_completion_length=4, max_seq_length=32,
    )

    gen = trainer._grpo_rollout_generator()
    next(gen)  # must NOT raise: rendering goes through the wrapper
    # The wrapper's rendered output ("WRAP:...") is what gets encoded, proving the
    # render used the wrapper template rather than the inner HF tokenizer.
    assert encoded_texts and encoded_texts[0].startswith("WRAP:")


def test_grpo_rollout_excludes_view_added_token_columns_from_reward_kwargs(monkeypatch):
    # When a GRPO dataset carries a completion column, the base tokenized dataset
    # view (_MLXTokenizedDatasetView) injects synthetic input_ids/attention_mask
    # onto every row for SFT parity. TRL builds reward kwargs from the original
    # dataset columns only (excluding prompt/completion/completion_ids) and never
    # forwards tokenized input_ids/attention_mask, so a valid reward callback with
    # a strict signature (accepts its real columns but no token kwargs) must not
    # crash. The rollout must therefore source reward kwargs from the original
    # rows, not the view.
    import types
    import mlx_lm
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer, _MLXTokenizedDatasetView

    class _Tok:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    resp = types.SimpleNamespace(texts=["a", "b"])
    monkeypatch.setattr(mlx_lm, "batch_generate",
                        lambda *a, **k: resp, raising=False)

    class _Model:
        def __init__(self):
            self.training = True
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    tok = _Tok()
    # A prompt+completion GRPO dataset (plain strings need no chat template): the
    # view renders "p"+"c" and so injects input_ids/attention_mask onto the row.
    original = [{"prompt": "p", "completion": "c", "answer": "a"}]
    view = _MLXTokenizedDatasetView(original, tok, 32)
    # Sanity: the view really does inject the synthetic token columns.
    assert "input_ids" in list(view)[0]
    assert "attention_mask" in list(view)[0]

    # Strict-signature reward: accepts the real 'answer' column but NOT the
    # synthetic token columns; it raises TypeError if input_ids/attention_mask leak
    # in (as they did before the fix). No **kwargs on purpose.
    def _reward(completions=None, prompts=None, answer=None):
        return [0.1, 0.2]

    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = _Model()
    trainer.tokenizer = tok
    trainer.train_dataset = view
    trainer._mlx_train_dataset_for_batches = original
    trainer.reward_funcs = [_reward]
    trainer.args = types.SimpleNamespace(
        num_generations=2, temperature=1.0,
        max_completion_length=4, max_seq_length=32,
    )

    gen = trainer._grpo_rollout_generator()
    batch, lengths, advantages = next(gen)  # must not raise on leaked token kwargs
    assert batch.shape[0] == 2
    assert advantages.shape[0] == 2


def test_grpo_rollout_passes_generated_completion_ids_to_reward_funcs(monkeypatch):
    # TRL forwards the GENERATED completion_ids (per-completion token id lists) to
    # every reward function so token-level rewards (length / special-token counts)
    # can score the exact sampled tokens. The MLX rollout must do the same: a
    # reward callback that declares a completion_ids parameter must receive the
    # generated ids used to build the loss rows (the [response_start, end) span),
    # not any dataset completion_ids column. The pass is signature-gated so a
    # strict-signature reward (no completion_ids / no **kwargs) still works.
    import types
    import mlx_lm
    from unsloth_zoo.mlx.trainer import (
        MLXGRPOTrainer,
        _reward_func_wants_completion_ids,
    )

    # Unit-level: the signature gate accepts a completion_ids param or **kwargs and
    # rejects a strict signature that has neither.
    assert _reward_func_wants_completion_ids(
        lambda completions, prompts, completion_ids: None)
    assert _reward_func_wants_completion_ids(
        lambda completions, prompts, **kw: None)
    assert not _reward_func_wants_completion_ids(
        lambda completions, prompts, answer: None)

    class _Tok:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    # Newer mlx-lm surfaces the generated ids (its signature has return_token_ids),
    # so the rollout takes the comp_ids branch and scores the exact sampled tokens.
    def _gen(model, tokenizer, prompts=None, max_tokens=None, sampler=None,
             verbose=False, return_token_ids=False):
        return types.SimpleNamespace(
            texts=["aa", "bb"], token_ids=[[10, 11], [12, 13]])

    monkeypatch.setattr(mlx_lm, "batch_generate", _gen, raising=False)

    class _Model:
        def __init__(self):
            self.training = True
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    captured = {}

    # Token-level reward: declares completion_ids, so it must receive the generated
    # ids. Also carries **kwargs to absorb the passed-through 'answer' column.
    def reward_with_ids(completions=None, prompts=None, completion_ids=None, **kw):
        captured["completion_ids"] = completion_ids
        return [0.1, 0.2]

    # Strict-signature reward (round-3 back-compat): accepts its real 'answer'
    # column but NOT completion_ids and has no **kwargs; it must NOT be handed the
    # generated completion_ids (that would raise TypeError: unexpected keyword).
    def reward_strict(completions=None, prompts=None, answer=None):
        captured["strict_ran"] = True
        return [0.3, 0.4]

    tok = _Tok()
    original = [{"prompt": "p", "answer": "a"}]
    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = _Model()
    trainer.tokenizer = tok
    trainer.train_dataset = original
    trainer.reward_funcs = [reward_with_ids, reward_strict]
    trainer.args = types.SimpleNamespace(
        num_generations=2, temperature=1.0,
        max_completion_length=4, max_seq_length=32,
    )

    gen = trainer._grpo_rollout_generator()
    batch, lengths, advantages = next(gen)  # strict reward must not raise
    rows = batch.tolist()
    lens = lengths.tolist()
    pe = lens[0][0]
    # The completion_ids the reward received are the GENERATED ids that build the
    # loss rows: exactly each row's scored [pe, end) span (generated ids [10, 11] /
    # [12, 13] with the appended EOS 0), not the dataset column.
    assert captured["completion_ids"] == [
        rows[0][pe:lens[0][1]],
        rows[1][pe:lens[1][1]],
    ]
    assert captured["completion_ids"] == [[10, 11, 0], [12, 13, 0]]
    # The strict-signature reward still ran (received no completion_ids keyword).
    assert captured["strict_ran"] is True


def test_grpo_rollout_reencode_fallback_appends_eos_on_normal_stop(monkeypatch):
    # On older mlx-lm without return_token_ids, the rollout falls back to
    # re-encoding the decoded completion text. batch_generate strips the terminal
    # EOS from resp.texts on a normal stop, so a completion that stopped before the
    # generation cap has no EOS to score. Mirroring the comp_ids branch (and TRL,
    # whose completion mask is inclusive of the EOS), the fallback must re-append
    # the EOS for a normally-stopped row and leave a length-truncated row (its span
    # == the cap) untouched.
    import types
    import mlx_lm
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    class _Tok:
        eos_token_id = 7
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            # Deterministic map so the re-encoded completion span is controllable:
            # prompt "P" -> 3 tokens (pe = 3); "Pshort" adds 1 completion token
            # (stopped early, < cap 4); "Plong" adds 4 (filled the cap == 4).
            table = {
                "P": [1, 2, 3],
                "Pshort": [1, 2, 3, 4],
                "Plong": [1, 2, 3, 4, 5, 6, 7],
            }
            return list(table[text])

    # Older batch_generate: no return_token_ids parameter -> fallback re-encode.
    def _old_generate(model, tokenizer, prompts=None, max_tokens=None,
                      sampler=None, verbose=False):
        return types.SimpleNamespace(texts=["short", "long"])

    monkeypatch.setattr(mlx_lm, "batch_generate", _old_generate, raising=False)

    class _Model:
        def __init__(self):
            self.training = True
        def eval(self):
            self.training = False
        def train(self, mode=True):
            self.training = mode

    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = _Model()
    trainer.tokenizer = _Tok()
    trainer.train_dataset = [{"prompt": "P", "answer": "x"}]
    trainer.reward_funcs = [
        lambda completions=None, prompts=None, **kw: [0.1, 0.2]
    ]
    trainer.args = types.SimpleNamespace(
        num_generations=2, temperature=1.0,
        max_completion_length=4, max_seq_length=32,
    )

    gen = trainer._grpo_rollout_generator()
    batch, lengths, _adv = next(gen)
    rows = batch.tolist()
    lens = lengths.tolist()
    pe = lens[0][0]
    assert pe == 3
    # Row 0 stopped early (1 completion token < cap 4): EOS (7) re-appended.
    assert rows[0][pe:lens[0][1]] == [4, 7]
    # Row 1 filled the cap (4 completion tokens == 4): length-truncated, no EOS.
    assert rows[1][pe:lens[1][1]] == [4, 5, 6, 7]


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


def test_grpo_loss_temperature_scales_logits():
    # TRL scales logits by the sampling temperature before every log-prob used in
    # the GRPO loss (policy/old/reference: grpo_trainer.py logits = logits /
    # self.temperature), so the policy-gradient magnitude (~1/T) and the k3 KL term
    # match TRL at temperature != 1.0; they agree exactly at the default 1.0. The
    # untempered version diverged from TRL whenever a user set temperature != 1.
    import inspect
    from unsloth_zoo.mlx.utils import make_grpo_loss_fn
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXGRPOTrainer

    sig = inspect.signature(make_grpo_loss_fn)
    assert "temperature" in sig.parameters
    assert sig.parameters["temperature"].default == 1.0
    lsrc = inspect.getsource(make_grpo_loss_fn)
    assert "logits = logits / temperature" in lsrc
    # The trainer passes the rollout sampling temperature into the loss builder ...
    tsrc = inspect.getsource(MLXTrainer._train_inner)
    assert 'temperature=float(getattr(args, "temperature", 1.0)' in tsrc
    # ... and the logged-KL probe (MLXGRPOTrainer) scales logits by the same temp.
    ksrc = inspect.getsource(MLXGRPOTrainer._grpo_mean_kl)
    assert "logits = logits / _temp" in ksrc


# ---------------------------------------------------------------------------
# PR #832 review: MLX DDP sharding for GRPO/preference, and conversational GRPO
# reward-completion wrapping (TRL parity).
# ---------------------------------------------------------------------------

def test_preference_prepare_data_fails_fast_under_distributed_ddp():
    # Multi-GPU (MLX DDP) DPO/ORPO is NOT sharded: create_preference_batches has
    # no comm_group/rank argument, so every rank builds the same sorted
    # preference batches (identical seed/data/sort) and _apply_update all-reduces
    # duplicate gradients -- silently mistraining on one rank's stream instead of
    # the intended global shard (unlike the SFT path, which slices
    # items[rank::world_size]). Correctly sharding the concatenated
    # [chosen; rejected] builder cannot be validated on single-process CI, so the
    # preference path must fail fast under world_size > 1 rather than mistrain.
    import types
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXDPOConfig, MLXORPOConfig

    class _Tok:
        chat_template = "{{ messages }}"
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    for cfg_cls in (MLXDPOConfig, MLXORPOConfig):
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.args = cfg_cls()
        trainer.model = types.SimpleNamespace(_config={}, _hf_repo=None)
        trainer.tokenizer = _Tok()
        trainer.train_dataset = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
        trainer._batches = None
        # Simulate a 2-rank MLX DDP group (cached so no real init runs).
        trainer._distributed_initialized = True
        trainer._distributed_world = object()
        trainer._distributed_world_size = 2
        trainer._distributed_rank = 0
        trainer._distributed_is_main_process = True
        with pytest.raises(NotImplementedError, match="distributed"):
            MLXTrainer._prepare_data(trainer, is_vlm=False)


def _make_ddp_grpo_trainer(rank, world, train_dataset, reward_funcs, tokenizer):
    """MLXGRPOTrainer shell wired for a specific (rank, world) DDP group."""
    import types
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    class _Model:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = _Model()
    trainer.tokenizer = tokenizer
    trainer.train_dataset = train_dataset
    trainer.reward_funcs = list(reward_funcs)
    trainer.args = types.SimpleNamespace(
        num_generations=2, temperature=1.0,
        max_completion_length=4, max_seq_length=32,
    )
    trainer._distributed_initialized = True
    trainer._distributed_world = object()
    trainer._distributed_rank = rank
    trainer._distributed_world_size = world
    trainer._distributed_is_main_process = rank == 0
    return trainer


def test_grpo_rollout_shards_prompt_cycle_across_ddp_ranks(monkeypatch):
    # Under MLX DDP every rank must roll out a DISJOINT slice of prompts
    # (order[rank::world_size]) so the all-reduced gradient aggregates
    # world_size distinct prompts' groups per step, mirroring the SFT path's
    # items[rank::world_size] slicing. Without the offset every rank starts idx
    # at 0 and picks the SAME prompt for every microbatch (duplicated rollout /
    # reward work, prompt diversity of one rank). Rank 1 of a 2-rank group starts
    # at position 1 and strides by 2. This isolates the offset/stride mechanism
    # from the (now default) seeded prompt shuffle by requesting sequential order,
    # so the visitation order is the plain 0,1,2 cycle; the shuffle itself is
    # covered separately (test_grpo_rollout_shuffles_prompt_order_by_default).
    import types
    import mlx_lm

    class _Tok:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    monkeypatch.setattr(
        mlx_lm, "batch_generate",
        lambda *a, **k: types.SimpleNamespace(texts=["a", "b"]),
        raising=False,
    )

    seen_prompts = []

    def _reward(completions=None, prompts=None, **kw):
        seen_prompts.append(prompts[0])
        return [0.1, 0.2]

    dataset = [{"prompt": f"p{i}", "answer": str(i)} for i in range(4)]

    def _seq_trainer(rank, world):
        t = _make_ddp_grpo_trainer(rank, world, dataset, [_reward], _Tok())
        t.args.dataset_order = "sequential"
        return t

    # Rank 1 of world 2: odd-indexed prompts 1, 3.
    seen_prompts.clear()
    gen = _seq_trainer(1, 2)._grpo_rollout_generator()
    next(gen)
    next(gen)
    assert seen_prompts == ["p1", "p3"]

    # Rank 0 of world 2: even-indexed prompts 0, 2 -- disjoint from rank 1.
    seen_prompts.clear()
    gen = _seq_trainer(0, 2)._grpo_rollout_generator()
    next(gen)
    next(gen)
    assert seen_prompts == ["p0", "p2"]

    # Single GPU (world 1): sequential cycle 0, 1, 2.
    seen_prompts.clear()
    gen = _seq_trainer(0, 1)._grpo_rollout_generator()
    next(gen)
    next(gen)
    next(gen)
    assert seen_prompts == ["p0", "p1", "p2"]


def test_grpo_rollout_wraps_conversational_completions_for_reward_funcs(monkeypatch):
    # TRL converts the generated completion TEXT into assistant-message lists
    # before calling reward functions when the prompt is conversational
    # (grpo_trainer.py: completions.append([{"role":"assistant","content":
    # bootstrap + completion}])), so a chat-style reward fn indexing message
    # roles/content receives the expected shape. A raw generated string would
    # crash such a reward fn. Plain-string prompts keep raw-string completions.
    import types
    import mlx_lm

    class _Tok:
        eos_token_id = 0
        bos_token = None

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False,
                                continue_final_message=False):
            return "RENDER:" + messages[-1]["content"]

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    monkeypatch.setattr(
        mlx_lm, "batch_generate",
        lambda *a, **k: types.SimpleNamespace(texts=["hello", "world"]),
        raising=False,
    )

    seen = {}

    def _chat_reward(completions=None, prompts=None, **kw):
        # Indexes message structure -- would crash on a raw string completion.
        seen["completions"] = completions
        seen["prompts"] = prompts
        return [float(len(c[0]["content"])) for c in completions]

    # Conversational prompt (last role user): completions wrapped as assistant
    # messages with no bootstrap prefix; the reward prompt is the message list.
    seen.clear()
    trainer = _make_ddp_grpo_trainer(
        0, 1, [{"prompt": [{"role": "user", "content": "hi"}], "answer": "x"}],
        [_chat_reward], _Tok(),
    )
    next(trainer._grpo_rollout_generator())
    assert seen["completions"] == [
        [{"role": "assistant", "content": "hello"}],
        [{"role": "assistant", "content": "world"}],
    ]
    assert seen["prompts"][0] == [{"role": "user", "content": "hi"}]

    # Continued final assistant message: TRL drops it from the reward prompt and
    # bootstraps each completion with its content.
    seen.clear()
    trainer = _make_ddp_grpo_trainer(
        0, 1,
        [{
            "prompt": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "PRE"},
            ],
            "answer": "x",
        }],
        [_chat_reward], _Tok(),
    )
    next(trainer._grpo_rollout_generator())
    assert seen["completions"] == [
        [{"role": "assistant", "content": "PREhello"}],
        [{"role": "assistant", "content": "PREworld"}],
    ]
    assert seen["prompts"][0] == [{"role": "user", "content": "hi"}]

    # Plain-string prompt: raw-string completions (back-compat / no-op).
    seen.clear()

    def _string_reward(completions=None, prompts=None, **kw):
        seen["completions"] = completions
        return [float(len(c)) for c in completions]

    trainer = _make_ddp_grpo_trainer(
        0, 1, [{"prompt": "plain prompt", "answer": "x"}],
        [_string_reward], _Tok(),
    )
    next(trainer._grpo_rollout_generator())
    assert seen["completions"] == ["hello", "world"]


# ---------------------------------------------------------------------------
# PR #832 round-7 review fixes: mlx-lm floor, empty rewards, streaming guards,
# unsupported loss_type, GRPO train_on_responses_only.
# ---------------------------------------------------------------------------
def test_grpo_rollout_raises_clear_error_when_batch_generate_missing(monkeypatch):
    # batch_generate was added to mlx-lm 0.28.0, but the pyproject mlx extra only
    # pins mlx-lm>=0.22.0. On a declared-compatible 0.22-0.27.x install the
    # `from mlx_lm import batch_generate` in the rollout would raise a cryptic
    # "cannot import name 'batch_generate'". The rollout must surface a clear,
    # actionable message naming the required floor instead.
    import sys

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    # Simulate an mlx-lm older than 0.28.0 (no batch_generate export). The test
    # mlx_lm stub fabricates any missing attribute via __getattr__, so swap in a
    # bare module that lacks batch_generate to make `from mlx_lm import
    # batch_generate` raise ImportError the way a real old install would.
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.__version__ = "0.27.1"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    gen = trainer._grpo_rollout_generator()
    with pytest.raises(ImportError, match=r"mlx-lm>=0\.28\.0"):
        next(gen)


def test_grpo_trainer_rejects_empty_reward_funcs():
    # An empty reward_funcs list is accepted by the list coercion, but then every
    # completion scores 0, all group-relative advantages are 0, and the policy
    # receives no reward gradient (a silent no-op update). TRL's GRPOTrainer
    # requires at least one reward function; fail fast at construction.
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    with pytest.raises(ValueError, match="at least one reward function"):
        MLXGRPOTrainer(
            model=None, tokenizer=None, train_dataset=None, reward_funcs=[],
        )


def test_prepare_data_rejects_unsupported_loss_type():
    # A real TRL DPO loss variant that is not implemented on MLX ("ipo", "sigmoid",
    # ...) or a typo would otherwise fall through to the SFT/CCE path and silently
    # train a plain cross-entropy objective. Fail fast on any loss_type outside
    # {sft, orpo, dpo} (grpo is handled by MLXGRPOTrainer's own guard).
    from unsloth_zoo.mlx.trainer import MLXTrainer

    for lt in ("ipo", "sigmoid", "hinge", "kto", "dppo"):
        inst = MLXTrainer.__new__(MLXTrainer)
        inst.args = types.SimpleNamespace(loss_type=lt)
        with pytest.raises(ValueError, match="unsupported loss_type"):
            MLXTrainer._prepare_data(inst, is_vlm=False)


def test_preference_prepare_data_fails_fast_under_streaming():
    # create_preference_batches materializes the WHOLE dataset before batching, so
    # an unbounded streaming IterableDataset would hang or OOM instead of stopping
    # at max_steps. The preference path has no bounded streaming iterator (unlike
    # SFT/VLM), so it must fail fast under streaming=True.
    import types as _types
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXDPOConfig, MLXORPOConfig

    class _Tok:
        chat_template = "{{ messages }}"
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    for cfg_cls in (MLXDPOConfig, MLXORPOConfig):
        trainer = MLXTrainer.__new__(MLXTrainer)
        trainer.args = cfg_cls(streaming=True)
        trainer.model = _types.SimpleNamespace(_config={}, _hf_repo=None)
        trainer.tokenizer = _Tok()
        trainer.train_dataset = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
        trainer._batches = None
        # Single-process group (cached so no real init runs): the DDP guard is a
        # no-op here, so the streaming guard is what must fire.
        trainer._distributed_initialized = True
        trainer._distributed_world = None
        trainer._distributed_world_size = 1
        trainer._distributed_rank = 0
        trainer._distributed_is_main_process = True
        with pytest.raises(NotImplementedError, match="streaming DPO/ORPO"):
            MLXTrainer._prepare_data(trainer, is_vlm=False)


def test_grpo_prepare_data_fails_fast_under_streaming(monkeypatch):
    # The GRPO rollout generator materializes the whole dataset (walks every prompt
    # and lists all rows) before the first generation, so an unbounded streaming
    # dataset would hang before a single rollout despite max_steps. GRPO has no
    # streaming rollout path, so _prepare_data must fail fast under streaming=True.
    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    trainer.args.streaming = True
    with pytest.raises(NotImplementedError, match="streaming GRPO"):
        trainer._prepare_data(is_vlm=False)


def test_train_on_responses_only_rejects_grpo_loss_type():
    # train_on_responses_only builds response-only SFT batches and replaces the
    # training dataset, but GRPO trains from rollouts that need the original prompt
    # rows (and per-row reward kwargs). It is not caught by the DPO/ORPO guard, so
    # it must fail fast for grpo too rather than silently corrupt the rollout data.
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    trainer = types.SimpleNamespace(args=types.SimpleNamespace(loss_type="grpo"))
    with pytest.raises(ValueError, match="GRPO trains from rollouts"):
        train_on_responses_only(trainer)


def test_train_on_responses_only_return_function_allowed_for_grpo(monkeypatch):
    # return_function=True only builds and returns the masking closure; it never
    # touches trainer._batches or the dataset, and reusing the trainer's
    # tokenizer/template autodetection is a supported call. The grpo/dpo/orpo
    # guards must therefore skip the closure-return path and only block the
    # mutating trainer path (which stays rejected by the test above).
    import unsloth_zoo.dataset_utils as dataset_utils
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    class CallableTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    def fake_hf(trainer, *, instruction_part=None, response_part=None,
                force_match=True, tokenizer=None, return_function=False,
                num_proc=None, last_response_only=False):
        return lambda batch: batch

    monkeypatch.setattr(dataset_utils, "train_on_responses_only", fake_hf)
    for loss_type in ("grpo", "dpo", "orpo"):
        trainer = types.SimpleNamespace(
            args=types.SimpleNamespace(loss_type=loss_type),
            tokenizer=CallableTokenizer(),
        )
        fn = train_on_responses_only(
            trainer,
            instruction_part="<user>",
            response_part="<assistant>",
            return_function=True,
        )
        assert callable(fn)


# ---------------------------------------------------------------------------
# PR #832 review: GRPO resume must not replay rollouts, the MLX RNG must be
# seeded from args.seed for reproducible sampling, temperature <= 0 must fail
# fast (silent no-op), and reward funcs declaring trainer_state must receive a
# minimal state (TRL parity).
# ---------------------------------------------------------------------------

def test_grpo_resume_skips_rollout_replay_and_advances_prompt_cursor(monkeypatch):
    # On resume, _train_inner fast-forwards a non-None batch_iter to the resume
    # position. For GRPO that iterator is the on-policy rollout generator, so the
    # generic consume-based fast-forward would REGENERATE num_generations
    # completions per skipped step with the already-updated checkpoint model and
    # re-run the (possibly side-effecting) reward funcs -- slow, non-reproducible,
    # and divergent from the killed run. The GRPO override must instead advance
    # only the prompt cursor: no generation, no scoring, landing on the correct
    # resume prompt.
    import mlx_lm

    class _Tok:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    calls = {"generate": 0}

    def _gen(model, tokenizer, prompts=None, max_tokens=None, sampler=None,
             verbose=False, return_token_ids=False):
        calls["generate"] += 1
        return types.SimpleNamespace(texts=["a", "b"], token_ids=[[7], [8]])

    monkeypatch.setattr(mlx_lm, "batch_generate", _gen, raising=False)

    seen_prompts = []

    def _reward(completions=None, prompts=None, **kw):
        seen_prompts.append(prompts[0])
        return [0.1, 0.2]

    N = 6
    dataset = [{"prompt": f"p{i}", "answer": str(i)} for i in range(N)]
    trainer = _make_ddp_grpo_trainer(0, 1, dataset, [_reward], _Tok())
    trainer.args.preserve_dataset_order = True  # sequential 0..N cursor

    old_gen = trainer._grpo_rollout_generator()
    new_gen = trainer._fast_forward_resume_batches(old_gen, 3)

    # The resume fast-forward regenerated / re-scored NOTHING.
    assert calls["generate"] == 0
    assert seen_prompts == []

    # The first post-resume rollout lands on prompt index 3 (where the killed run
    # would train next), and only THAT rollout generates.
    next(new_gen)
    assert calls["generate"] == 1
    assert seen_prompts == ["p3"]


def test_base_fast_forward_resume_consumes_stream_but_grpo_overrides():
    # The base (streaming SFT) fast-forward advances by consuming n_skip items;
    # GRPO overrides it so it does NOT consume its rollout generator.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXGRPOTrainer

    trainer = MLXTrainer.__new__(MLXTrainer)
    out = trainer._fast_forward_resume_batches(iter(range(10)), 4)
    assert next(out) == 4  # first four items were consumed

    # The override is a distinct implementation (no consume-based replay).
    assert (MLXGRPOTrainer._fast_forward_resume_batches
            is not MLXTrainer._fast_forward_resume_batches)
    grpo_src = inspect.getsource(MLXGRPOTrainer._fast_forward_resume_batches)
    assert "skip_rollouts" in grpo_src
    assert "next(" not in grpo_src  # no consume/regeneration in the GRPO path


def test_grpo_resume_cursor_matches_generic_fast_forward_position(monkeypatch):
    # The skip_rollouts cursor must land on EXACTLY the prompt the old
    # consume-based fast-forward reached (rank + n_skip * world for the next
    # rollout), so resume trains the same prompt sequence -- just without
    # regenerating the skipped ones. Verify against the reference cursor formula.
    import mlx_lm

    class _Tok:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    monkeypatch.setattr(
        mlx_lm, "batch_generate",
        lambda *a, **k: types.SimpleNamespace(texts=["a", "b"]),
        raising=False,
    )

    seen = []

    def _reward(completions=None, prompts=None, **kw):
        seen.append(prompts[0])
        return [0.1, 0.2]

    N = 5
    dataset = [{"prompt": f"p{i}", "answer": str(i)} for i in range(N)]
    # Rank 1 of a 2-rank group, sequential order: consume-based fast-forward would
    # leave idx at rank + n_skip*world = 1 + 3*2 = 7 -> prompt index 7 % 5 == 2.
    trainer = _make_ddp_grpo_trainer(1, 2, dataset, [_reward], _Tok())
    trainer.args.preserve_dataset_order = True
    gen = trainer._fast_forward_resume_batches(
        trainer._grpo_rollout_generator(), 3,
    )
    next(gen)
    assert seen == ["p2"]


def test_grpo_seeds_mlx_rng_from_args_seed_with_rank_offset(monkeypatch):
    # GRPO rollout sampling draws from the global MLX RNG, which nothing seeds
    # from args.seed. _maybe_seed_grpo_rng must seed it (rank-offset so DDP ranks
    # draw distinct completions yet stay deterministic), mirroring TRL's
    # set_seed(args.seed).
    import mlx.core as mx
    from unsloth_zoo.mlx.utils import _normalize_seed

    seen = {}
    monkeypatch.setattr(mx.random, "seed",
                        lambda s: seen.__setitem__("seed", s))

    trainer = _make_ddp_grpo_trainer(
        2, 4, [{"prompt": "p"}],
        [lambda completions=None, prompts=None, **kw: [0.0, 0.0]], object(),
    )
    trainer.args.loss_type = "grpo"
    trainer.args.seed = 100
    applied = trainer._maybe_seed_grpo_rng()
    assert applied == seen["seed"] == (_normalize_seed(100) + 2) % (2 ** 32)


def test_maybe_seed_grpo_rng_is_noop_for_non_grpo(monkeypatch):
    # SFT/DPO/ORPO do not sample; seeding is scoped to GRPO so their existing RNG
    # behavior is byte-for-byte unchanged.
    import mlx.core as mx

    seen = {}
    monkeypatch.setattr(mx.random, "seed",
                        lambda s: seen.__setitem__("seed", s))

    trainer = _make_ddp_grpo_trainer(
        0, 1, [{"prompt": "p"}],
        [lambda completions=None, prompts=None, **kw: [0.0, 0.0]], object(),
    )
    trainer.args.loss_type = "sft"
    assert trainer._maybe_seed_grpo_rng() is None
    assert "seed" not in seen


def test_train_inner_seeds_grpo_rng_before_capturing_state():
    # The seed must be applied BEFORE `state = [..., mx.random.state]` is captured,
    # so the compiled step threads the seeded RNG (else the captured reference
    # would be stale and the seed ignored by the grad step).
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    seed_call = src.index("self._maybe_seed_grpo_rng()")
    state_capture = src.index(
        "state = [model.state, optimizer.state, mx.random.state]"
    )
    assert seed_call < state_capture


@pytest.mark.parametrize("bad_temp", [0.0, -0.5])
def test_grpo_rollout_rejects_non_positive_temperature(monkeypatch, bad_temp):
    # temperature <= 0 -> greedy argmax -> identical completions per group ->
    # zero reward variance -> all advantages 0 -> silent no-op GRPO objective.
    # Fail fast, mirroring the num_generations<2 guard.
    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[lambda completions=None, prompts=None, **kw: [0.1, 0.2]],
        batch_texts=["a", "b"],
    )
    trainer.args.temperature = bad_temp
    gen = trainer._grpo_rollout_generator()
    with pytest.raises(ValueError, match="temperature > 0"):
        next(gen)


def test_grpo_config_default_temperature_is_positive():
    # The default must be > 0 so the new guard never fires on a normal run.
    from unsloth_zoo.mlx.trainer import MLXGRPOConfig

    assert MLXGRPOConfig().temperature > 0


def test_grpo_rollout_passes_trainer_state_to_declaring_reward(monkeypatch):
    # TRL forwards trainer_state (transformers TrainerState) to every reward func
    # for progress-aware (e.g. curriculum) shaping. A reward that declares a
    # REQUIRED trainer_state param would otherwise raise TypeError before the
    # first rollout. Signature-gate the pass and hand it a minimal state exposing
    # the real global_step / max_steps the trainer tracks.
    from unsloth_zoo.mlx.trainer import _reward_func_wants_trainer_state

    assert _reward_func_wants_trainer_state(
        lambda completions, prompts, trainer_state: None)
    assert _reward_func_wants_trainer_state(
        lambda completions, prompts, **kw: None)
    assert not _reward_func_wants_trainer_state(
        lambda completions, prompts, answer: None)

    captured = {}

    def curriculum_reward(completions, prompts, answer, trainer_state):
        # Required trainer_state (no default, no **kwargs): must be supplied.
        captured["global_step"] = trainer_state.global_step
        captured["max_steps"] = trainer_state.max_steps
        return [0.1, 0.2]

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[curriculum_reward], batch_texts=["a", "b"],
    )
    trainer._global_step = 5
    trainer._planned_total_steps = 20
    gen = trainer._grpo_rollout_generator()
    next(gen)  # must not raise TypeError for the required trainer_state param
    assert captured == {"global_step": 5, "max_steps": 20}


def test_grpo_rollout_omits_trainer_state_for_strict_reward(monkeypatch):
    # A strict-signature reward (no trainer_state, no **kwargs) must NOT be handed
    # a trainer_state keyword (that would raise TypeError: unexpected keyword).
    ran = {}

    def strict_reward(completions=None, prompts=None, answer=None):
        ran["ok"] = True
        return [0.1, 0.2]

    trainer = _make_grpo_trainer_for_rollout(
        monkeypatch, num_generations=2,
        reward_funcs=[strict_reward], batch_texts=["a", "b"],
    )
    gen = trainer._grpo_rollout_generator()
    next(gen)
    assert ran == {"ok": True}


# ---------------------------------------------------------------------------
# PR #832 review round: GRPO KL default (TRL GRPOConfig.beta 0.0), resume-
# reproducible per-rollout sampling seed, and TRL-parity full-tree dropout
# disable for DPO/ORPO.
# ---------------------------------------------------------------------------

def test_grpo_config_default_beta_is_zero_no_kl_path():
    # TRL GRPOConfig.beta defaults to 0.0 (reference model / KL loaded only when
    # set > 0). The MLX default must match: a plain MLXGRPOConfig() must not enable
    # the KL path, which would otherwise silently add an unrequested KL term for
    # LoRA runs and trip the no-reference guard for non-LoRA full-finetune runs.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXGRPOConfig
    from unsloth_zoo.mlx.utils import make_grpo_loss_fn

    assert MLXGRPOConfig().grpo_beta == 0.0
    # The loss builder's default matches, so the reference/KL branch is off ...
    assert inspect.signature(make_grpo_loss_fn).parameters["beta"].default == 0.0
    # ... and building the default loss with no LoRA reference does NOT raise the
    # no-reference guard (KL path is disabled at beta == 0).
    make_grpo_loss_fn()  # default beta 0.0, no lora_mods -> no reference/KL
    make_grpo_loss_fn(beta=0.0, lora_mods=None)
    # But an OPT-IN non-zero beta with no LoRA reference DOES raise, confirming the
    # guard fires only when KL is actually requested (so default runs never hit it).
    with pytest.raises(ValueError, match="not yet supported for full"):
        make_grpo_loss_fn(beta=0.04, lora_mods=None, reference_free=False)


def test_grpo_rollout_sampling_is_resume_reproducible(monkeypatch):
    # The round-8 fix seeds mx.random ONCE up front and, on resume, only advances
    # the prompt cursor -- so a resumed run's first rollout samples from the initial
    # RNG subsequence, not the state an uninterrupted run would hold after N
    # rollouts, and its completions/rewards/gradients diverge despite the same seed.
    # The rollout must instead derive a per-rollout seed from the rollout index, so
    # rollout k samples IDENTICALLY whether reached fresh or via resume.
    import mlx_lm

    class _Tok:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    def _gen(*a, **k):
        # Draw one token per completion from the (per-rollout-seeded) global RNG,
        # so the returned completions reflect the sampling RNG state. mx.random.seed
        # maps to torch.manual_seed under the shim, so torch's global RNG is the one
        # the rollout seeds.
        prompts = k.get("prompts")
        n = len(prompts)
        toks = [int(torch.randint(0, 10 ** 6, (1,)).item()) for _ in range(n)]
        return types.SimpleNamespace(texts=[f"c{t}" for t in toks])

    monkeypatch.setattr(mlx_lm, "batch_generate", _gen, raising=False)

    captured = []

    def _reward(completions=None, prompts=None, **kw):
        captured.append(list(completions))
        return [0.1, 0.2]

    N = 6
    dataset = [{"prompt": f"p{i}"} for i in range(N)]

    def _mk():
        t = _make_ddp_grpo_trainer(0, 1, dataset, [_reward], _Tok())
        t.args.preserve_dataset_order = True  # sequential cursor: index k -> prompt k
        t.args.seed = 42
        return t

    # Fresh run: step through rollouts 0..3, recording each rollout's completions.
    captured.clear()
    fresh = _mk()._grpo_rollout_generator()
    for _ in range(4):
        next(fresh)
    fresh_seq = list(captured)
    fresh_at_3 = fresh_seq[3]
    # The per-rollout seed genuinely varies the sampling (else a constant mock would
    # pass the resume check trivially): distinct rollout indices sample differently.
    assert fresh_seq[0] != fresh_seq[3]

    # Resumed run at index 3: the first (and only) rollout must REPRODUCE index 3
    # bit-for-bit, even though the fresh run drew RNG for indices 0..2 in between.
    captured.clear()
    resumed = _mk()._grpo_rollout_generator(skip_rollouts=3)
    next(resumed)
    assert captured[0] == fresh_at_3


def test_disable_dropout_neutralizes_base_model_dropout():
    # TRL's disable_dropout_in_model sets p = 0 on EVERY nn.Dropout, not only the
    # LoRA adapter dropout. The MLX disable must likewise walk the whole module tree
    # and neutralize a standalone (non-LoRA) Dropout in place. mlx.nn.Dropout is a
    # no-op when its keep-prob _p_1 == 1.0 (p == 0), so setting it there matches
    # TRL's module.p = 0.
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import _mlx_disable_lora_dropout

    class StubDropout(nn.Module):
        def __init__(self, p=0.5):
            super().__init__()
            self._p_1 = 1.0 - p  # mlx.nn.Dropout keep-prob convention

        def __call__(self, x):
            if self._p_1 == 1.0:
                return x
            return x * 0.0  # active dropout stand-in: zeros its input

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = StubDropout(0.5)

    root = Root()
    x = mx.ones((2, 4))
    # Active before the disable: the stub Dropout zeros its input.
    assert root.drop(x).tolist() == (x * 0.0).tolist()

    n = _mlx_disable_lora_dropout(root)

    # The one standalone (non-LoRA) Dropout was neutralized ...
    assert n == 1
    assert root.drop._p_1 == 1.0
    # ... so its forward is now identity, matching disable_dropout_in_model.
    assert root.drop(x).tolist() == x.tolist()


# ---------------------------------------------------------------------------
# PR #832 round-10 review fixes.
# ---------------------------------------------------------------------------


def test_preference_disable_dropout_preserves_saved_lora_dropout_metadata():
    # Regression (DROPOUT-METADATA, PORTED FROM #830 -- shared _mlx_disable_lora_
    # dropout + adapter-save read-sites): _mlx_disable_lora_dropout replaces each
    # adapter's dropout with _mlx_identity to make the ORPO/DPO forward
    # deterministic. That runs at setup, BEFORE save_model, and permanently
    # mutates the live module -- so the save path (_extract_mlx_lora_parameters /
    # save_model / _enrich_mlx_adapter_config), which reads lora_dropout back off
    # the live module, would record dropout=0.0 in adapter_config.json for a run
    # trained with nonzero lora_dropout, and the reloaded adapter would no longer
    # match the original LoRA config. The disable must stash the configured
    # probability first so the forward stays deterministic AND the saved metadata
    # stays correct.
    import mlx.core as mx
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import _mlx_disable_lora_dropout, _mlx_identity
    from unsloth_zoo.mlx.utils import (
        _extract_mlx_lora_parameters,
        _read_mlx_lora_dropout,
        _get_mlx_dropout_probability,
    )

    class _Dropout(nn.Module):
        # Stand-in for mlx.nn.Dropout: keep-prob in _p_1, identity when _p_1 == 1.
        def __init__(self, p):
            super().__init__()
            self._p_1 = 1.0 - p

        def __call__(self, x):
            return x

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = mx.zeros((4, 8))
            self.lora_b = mx.zeros((8, 4))
            self.dropout = _Dropout(0.25)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = Adapter()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block(), Block()]

    model = Model()
    # Baseline: the save path reads the configured lora_dropout off the live
    # module (rank comes from lora_a's trailing axis: shape (4, 8) -> 8).
    rank_before, _scale_before, dropout_before = _extract_mlx_lora_parameters(model)
    assert rank_before == 8
    assert dropout_before == pytest.approx(0.25)

    _mlx_disable_lora_dropout(model)

    # The forward is now a deterministic identity (dropout genuinely disabled)...
    for blk in model.layers:
        assert blk.q_proj.dropout is _mlx_identity
        assert _get_mlx_dropout_probability(blk.q_proj.dropout) == 0.0
        # ...but the stash preserves the configured probability for the save path.
        assert blk.q_proj._unsloth_orig_lora_dropout == pytest.approx(0.25)
        assert _read_mlx_lora_dropout(blk.q_proj) == pytest.approx(0.25)

    # So the adapter_config still reports the ORIGINAL dropout after ORPO/DPO,
    # not 0.0 -- reload parity with the trained LoRA config is preserved.
    rank_after, _scale_after, dropout_after = _extract_mlx_lora_parameters(model)
    assert rank_after == 8
    assert dropout_after == pytest.approx(0.25)


def test_preference_disable_dropout_all_save_read_sites_use_the_stash():
    # The stash only fixes the regression if EVERY adapter-save read-site routes
    # through _read_mlx_lora_dropout instead of reading dropout off the (now
    # identity) live module. Guard the three save read-sites structurally so a
    # future edit that reintroduces a raw _get_mlx_dropout_probability(module.
    # dropout) at save time is caught.
    import inspect
    from unsloth_zoo.mlx import trainer as trainer_mod, utils as utils_mod

    # main's #918 rework split save_model into a thin wrapper + _save_model_impl;
    # the adapter-save dropout read lives in the impl now.
    save_src = inspect.getsource(trainer_mod.MLXTrainer._save_model_impl)
    assert "_read_mlx_lora_dropout(m)" in save_src
    assert "_get_mlx_dropout_probability" not in save_src

    extract_src = inspect.getsource(utils_mod._extract_mlx_lora_parameters)
    assert "_read_mlx_lora_dropout(m)" in extract_src

    enrich_src = inspect.getsource(utils_mod._enrich_mlx_adapter_config)
    assert "_read_mlx_lora_dropout(module)" in enrich_src

    # The disable stashes before the identity swap.
    disable_src = inspect.getsource(trainer_mod._mlx_disable_lora_dropout)
    assert "_unsloth_orig_lora_dropout" in disable_src
    stash_at = disable_src.index("_unsloth_orig_lora_dropout")
    swap_at = disable_src.index("mod.dropout = _mlx_identity")
    assert stash_at < swap_at


def _make_grpo_trainer_for_prepare_data(prompts, args):
    import types
    from unsloth_zoo.mlx.trainer import MLXGRPOTrainer

    class _Tok:
        eos_token_id = 0
        chat_template = None

        def encode(self, text, add_special_tokens=True):
            return [1, 2, 3]

    trainer = MLXGRPOTrainer.__new__(MLXGRPOTrainer)
    trainer.model = types.SimpleNamespace(_config={}, _hf_repo=None)
    trainer.tokenizer = _Tok()
    trainer.train_dataset = [{"prompt": p} for p in prompts]
    trainer._distributed_world_size = 1
    trainer._distributed_initialized = True
    trainer.args = args
    return trainer


def test_grpo_prepare_data_plans_epoch_steps_for_finite_dataset():
    # Regression (EPOCH-GRPO): a finite GRPO run configured the standard HF/TRL
    # way (max_steps<=0, num_train_epochs>0) must NOT be rejected as streaming.
    # GRPO's _prepare_data returns batches=None (a rollout generator), so
    # _train_inner cannot count materialized batches; it previously fell into the
    # streaming branch and raised "num_train_epochs requires a finite dataset"
    # even though train_dataset is finite. _prepare_data now plans the step total
    # from the finite prompt set, mirroring the SFT/DPO epoch formula
    # (batches_per_epoch * num_epochs // grad_accum).
    import types

    template = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"

    # 4 prompts, grad_accum=1, world=1, 2 epochs -> (4 * 2) // 1 = 8 steps.
    args = types.SimpleNamespace(
        chat_template=template,
        max_steps=-1,
        num_train_epochs=2,
        gradient_accumulation_steps=1,
        streaming=False,
    )
    trainer = _make_grpo_trainer_for_prepare_data(["a", "b", "c", "d"], args)
    batches, batch_iter = trainer._prepare_data(is_vlm=False)
    assert batches is None
    assert batch_iter is not None
    assert trainer._grpo_epoch_total_steps == 8

    # grad_accum=2 halves the optimizer steps: (4 * 3) // 2 = 6.
    args3 = types.SimpleNamespace(
        chat_template=template,
        max_steps=0,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        streaming=False,
    )
    trainer3 = _make_grpo_trainer_for_prepare_data(["a", "b", "c", "d"], args3)
    trainer3._prepare_data(is_vlm=False)
    assert trainer3._grpo_epoch_total_steps == 6


def test_grpo_prepare_data_leaves_epoch_plan_none_when_max_steps_set():
    # max_steps>0 keeps the existing behavior (num_train_epochs ignored, matching
    # HF), so no epoch plan is stashed and _train_inner uses max_steps directly.
    import types

    template = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
    args = types.SimpleNamespace(
        chat_template=template,
        max_steps=5,
        num_train_epochs=3,
        gradient_accumulation_steps=1,
        streaming=False,
    )
    trainer = _make_grpo_trainer_for_prepare_data(["a", "b", "c", "d"], args)
    trainer._prepare_data(is_vlm=False)
    assert trainer._grpo_epoch_total_steps is None


def test_train_inner_uses_grpo_epoch_plan_before_streaming_raise():
    # Structural guard: _train_inner must consult the stashed GRPO epoch plan in
    # its total_steps selection BEFORE the streaming "num_train_epochs requires a
    # finite dataset" raise, so the finite epoch-based GRPO path no longer errors.
    import inspect
    from unsloth_zoo.mlx.trainer import MLXTrainer

    src = inspect.getsource(MLXTrainer._train_inner)
    assert "_grpo_epoch_total_steps" in src
    plan_at = src.index("_grpo_epoch_total_steps")
    # main's #918 rework moved the streaming "requires a finite dataset" raise
    # into the module-level _resolve_training_steps fallback; _train_inner must
    # consult the GRPO epoch plan BEFORE falling through to that helper.
    fallback_at = src.index("_resolve_training_steps")
    assert plan_at < fallback_at


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
