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

"""Real-runtime contracts of MLX preference training."""

import functools

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

pytestmark = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")


def _tiny():
    import mlx.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.proj = nn.Linear(8, 8, bias=False)
            self.drop = nn.Dropout(0.0)
            self.head = nn.Linear(8, 16, bias=False)

        def __call__(self, tokens):
            return self.head(self.drop(self.proj(self.embed(tokens))))

    mx.random.seed(0)
    model = Tiny()
    mx.eval(model.parameters())
    return model


def _adapted(*, dora=False, train_embedding=False):
    from mlx_lm.tuner.dora import DoRAEmbedding, DoRALinear
    from mlx_lm.tuner.lora import LoRALinear

    model = _tiny()
    keys = ["lora_a", "lora_b"] + (["m"] if dora else [])
    for name in ("proj", "head"):
        wrapped = (DoRALinear if dora else LoRALinear).from_base(
            getattr(model, name), r=2, scale=1.0,
        )
        setattr(model, name, wrapped)
    if dora:
        model.embed = DoRAEmbedding.from_base(model.embed, r=2, scale=1.0)
    model.freeze()
    for name in ("proj", "head") + (("embed",) if dora else ()):
        getattr(model, name).unfreeze(keys=keys)
    if train_embedding:
        model.embed.unfreeze()
    mx.eval(model.parameters())
    return model


def _batch():
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 0], [1, 2, 9, 10, 0, 0],
                       [1, 2, 11, 12, 13, 0]], dtype=mx.int32)
    return tokens, mx.array([[3, 6], [3, 5], [2, 4], [2, 5]], dtype=mx.int32)


def _build(model, **options):
    from unsloth_zoo.mlx.preference import build_reference_policy
    return build_reference_policy(
        model, reference_free=False, resume_provenance=None, **options,
    )[0]


@pytest.mark.parametrize("mode", [
    "adapter", "full", "ref_model", "synced_full", "synced_model",
    "synced_adapter", "synced_dora", "synced_adapted_model",
])
def test_reference_overrides_hold_inside_a_compiled_step(mode):
    """The compiled step scores against the initial weights or the synced mix."""
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    from unsloth_zoo.mlx.preference import _response_logps

    synced = mode.startswith("synced")
    with_model = mode in ("ref_model", "synced_model", "synced_adapted_model")
    adapted = mode in ("adapter", "ref_model", "synced_adapter", "synced_dora",
                       "synced_adapted_model")
    build = functools.partial(
        _adapted, dora=mode == "synced_dora", train_embedding=True,
    ) if adapted else _tiny
    model = build()
    pristine = build() if synced and adapted else _tiny()
    tokens, lengths = _batch()
    options = {"sync_ref_model": synced}
    if with_model:
        reference = build() if mode == "synced_adapted_model" else _tiny()
        # Inert only because the build puts the reference in eval mode.
        reference.drop = nn.Dropout(0.5)
        options.update(ref_model=reference, force_use_ref_model=True)
    policy = _build(model, **options)
    if with_model:
        assert policy.state is reference.state
    optimizer = optim.SGD(learning_rate=0.5)

    def loss_fn(model, tokens, lengths):
        scored = policy.forward(model, tokens, lengths)
        return (_response_logps(model, tokens, lengths) - scored).sum(), scored

    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, policy.state]

    @functools.partial(mx.compile, inputs=state, outputs=state)
    def step(tokens, lengths):
        (loss, scored), grads = value_and_grad(model, tokens, lengths)
        optimizer.update(model, grads)
        return loss, scored

    embedding = model.embed.embedding if mode == "synced_dora" else model.embed
    initial = embedding.weight
    for index in range(3):
        loss, scored = step(tokens, lengths)
        mx.eval(loss, scored, model.parameters())
        if with_model and index == 0:
            scaled = pristine.embed.weight * 1.5
            reference.embed.weight = scaled
            pristine.embed.weight = scaled
        if synced and index == 1:
            policy.sync(model, 0.25)
            pristine.update(tree_map(
                lambda held, trained: 0.75 * held + 0.25 * trained,
                pristine.trainable_parameters(), model.trainable_parameters(),
            ))

    assert not mx.allclose(embedding.weight, initial).item(), "the policy moved"
    if mode == "adapter":
        assert (model.proj.scale, model.head.scale) == (1.0, 1.0)
    if with_model:
        assert reference.training is False
    expected = _response_logps(pristine, tokens, lengths)
    assert mx.allclose(scored, expected, atol=1e-5).item()
    assert loss.item() != 0.0, "the loss sees the gap"


def test_a_qualified_vision_plan_is_surveyed_and_planned_for_compile():
    from types import SimpleNamespace
    from unsloth_zoo.mlx.compile import build_compile_policy
    from unsloth_zoo.mlx.preference import FiniteVisionPreferenceBatchPlan
    from unsloth_zoo.mlx.trainer import MLXDPOConfig, _plan_single_process_text_shapes

    class Plan(FiniteVisionPreferenceBatchPlan):
        pixel_width = 33
        def _inputs(self, rows, batch, lengths):
            inputs = {"input_ids": mx.array(batch), "attention_mask": mx.array(
                (batch != self.pad_id).astype("int32"))}
            if rows[0].image:
                inputs["pixel_values"] = mx.zeros((len(batch), self.pixel_width))
            return inputs

    row = lambda length, image: SimpleNamespace(
        chosen=(1,) * length, rejected=(1,) * (length - 2), image=image,
        chosen_prompt_ids=(1, 1), rejected_prompt_ids=(1, 1))

    plan = Plan(
        [row(20, False), row(50, True)], [[0], [1]], processor=None, model_config={},
        normalizers=[(1, 1, 1)] * 2, cycle_length=2, max_seq_length=64, pad_id=0)
    args = MLXDPOConfig(max_steps=2, gradient_accumulation_steps=1, max_grad_norm=0.0)

    def planned(decision):
        return _plan_single_process_text_shapes(
            plan, None, args=args, total_steps=2, is_vlm=True, distributed_world_size=1,
            compile_policy=build_compile_policy(args=args), vlm_compile_decision=decision)

    assert planned(None)[1].reason == "vlm_compile_unqualified" and plan.batch_width(0) == 33
    assert planned(SimpleNamespace(enabled=True))[1:3] == (plan._shape_plan.report, True)
    # The text batch steps off the width the image batch's pixels already use.
    assert [plan.batch_width(index) for index in (0, 1)] == [34, 64]
    assert plan.batch_family(0)[4] != plan.batch_family(1)[4]
    text, image = (plan.materialize(index, phase="single")[0] for index in (0, 1))
    assert text["input_ids"].shape == (2, 34) and "pixel_values" not in text
    assert image["input_ids"].shape == (2, 64) and image["pixel_values"].shape == (2, 33)
    assert plan[0][0]["input_ids"].shape == (2, 33)
    with pytest.raises(RuntimeError, match="not admitted"):
        plan.materialize(1, phase="update")
    Plan.pixel_width = 16
    with pytest.raises(RuntimeError, match="drifted from its surveyed compile family"):
        plan.materialize(1, phase="single")
