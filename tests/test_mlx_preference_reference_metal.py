"""Real-runtime contracts of the MLX DPO reference policy."""

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
            self.head = nn.Linear(8, 16, bias=False)

        def __call__(self, tokens):
            return self.head(self.proj(self.embed(tokens)))

    mx.random.seed(0)
    model = Tiny()
    mx.eval(model.parameters())
    return model


def _adapted(*, dora=False, train_embedding=False):
    """Two adapted linears and, for DoRA, an adapted embedding."""
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


def test_reference_overrides_hold_inside_a_compiled_step():
    """The reference stays at the initial embedding while the policy trains."""
    import mlx.nn as nn
    import mlx.optimizers as optim
    from unsloth_zoo.mlx.preference import _response_logps

    model = _adapted(train_embedding=True)
    pristine = _adapted(train_embedding=True)
    pristine.update(model.parameters())
    tokens, lengths = _batch()
    policy = _build(model)
    optimizer = optim.SGD(learning_rate=0.5)

    def loss_fn(model, tokens, lengths):
        reference = policy.forward(model, tokens, lengths)
        return (_response_logps(model, tokens, lengths) - reference).sum(), reference

    value_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, policy.state]

    @functools.partial(mx.compile, inputs=state, outputs=state)
    def step(tokens, lengths):
        (loss, reference), grads = value_and_grad(model, tokens, lengths)
        optimizer.update(model, grads)
        return loss, reference

    for _ in range(3):
        loss, reference = step(tokens, lengths)
        mx.eval(loss, reference, model.parameters())

    assert not mx.allclose(model.embed.weight, pristine.embed.weight).item()
    assert (model.proj.scale, model.head.scale) == (1.0, 1.0)
    # pristine still holds the initial weights and a zero-delta adapter.
    expected = _response_logps(pristine, tokens, lengths)
    assert mx.allclose(reference, expected, atol=1e-5).item()
    assert loss.item() != 0.0, "the policy moved away from the reference"
