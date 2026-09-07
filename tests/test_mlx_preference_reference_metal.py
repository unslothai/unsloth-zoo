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
            self.drop = nn.Dropout(0.0)
            self.head = nn.Linear(8, 16, bias=False)

        def __call__(self, tokens):
            return self.head(self.drop(self.proj(self.embed(tokens))))

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


@pytest.mark.parametrize("mode", ["adapter", "full", "ref_model"])
def test_reference_overrides_hold_inside_a_compiled_step(mode):
    """The reference stays at the initial weights while the policy trains; the
    unadapted twin from the same seed is the oracle."""
    import mlx.nn as nn
    import mlx.optimizers as optim
    from unsloth_zoo.mlx.preference import _response_logps

    model = _tiny() if mode == "full" else _adapted(train_embedding=True)
    pristine = _tiny()
    tokens, lengths = _batch()
    options = {}
    if mode == "ref_model":
        reference = _tiny()
        # Inert only because the build puts the reference in eval mode.
        reference.drop = nn.Dropout(0.5)
        options = {"ref_model": reference, "force_use_ref_model": True}
    policy = _build(model, **options)
    if mode == "ref_model":
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

    initial = model.embed.weight
    for index in range(3):
        loss, scored = step(tokens, lengths)
        mx.eval(loss, scored, model.parameters())
        if mode == "ref_model" and index == 0:
            scaled = pristine.embed.weight * 1.5
            reference.embed.weight = scaled
            pristine.embed.weight = scaled

    assert not mx.allclose(model.embed.weight, initial).item(), "the policy moved"
    if mode == "adapter":
        assert (model.proj.scale, model.head.scale) == (1.0, 1.0)
    if mode == "ref_model":
        assert reference.training is False
    expected = _response_logps(pristine, tokens, lengths)
    assert mx.allclose(scored, expected, atol=1e-5).item()
    assert loss.item() != 0.0, "the loss sees the gap"
