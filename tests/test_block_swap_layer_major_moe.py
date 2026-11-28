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

"""Layer-major gradient accumulation on mixture-of-experts decoders (CPU): with the router load-balancing loss on
(output_router_logits, router_aux_loss_coef > 0), a router z-loss, and router logits returned by the layer itself,
losses (aux included) and every trainable gradient (LoRA and the full-finetuned router) equal standard accumulation
bitwise under reentrant, non-reentrant and no gradient checkpointing; decoder buffers written in the forward are
refused."""

import functools
import importlib.util
import os
import re

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "block_swap_layer_major_moe_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap_layer_major.py"))
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)

torch.set_num_threads(min(4, torch.get_num_threads()))
_ROUTER = re.compile(r"\.(gate|router)\.(weight|bias)$")

MOE = {
    "qwen3_moe": ("Qwen3MoeConfig", "Qwen3MoeForCausalLM", dict(
        intermediate_size = 64, moe_intermediate_size = 16, head_dim = 8, num_experts = 4, num_experts_per_tok = 2,
        decoder_sparse_step = 1, mlp_only_layers = [])),
    "mixtral": ("MixtralConfig", "MixtralForCausalLM", dict(
        intermediate_size = 32, num_local_experts = 4, num_experts_per_tok = 2)),
    "gpt_oss": ("GptOssConfig", "GptOssForCausalLM", dict(
        intermediate_size = 32, head_dim = 8, num_local_experts = 4, num_experts_per_tok = 2,
        layer_types = ["sliding_attention", "full_attention", "sliding_attention"], sliding_window = 8)),
}


def _model(arch, gc, coef = 0.5, seed = 0):
    cfg_name, model_name, extra = MOE[arch]
    if not hasattr(transformers, model_name):
        pytest.skip(f"transformers {transformers.__version__} has no {model_name}")
    torch.manual_seed(seed)
    cfg = getattr(transformers, cfg_name)(hidden_size = 32, num_hidden_layers = 3, num_attention_heads = 4,
                                          num_key_value_heads = 2, vocab_size = 64, max_position_embeddings = 64,
                                          output_router_logits = True, router_aux_loss_coef = coef,
                                          tie_word_embeddings = False, **extra)
    cfg._attn_implementation = "eager"
    model = getattr(transformers, model_name)(cfg)
    model.config.use_cache = False
    if gc != "none":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": gc == "reentrant"})
    model = peft.get_peft_model(model, peft.LoraConfig(r = 4, lora_alpha = 8, init_lora_weights = False,
                                                       target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.enable_input_require_grads()
    routers = [p for n, p in model.named_parameters() if _ROUTER.search(n)]
    assert routers
    for p in routers:
        p.requires_grad_(True)     # full finetuning of the router weights
    model.train()
    return model


def _batches(n = 3, seed = 1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for m in range(n):
        rows, cols = 2, 6 + 3 * m
        ids = torch.randint(0, 64, (rows, cols), generator = g)
        mask = torch.ones(rows, cols, dtype = torch.long)
        mask[1, cols - 2 - m:] = 0
        out.append(dict(input_ids = ids, attention_mask = mask, labels = ids.masked_fill(mask == 0, -100)))
    items = sum(int((b["labels"][..., 1:] != -100).sum()) for b in out)
    return out, torch.tensor(items)


def _step(model, batch, items, zloss = False):
    out = model(**batch, num_items_in_batch = items)
    loss = out.loss
    if zloss:
        # A router z-loss on the recorded router logits (as switch-style routers add it).
        loss = loss + 1e-2 * sum(torch.logsumexp(r.float(), -1).square().mean() for r in out.router_logits)
    loss.backward()
    return loss.detach(), out.aux_loss.detach()


def _layers(model):
    return model.get_base_model().model.layers


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def _compare(model, zloss = False):
    batches, items = _batches()
    model.zero_grad(set_to_none = True)
    base = [_step(model, b, items, zloss) for b in batches]
    base_grads = _grads(model)
    model.zero_grad(set_to_none = True)
    lm.layer_major_stats(reset = True)
    fns = [functools.partial(_step, model, b, items, zloss) for b in batches]
    results, why = lm.run_layer_major(_layers(model), fns, model = model)
    assert why is None, why
    stats = lm.layer_major_stats()
    assert (stats["windows"], stats["aborts"]) == (1, 0), stats
    grads = _grads(model)
    for (loss, aux), (loss2, aux2) in zip(base, results):
        assert torch.equal(loss, loss2) and torch.equal(aux, aux2)
    assert base_grads.keys() == grads.keys() and any(_ROUTER.search(k) for k in grads)
    for k in base_grads:
        assert torch.equal(base_grads[k], grads[k]), k
    # No capture wrapper survives the window (transformers 4.x itself leaves the bound original behind).
    assert all(getattr(m.__dict__["forward"], "__self__", None) is m for m in model.modules() if "forward" in m.__dict__)
    return base_grads


@pytest.mark.parametrize("gc", ["reentrant", "non_reentrant", "none"])
@pytest.mark.parametrize("arch", sorted(MOE))
def test_moe_router_aux_loss_bitwise(arch, gc):
    _compare(_model(arch, gc))


def test_router_z_loss_bitwise():
    _compare(_model("qwen3_moe", "non_reentrant"), zloss = True)


def test_the_aux_loss_gradient_reaches_the_router():
    """Non-reentrant checkpointing differentiates the recorded router logits: the router gradient with the aux loss
    differs from the one without, and the window reproduces both."""
    with_aux = _compare(_model("qwen3_moe", "non_reentrant", coef = 0.5))
    without = _compare(_model("qwen3_moe", "non_reentrant", coef = 0.0))
    routers = [k for k in with_aux if _ROUTER.search(k)]
    assert any(not torch.equal(with_aux[k], without[k]) for k in routers)


class _RouterLayer(torch.nn.Module):
    """A decoder layer that returns its router logits after the hidden states (older / remote-code MoE layers)."""
    def __init__(self, w = 16, experts = 4, ckpt = None):
        super().__init__()
        self.lin = torch.nn.Linear(w, w)
        self.router = torch.nn.Linear(w, experts)
        self.gradient_checkpointing = ckpt is not None
        self._gradient_checkpointing_func = ckpt

    def _inner(self, x):
        logits = self.router(x)
        weight = torch.softmax(logits, -1).amax(-1, keepdim = True)
        return torch.tanh(self.lin(x)) * weight, logits

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(self._inner, x)
        return self._inner(x)


@pytest.mark.parametrize("ckpt", ["reentrant", "none"])
def test_router_logits_returned_by_the_layer_bitwise(ckpt):
    func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant = True) if ckpt == "reentrant" else None
    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_RouterLayer(ckpt = func) for _ in range(3)])

    def fn(m):
        x = torch.randn(2, 5, 16, generator = torch.Generator().manual_seed(m)).requires_grad_(True)
        aux = 0
        for layer in layers:
            x, logits = layer(x)
            aux = aux + torch.logsumexp(logits, -1).square().mean()
        loss = x.square().sum() + 0.1 * aux
        loss.backward()
        return loss.detach()

    for m in range(2):
        fn(m)
    base = {n: p.grad.clone() for n, p in layers.named_parameters()}
    for p in layers.parameters():
        p.grad = None
    lm.layer_major_stats(reset = True)
    results, why = lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], model = layers)
    assert why is None, why
    assert lm.layer_major_stats()["windows"] == 1
    for n, p in layers.named_parameters():
        assert torch.equal(base[n], p.grad), n


def test_a_buffer_written_in_the_forward_is_refused():
    """An expert bias updated inside the forward from the routed load (aux-free balancing done in the layer): the
    standard order updates it micro-batch after micro-batch, so the window refuses and leaves no gradient."""
    class Biased(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(16, 16)
            self.register_buffer("bias", torch.zeros(16))

        def forward(self, x):
            out = torch.tanh(self.lin(x) + self.bias)
            with torch.no_grad():
                self.bias.add_(0.1 * out.mean((0, 1)))
            return out

    torch.manual_seed(0)
    layers = torch.nn.ModuleList([Biased() for _ in range(3)])

    def fn(m):
        x = torch.randn(2, 5, 16, generator = torch.Generator().manual_seed(m)).requires_grad_(True)
        for layer in layers:
            x = layer(x)
        x.square().sum().backward()

    before = [layer.bias.clone() for layer in layers]
    results, why = lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], model = layers)
    assert results is None and "buffers" in why, why
    assert all(p.grad is None for p in layers.parameters())
    # The standard retry starts from the values the window found, not from its micro-batches' updates.
    assert all(torch.equal(layer.bias, b) for layer, b in zip(layers, before))


def test_capture_hooks_installed_by_the_first_call_inside_a_window():
    """transformers 5.x installs its output-capture hooks on the first call that records: here that call is a
    window's first micro-batch (no standard step before it on this model). Bitwise to a twin model trained the
    standard way."""
    batches, items = _batches()
    twin = _model("qwen3_moe", "non_reentrant")
    for b in batches:
        _step(twin, b, items)
    base = _grads(twin)
    model = _model("qwen3_moe", "non_reentrant")
    lm.layer_major_stats(reset = True)
    fns = [functools.partial(_step, model, b, items) for b in batches]
    results, why = lm.run_layer_major(_layers(model), fns, model = model)
    assert why is None, why
    grads = _grads(model)
    assert base.keys() == grads.keys()
    for k in base:
        assert torch.equal(base[k], grads[k]), k


def _unsloth_non_reentrant_checkpoint():
    """A stand-in with the identity of Unsloth's process-wide non-reentrant checkpoint (unsloth's
    non_reentrant_checkpoint(), which the trainer installs when the MoE router loss is on)."""
    original = torch.utils.checkpoint.checkpoint

    def non_reentrant_checkpoint(original):
        def _nonre_checkpoint(function, *args, **kwargs):
            kwargs["use_reentrant"] = False
            return original(function, *args, **kwargs)
        _nonre_checkpoint._unsloth_original = original
        _nonre_checkpoint._unsloth_offloads = False
        return _nonre_checkpoint
    fn = non_reentrant_checkpoint(original)
    fn.__module__ = "unsloth.models.non_reentrant_checkpoint"
    fn.__qualname__ = "non_reentrant_checkpoint.<locals>._nonre_checkpoint"
    return fn


def test_unsloth_non_reentrant_checkpoint_engages_bitwise():
    model = _model("qwen3_moe", "reentrant")
    fn = _unsloth_non_reentrant_checkpoint()
    for layer in _layers(model):
        layer._gradient_checkpointing_func = fn
    assert lm._checkpoint_mode(_layers(model)[0]) == (False, None)
    with_aux = _compare(model)
    # The router gets the load-balancing loss's gradient there (non-reentrant), and the window reproduces it.
    assert any(_ROUTER.search(k) and with_aux[k].abs().sum() > 0 for k in with_aux)


def test_forward_swapping_capture_in_compiled_layers_bitwise():
    """transformers 4.x records router logits by assigning capture wrappers to the recorded modules' instance forward
    per model call: the window swaps those per turn, so compiled decoder code sees the same instance-forward protocol
    as in standard training. Once standard steps compiled the graphs, a window runs bitwise and compiles nothing."""
    if not lm._forward_swapping_capture():
        pytest.skip("transformers 5.x records router logits per thread through hooks")
    model = _model("qwen3_moe", "non_reentrant")
    for layer in _layers(model):
        layer.compile(backend = "aot_eager")
    batches, items = _batches()
    for _ in range(2):   # standard steps compile every graph the window will need
        model.zero_grad(set_to_none = True)
        [_step(model, b, items) for b in batches]
    _compare(model)
    assert all(getattr(m.__dict__["forward"], "__self__", None) is m for m in model.modules() if "forward" in m.__dict__)


@pytest.mark.parametrize("gc", ["non_reentrant", "none"])
def test_compiled_layers_with_a_grad_mode_standard_forward_bitwise(gc):
    """Compiled decoder layers whose standard forward runs under grad mode: the window runs that same call (through
    the layer's own non-reentrant checkpoint), so the compiled training graph is the one standard training runs."""
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin, self.router = torch.nn.Linear(16, 16), torch.nn.Linear(16, 4)
            self.gradient_checkpointing = gc != "none"
            self._gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint,
                                                                  use_reentrant = False)

        def _inner(self, x):
            weight = torch.softmax(self.router(x), -1).amax(-1, keepdim = True)
            return torch.tanh(self.lin(x)) * weight

        def forward(self, x):
            if self.gradient_checkpointing and self.training:
                return self._gradient_checkpointing_func(self._inner, x)
            return self._inner(x)

    torch.manual_seed(0)
    layers = torch.nn.ModuleList([Layer() for _ in range(3)])
    for layer in layers:
        layer.compile(backend = "aot_eager")
    assert lm._compiled_layers(layers)

    def fn(m):
        x = torch.randn(2, 5, 16, generator = torch.Generator().manual_seed(m)).requires_grad_(True)
        for layer in layers:
            x = layer(x)
        loss = x.square().sum()
        loss.backward()
        return loss.detach()

    base_losses = [fn(m) for m in range(2)]
    base = {n: p.grad.clone() for n, p in layers.named_parameters()}
    for p in layers.parameters():
        p.grad = None
    lm.layer_major_stats(reset = True)
    results, why = lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], model = layers)
    assert why is None, why
    assert lm.layer_major_stats()["windows"] == 1
    assert all(torch.equal(a, b) for a, b in zip(base_losses, results))
    for n, p in layers.named_parameters():
        assert torch.equal(base[n], p.grad), n


def test_per_call_router_recorder_hooks_bitwise():
    """A model forward that records router logits with forward hooks it registers per call and removes after
    (Unsloth's gpt-oss forward), each hook recording only its own thread's calls: every micro-batch gets its own
    router logits, and the load-balancing style loss on them is bitwise to standard accumulation."""
    import threading
    torch.manual_seed(0)
    layers = torch.nn.ModuleList()
    for _ in range(3):
        layer = torch.nn.Module()
        layer.lin, layer.router = torch.nn.Linear(16, 16), torch.nn.Linear(16, 4)
        layers.append(layer)

    class Layer(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            weight = torch.softmax(self.inner.router(x), -1).amax(-1, keepdim = True)
            return torch.tanh(self.inner.lin(x)) * weight
    stack = torch.nn.ModuleList([Layer(l) for l in layers])

    def fn(m):
        sink, me = [], threading.get_ident()

        def record(module, args, output):
            if threading.get_ident() == me:
                sink.append(output)
        record._unsloth_router_recorder = (me, sink)
        handles = [layer.inner.router.register_forward_hook(record) for layer in stack]
        try:
            x = torch.randn(2, 5, 16, generator = torch.Generator().manual_seed(m)).requires_grad_(True)
            for layer in stack:
                x = layer(x)
        finally:
            for h in handles:
                h.remove()
        assert len(sink) == 3
        loss = x.square().sum() + 0.1 * sum(torch.logsumexp(r, -1).square().mean() for r in sink)
        loss.backward()
        return loss.detach()

    base_losses = [fn(m) for m in range(2)]
    base = {n: p.grad.clone() for n, p in stack.named_parameters()}
    for p in stack.parameters():
        p.grad = None
    lm.layer_major_stats(reset = True)
    results, why = lm.run_layer_major(stack, [functools.partial(fn, m) for m in range(2)], model = stack)
    assert why is None, why
    assert lm.layer_major_stats()["windows"] == 1
    assert all(torch.equal(a, b) for a, b in zip(base_losses, results))
    for n, p in stack.named_parameters():
        assert torch.equal(base[n], p.grad), n


def test_a_router_output_both_recorded_and_returned_is_backpropagated_once():
    """The same router logits recorded by a per-call hook and returned by the layer are one leaf: its gradient (both
    uses summed) goes through the recompute once, so the router gradient equals standard accumulation's."""
    import threading
    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_RouterLayer() for _ in range(3)])

    def fn(m):
        sink, me = [], threading.get_ident()

        def record(module, args, output):
            if threading.get_ident() == me:
                sink.append(output)
        record._unsloth_router_recorder = (me, sink)
        handles = [layer.router.register_forward_hook(record) for layer in layers]
        try:
            x = torch.randn(2, 5, 16, generator = torch.Generator().manual_seed(m)).requires_grad_(True)
            returned = []
            for layer in layers:
                x, logits = layer(x)
                returned.append(logits)
        finally:
            for h in handles:
                h.remove()
        loss = x.square().sum() + 0.1 * sum(r.square().mean() for r in sink) + 0.1 * sum(r.sum() for r in returned)
        loss.backward()
        return loss.detach()

    base_losses = [fn(m) for m in range(2)]
    base = {n: p.grad.clone() for n, p in layers.named_parameters()}
    for p in layers.parameters():
        p.grad = None
    results, why = lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], model = layers)
    assert why is None, why
    assert all(torch.equal(a, b) for a, b in zip(base_losses, results))
    for n, p in layers.named_parameters():
        assert torch.equal(base[n], p.grad), n
