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

"""Layer-major gradient accumulation (block_swap_layer_major): losses and LoRA / embedding / head gradients equal to
standard accumulation bitwise on a tiny Llama with LoRA (CPU), layer calls / recomputes counted, every protocol break
aborts before a decoder-layer gradient exists and puts the other gradients back. GPU tests run a real BlockSwap
(RAM and disk tiers): bitwise against standard accumulation and copies / reads per step divided by the micro-batch count."""

import contextlib
import functools
import importlib.util
import os
import sys
import threading
import warnings

import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "block_swap_layer_major_under_test", os.path.join(_HERE, "unsloth_zoo", "block_swap_layer_major.py"))
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)

torch.set_num_threads(min(4, torch.get_num_threads()))
_cuda = pytest.mark.skipif(not torch.cuda.is_available() or not sys.platform.startswith("linux"),
                           reason = "needs a CUDA / ROCm GPU on Linux")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# Standard dense decoders: (config class, model class, extra config) by family, as transformers names them.
FAMILIES = {
    "llama": ("LlamaConfig", "LlamaForCausalLM", {"tie_word_embeddings": False}),
    "qwen2": ("Qwen2Config", "Qwen2ForCausalLM", {"tie_word_embeddings": False}),
    "qwen3": ("Qwen3Config", "Qwen3ForCausalLM", {"head_dim": 16, "tie_word_embeddings": False}),
    "mistral": ("MistralConfig", "MistralForCausalLM", {"tie_word_embeddings": False}),
    # Gemma: tied embedding / head (frozen under LoRA), soft-capping, sliding-window layers shorter than the batches.
    "gemma2": ("Gemma2Config", "Gemma2ForCausalLM", {"head_dim": 16, "sliding_window": 8}),
    "gemma3": ("Gemma3TextConfig", "Gemma3ForCausalLM", {"head_dim": 16, "sliding_window": 8}),
}


def _model(reentrant = True, layers = 4, dtype = torch.float32, device = "cpu", train_embed = False,
           train_head = False, seed = 0, arch = "llama"):
    torch.manual_seed(seed)
    cfg_name, model_name, extra = FAMILIES[arch]
    if not hasattr(transformers, model_name):
        pytest.skip(f"transformers {transformers.__version__} has no {model_name}")
    cfg = getattr(transformers, cfg_name)(hidden_size = 64, intermediate_size = 128, num_hidden_layers = layers,
                                          num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 256,
                                          max_position_embeddings = 256, **extra)
    cfg._attn_implementation = "eager"
    model = getattr(transformers, model_name)(cfg).to(dtype = dtype, device = device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": reentrant})
    model = peft.get_peft_model(model, peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = TARGETS,
                                                       lora_dropout = 0.0, init_lora_weights = False))
    model.enable_input_require_grads()
    base = model.get_base_model()
    if train_embed:
        base.model.embed_tokens.weight.requires_grad_(True)
    if train_head:
        base.lm_head.weight.requires_grad_(True)
    model.train()
    return model


def _batches(n = 3, device = "cpu", seed = 1):
    """Micro-batches of different lengths, right padded, with masked labels (as a padding collator makes)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for m in range(n):
        rows, cols = 2, 12 + 5 * m
        ids = torch.randint(0, 256, (rows, cols), generator = g)
        mask = torch.ones(rows, cols, dtype = torch.long)
        mask[1, cols - 3 - m:] = 0
        labels = ids.masked_fill(mask == 0, -100)
        out.append({k: v.to(device) for k, v in dict(input_ids = ids, attention_mask = mask, labels = labels).items()})
    items = sum(int((b["labels"][..., 1:] != -100).sum()) for b in out)
    return out, torch.tensor(items, device = device)


def _step(model, batch, items):
    out = model(**batch, num_items_in_batch = items)
    out.loss.backward()
    return out.loss.detach()


def _layers(model):
    return model.get_base_model().model.layers


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def _assert_same(a, b):
    assert a.keys() == b.keys() and a
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _standard(model, batches, items):
    model.zero_grad(set_to_none = True)
    losses = [_step(model, b, items) for b in batches]
    return losses, _grads(model)


def _layer_major(model, batches, items, swapper = None):
    model.zero_grad(set_to_none = True)
    fns = [lambda b = b: _step(model, b, items) for b in batches]
    losses, why = lm.run_layer_major(_layers(model), fns, swapper = swapper, model = model)
    assert why is None, why
    return losses, _grads(model)


class _Holds:
    """A stand-in swapper recording hold / unhold, as BlockSwap exposes them."""
    def __init__(self):
        self.log = []

    def hold(self, idx):
        self.log.append(("hold", idx))

    def unhold(self, idx):
        self.log.append(("unhold", idx))


@pytest.mark.parametrize("reentrant", [True, False])
@pytest.mark.parametrize("extra", ["lora", "embed", "head"])
def test_bitwise_equal_to_standard_accumulation(reentrant, extra):
    model = _model(reentrant, train_embed = extra == "embed", train_head = extra == "head")
    batches, items = _batches(3)
    base_losses, base_grads = _standard(model, batches, items)
    lm.layer_major_stats(reset = True)
    losses, grads = _layer_major(model, batches, items)
    for a, b in zip(base_losses, losses):
        assert torch.equal(a, b)
    _assert_same(base_grads, grads)
    if extra == "embed":
        assert any("embed_tokens" in k for k in grads)
    if extra == "head":
        assert any("lm_head" in k for k in grads)
    s = lm.layer_major_stats()
    assert (s["windows"], s["micro_batches"], s["layer_calls"], s["recomputes"], s["aborts"]) == (1, 3, 12, 12, 0)


def test_mean_loss_without_num_items_and_two_steps():
    """Per micro-batch mean loss / GA (no num_items_in_batch), and a second window on the same model."""
    model = _model()
    batches, _ = _batches(4, seed = 3)

    def step(b):
        out = model(**b)
        (out.loss / 4).backward()
        return out.loss.detach()

    for _ in range(2):
        model.zero_grad(set_to_none = True)
        base = [step(b) for b in batches]
        base_grads = _grads(model)
        model.zero_grad(set_to_none = True)
        losses, why = lm.run_layer_major(_layers(model), [lambda b = b: step(b) for b in batches], model = model)
        assert why is None
        assert all(torch.equal(a, b) for a, b in zip(base, losses))
        _assert_same(base_grads, _grads(model))
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(p.grad, alpha = -1e-2)


def test_holds_each_layer_once_per_pass():
    model = _model()
    batches, items = _batches(3)
    holds = _Holds()
    _layer_major(model, batches, items, swapper = holds)
    L = len(_layers(model))
    forward = [x for l in range(L) for x in (("hold", l), ("unhold", l))]
    backward = [x for l in reversed(range(L)) for x in (("hold", l), ("unhold", l))]
    assert holds.log == forward + backward


def test_class_call_restored_and_layers_untouched_outside_a_window():
    model = _model()
    cls = type(_layers(model)[0])
    before = cls.__dict__.get("__call__")
    batches, items = _batches(2)
    _layer_major(model, batches, items)
    assert cls.__dict__.get("__call__") is before
    lm.layer_major_stats(reset = True)
    model.zero_grad(set_to_none = True)
    _step(model, batches[0], items)
    assert lm.layer_major_stats()["layer_calls"] == 0


def _abort_case(model, batches, items, fns = None):
    model.zero_grad(set_to_none = True)
    lm.layer_major_stats(reset = True)
    fns = fns or [lambda b = b: _step(model, b, items) for b in batches]
    losses, why = lm.run_layer_major(_layers(model), fns, model = model)
    assert losses is None and why
    assert lm.layer_major_stats()["aborts"] == 1
    return why


def _manual_forward(model, b, between = None):
    """The decoder loop by hand, so a test can change the hidden states between two layers."""
    decoder = model.get_base_model().model
    x = decoder.embed_tokens(b["input_ids"])
    pe = decoder.rotary_emb(x, torch.arange(x.shape[1])[None])
    for k, layer in enumerate(_layers(model)):
        x = layer(x, position_embeddings = pe)
        x = x[0] if isinstance(x, tuple) else x
        if between is not None and k == 0:
            x = between(x)
    return x.float().pow(2).mean()


def test_abort_when_hidden_states_change_between_layers():
    model = _model()
    batches, items = _batches(2)

    def fn(b):
        loss = _manual_forward(model, b, between = lambda x: x * 1.0)
        loss.backward()
        return loss.detach()

    why = _abort_case(model, batches, items, [lambda b = b: fn(b) for b in batches])
    assert "changed between" in why
    assert _grads(model) == {}
    # The same micro-batches then run the standard way.
    _standard(model, batches, items)


def test_manual_loop_matches_standard():
    """A hand-written decoder loop (no transformers model forward) through the window, against running it plainly."""
    model = _model()
    batches, _ = _batches(3)

    def fn(b):
        loss = _manual_forward(model, b)
        loss.backward()
        return loss.detach()

    model.zero_grad(set_to_none = True)
    base = [fn(b) for b in batches]
    base_grads = _grads(model)
    model.zero_grad(set_to_none = True)
    losses, why = lm.run_layer_major(_layers(model), [lambda b = b: fn(b) for b in batches], model = model)
    assert why is None
    assert all(torch.equal(a, b) for a, b in zip(base, losses))
    _assert_same(base_grads, _grads(model))


def test_abort_in_the_loss_part_restores_outside_gradients():
    """A second forward after the last layer (two model calls per micro-batch) aborts in the loss part: the head's
    gradient written by the first micro-batch is put back, no decoder-layer gradient exists."""
    model = _model(train_head = True)
    batches, items = _batches(2)

    def twice(b):
        loss = _step(model, b, items)
        model(**b)
        return loss

    why = _abort_case(model, batches, items, [lambda b = b: twice(b) for b in batches])
    assert "called decoder layer 0" in why
    assert _grads(model) == {}


def test_abort_on_an_input_that_needs_grad_besides_hidden_states():
    """A layer input with a graph of its own (position embeddings that need gradients) would lose it."""
    model = _model()
    batches, items = _batches(2)
    decoder = model.get_base_model().model

    def fn(b):
        x = decoder.embed_tokens(b["input_ids"])
        cos, sin = decoder.rotary_emb(x, torch.arange(x.shape[1])[None])
        cos = cos.clone().requires_grad_(True)
        for layer in _layers(model):
            x = layer(x, position_embeddings = (cos, sin))
        return x

    why = _abort_case(model, batches, items, [lambda b = b: fn(b) for b in batches])
    assert "other than the hidden states" in why


def test_abort_on_a_checkpoint_around_the_layer_call():
    """A reentrant checkpoint the model applies around the layer call runs it with gradients off."""
    from torch.utils.checkpoint import checkpoint
    model = _model()
    batches, items = _batches(2)
    decoder = model.get_base_model().model
    layer = _layers(model)[0]
    layer.gradient_checkpointing = False

    def fn(b):
        x = decoder.embed_tokens(b["input_ids"])
        pe = decoder.rotary_emb(x, torch.arange(x.shape[1])[None])
        return checkpoint(lambda t: layer(t, position_embeddings = pe), x, use_reentrant = True)

    why = _abort_case(model, batches, items, [lambda b = b: fn(b) for b in batches])
    assert "gradients off" in why


def test_real_errors_are_raised_and_classes_restored():
    model = _model()
    batches, items = _batches(2)
    cls = type(_layers(model)[0])
    before = cls.__dict__.get("__call__")

    def boom(b):
        model(**b)
        raise ValueError("boom")

    with pytest.raises(ValueError, match = "boom"):
        lm.run_layer_major(_layers(model), [lambda b = b: boom(b) for b in batches], model = model)
    assert cls.__dict__.get("__call__") is before


def test_dropout_masks_replayed_in_the_recompute():
    """With one micro-batch the window draws dropout masks in the standard order, and the recompute replays them:
    bitwise equal to standard checkpointed training with lora_dropout."""
    for reentrant in (True, False):
        model = _model(reentrant)
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.5
        batches, items = _batches(1)
        torch.manual_seed(7)
        base_losses, base_grads = _standard(model, batches, items)
        torch.manual_seed(7)
        losses, grads = _layer_major(model, batches, items)
        assert torch.equal(base_losses[0], losses[0])
        _assert_same(base_grads, grads)


@_cuda
def test_gpu_dropout_replayed_on_the_layers_device():
    """A model on a CUDA device other than the current one: the recompute replays that device's generator."""
    n_dev = torch.cuda.device_count()
    dev = torch.device("cuda", n_dev - 1)
    model = _model(device = dev)
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.5
    batches, items = _batches(1, device = dev)
    with torch.cuda.device(0):
        torch.manual_seed(7)
        base_losses, base_grads = _standard(model, batches, items)
        torch.manual_seed(7)
        losses, grads = _layer_major(model, batches, items)
    assert torch.equal(base_losses[0], losses[0])
    _assert_same(base_grads, grads)


def test_loss_on_intermediate_hidden_states():
    """A loss term on an intermediate layer's output, collected from the decoder loop as Unsloth's fast forward
    collects hidden states, reaches the layers below it."""
    model = _model()
    batches, _ = _batches(3)

    def step(b):
        decoder = model.get_base_model().model
        x = decoder.embed_tokens(b["input_ids"])
        pe = decoder.rotary_emb(x, torch.arange(x.shape[1])[None])
        mid = None
        for k, layer in enumerate(_layers(model)):
            x = layer(x, position_embeddings = pe)
            x = x[0] if isinstance(x, tuple) else x
            mid = x if k == 1 else mid
        loss = x.float().pow(2).mean() + 1e-1 * mid.float().square().mean()
        loss.backward()
        return loss.detach()

    model.zero_grad(set_to_none = True)
    base = [step(b) for b in batches]
    base_grads = _grads(model)
    model.zero_grad(set_to_none = True)
    losses, why = lm.run_layer_major(_layers(model), [lambda b = b: step(b) for b in batches], model = model)
    assert why is None
    assert all(torch.equal(a, b) for a, b in zip(base, losses))
    _assert_same(base_grads, _grads(model))


def test_output_recorder_hooks_abort_or_match():
    """transformers' output_hidden_states records layer outputs with per-call hooks (they would collect every
    interleaved micro-batch): the window aborts. Versions that return hidden states from the loop match instead."""
    model = _model()
    batches, items = _batches(2)

    def step(b):
        out = model(**b, num_items_in_batch = items, output_hidden_states = True)
        loss = out.loss + 1e-3 * out.hidden_states[2].float().square().mean()
        loss.backward()
        return loss.detach()

    model.zero_grad(set_to_none = True)
    base = [step(b) for b in batches]
    base_grads = _grads(model)
    model.zero_grad(set_to_none = True)
    losses, why = lm.run_layer_major(_layers(model), [lambda b = b: step(b) for b in batches], model = model)
    if losses is None:
        assert "output recorders" in why, why
        assert _grads(model) == {}
    else:
        assert all(torch.equal(a, b) for a, b in zip(base, losses))
        _assert_same(base_grads, _grads(model))


def test_abort_on_a_loss_on_the_first_layer_input():
    """A trainable embedding whose output also feeds a loss term: the loss part consumes the input-side graph
    before the window's backward, so the window aborts and puts the embedding gradient back."""
    model = _model(train_embed = True)
    batches, items = _batches(2)
    decoder = model.get_base_model().model

    def fn(b):
        x0 = decoder.embed_tokens(b["input_ids"])
        pe = decoder.rotary_emb(x0, torch.arange(x0.shape[1])[None])
        x = x0
        for layer in _layers(model):
            x = layer(x, position_embeddings = pe)
            x = x[0] if isinstance(x, tuple) else x
        loss = x.float().pow(2).mean() + x0.float().square().mean()
        loss.backward()
        return loss.detach()

    why = _abort_case(model, batches, items, [lambda b = b: fn(b) for b in batches])
    assert "first decoder layer's input" in why
    assert _grads(model) == {}


def test_abort_restores_decoder_gradients_written_by_the_loss_part():
    """A regulariser on a decoder weight writes its gradient in the loss part; a later abort puts it back too."""
    model = _model()
    batches, items = _batches(2)
    weight = next(p for n, p in _layers(model)[1].named_parameters() if p.requires_grad)

    def fn(b):
        out = model(**b, num_items_in_batch = items)
        (out.loss + weight.square().sum()).backward()
        model(**b)   # a second forward: protocol break in the loss part
        return out.loss.detach()

    why = _abort_case(model, batches, items, [lambda b = b: fn(b) for b in batches])
    assert "called decoder layer 0" in why
    assert _grads(model) == {}


def test_abort_on_a_live_kv_cache():
    """use_cache without checkpointing hands the layers a cache the recompute would update twice."""
    model = _model()
    model.get_base_model().gradient_checkpointing_disable()
    model.config.use_cache = True
    batches, items = _batches(2)

    def fn(b):
        out = model(**b, num_items_in_batch = items, use_cache = True)
        out.loss.backward()
        return out.loss.detach()

    why = _abort_case(model, batches, items, [lambda b = b: fn(b) for b in batches])
    assert "live KV cache" in why
    assert _grads(model) == {}


def test_out_of_memory_in_the_decoder_backward_is_an_abort():
    """The recompute runs out of memory after the loss parts wrote the head gradient: every gradient put back."""
    model = _model(train_head = True)
    batches, items = _batches(2)
    layer = _layers(model)[1]

    def oom(mod, args, out):
        if torch.is_grad_enabled():
            raise torch.OutOfMemoryError("simulated")

    h = layer.register_forward_hook(oom)
    try:
        why = _abort_case(model, batches, items)
    finally:
        h.remove()
    assert "out of memory" in why
    assert _grads(model) == {}


def test_out_of_memory_in_the_forward_frees_its_tensors():
    """An out-of-memory error in a worker's forward is an abort, and the tensors its frames held are freed when
    the call returns (no reference cycle through the stored traceback)."""
    import weakref
    model = _model()
    batches, items = _batches(2)
    refs = []

    def oom(mod, args, out):
        if not torch.is_grad_enabled():
            big = torch.empty(1 << 16)
            refs.append(weakref.ref(big))
            raise torch.OutOfMemoryError("simulated")

    h = _layers(model)[2].register_forward_hook(oom)
    try:
        why = _abort_case(model, batches, items)
    finally:
        h.remove()
    assert "OutOfMemoryError" in why
    assert refs and all(r() is None for r in refs)
    assert _grads(model) == {}


def test_distributed_is_refused(monkeypatch):
    model = _model()
    batches, items = _batches(2)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 2)
    losses, why = lm.run_layer_major(_layers(model), [lambda b = b: _step(model, b, items) for b in batches],
                                     model = model)
    assert losses is None and "distributed" in why
    assert _grads(model) == {}


def test_caller_autocast_reaches_the_micro_batches():
    # Non-reentrant: older torch's reentrant checkpoint under CPU autocast leaves the LoRA weights without
    # gradients even in standard accumulation.
    model = _model(reentrant = False)
    batches, items = _batches(2)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        base_losses, base_grads = _standard(model, batches, items)
        losses, grads = _layer_major(model, batches, items)
    assert all(torch.equal(a, b) for a, b in zip(base_losses, losses))
    _assert_same(base_grads, grads)


def test_abort_on_in_place_writes_between_layers():
    model = _model()
    batches, items = _batches(2)

    def under_grad(b):
        return _manual_forward(model, b, between = lambda x: x.mul_(1.0))

    def under_no_grad(b):
        def write(x):
            with torch.no_grad():
                x.mul_(1.0)
            return x
        return _manual_forward(model, b, between = write)

    for fn in (under_grad, under_no_grad):
        why = _abort_case(model, batches, items, [lambda b = b: fn(b) for b in batches])
        assert "in place" in why or "changed between" in why, why


def test_saved_tensor_hooks_are_refused():
    model = _model()
    batches, items = _batches(2)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        losses, why = lm.run_layer_major(_layers(model), [lambda b = b: _step(model, b, items) for b in batches],
                                         model = model)
    assert losses is None and "saved-tensor hooks" in why
    assert _grads(model) == {}


# ---------------------------------------------------------------- review fixes (each fails without its fix)
class _Toy(torch.nn.Module):
    def __init__(self, w = 16):
        super().__init__()
        self.lin = torch.nn.Linear(w, w)

    def forward(self, x, cache = None, **kwargs):
        if cache is not None:
            cache.n += 1
        return torch.tanh(self.lin(x))


def _toy_stack(cls = _Toy, n = 3, seed = 0):
    torch.manual_seed(seed)
    return torch.nn.ModuleList([cls() for _ in range(n)])


def _toy_run(layers, fns):
    for p in layers.parameters():
        p.grad = None
    return lm.run_layer_major(layers, fns, model = layers)


def _toy_x(m):
    return torch.randn(2, 5, 16, generator = torch.Generator().manual_seed(m))


def test_nested_auxiliary_layer_outputs_abort():
    class Aux(_Toy):
        def forward(self, x, **kwargs):
            h = torch.tanh(self.lin(x))
            return h, {"aux": h.square().mean()}

    layers = _toy_stack(Aux)

    def fn(m):
        x, aux = _toy_x(m), 0
        for layer in layers:
            x, extra = layer(x)
            aux = aux + extra["aux"]
        (x.sum() + aux).backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "not the hidden states" in why
    assert all(p.grad is None for p in layers.parameters())


def test_attention_recorder_aborts_or_matches():
    """transformers' output_attentions capture inside the decoder (no-grad there): abort, never a silent drop.
    Non-reentrant checkpointing: there standard accumulation differentiates the attentions."""
    model = _model(reentrant = False)
    batches, items = _batches(2)

    def step(b):
        out = model(**b, num_items_in_batch = items, output_attentions = True)
        loss = out.loss + out.attentions[1].float().square().mean()
        loss.backward()
        return loss.detach()

    model.zero_grad(set_to_none = True)
    base = [step(b) for b in batches]
    base_grads = _grads(model)
    model.zero_grad(set_to_none = True)
    losses, why = lm.run_layer_major(_layers(model), [lambda b = b: step(b) for b in batches], model = model)
    if losses is None:
        assert "output_attentions" in why or "output recorders" in why or "not the hidden states" in why, why
        assert _grads(model) == {}
    else:
        assert all(torch.equal(a, b) for a, b in zip(base, losses))
        _assert_same(base_grads, _grads(model))


@pytest.mark.parametrize("where", ["positional", "cache_params"])
def test_positional_and_state_caches_abort(where):
    class FakeCache:
        n = 0

    layers = _toy_stack()
    cache = FakeCache()

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x, cache) if where == "positional" else layer(x, cache_params = cache)
        x.sum().backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "live KV cache" in why
    assert cache.n == 0


def test_a_second_window_is_refused_while_one_runs():
    outer, inner = _toy_stack(seed = 0), _toy_stack(seed = 1)
    seen = []

    def fn(m):
        if m == 0:
            seen.append(_toy_run(inner, [lambda: None]))
        x = _toy_x(m)
        for layer in outer:
            x = layer(x)
        x.sum().backward()

    losses, why = _toy_run(outer, [functools.partial(fn, m) for m in range(2)])
    assert why is None
    assert seen[0][0] is None and "another layer-major window" in seen[0][1]
    assert "__call__" not in _Toy.__dict__


def test_saved_tensor_hooks_inside_a_micro_batch_abort():
    layers = _toy_stack()

    def fn(m):
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            x = _toy_x(m)
            for layer in layers:
                x = layer(x)
            x.sum().backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "inside the micro-batch" in why


def test_saved_tensor_hooks_detected_without_the_private_query(monkeypatch):
    monkeypatch.delattr(torch._C._autograd, "_top_saved_tensors_default_hooks", raising = False)
    assert lm._saved_tensor_hooks_active() is False
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        assert lm._saved_tensor_hooks_active() is True


def test_out_of_memory_saving_existing_gradients_is_a_refusal(monkeypatch):
    layers = _toy_stack()
    for p in layers.parameters():
        p.grad = torch.ones_like(p)

    def clone(self, *a, **k):
        raise torch.OutOfMemoryError("simulated")

    fns = [lambda: None, lambda: None]
    monkeypatch.setattr(torch.Tensor, "clone", clone)
    losses, why = lm.run_layer_major(layers, fns, model = layers)
    monkeypatch.undo()
    assert losses is None and "out of memory" in why
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in layers.parameters())


def test_replay_output_mismatch_rolls_back():
    class GradOnlyAux(_Toy):
        def forward(self, x, **kwargs):
            h = torch.tanh(self.lin(x))
            return (h, h.sum()) if torch.is_grad_enabled() else h

    layers = _toy_stack(GradOnlyAux)
    head = torch.nn.Parameter(torch.ones(16))

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
            x = x[0] if isinstance(x, tuple) else x
        (x * head).sum().backward()

    head.grad = None
    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "recompute" in why
    assert all(p.grad is None for p in layers.parameters())


def test_recompile_limit_in_the_backward_rolls_back():
    exc = pytest.importorskip("torch._dynamo.exc")
    kind = getattr(exc, "FailOnRecompileLimitHit", None) or getattr(exc, "RecompileLimitExceeded", None)
    if kind is None:
        pytest.skip("no recompile-limit exception in this torch")

    class Limited(_Toy):
        def forward(self, x, **kwargs):
            if torch.is_grad_enabled():
                raise kind("simulated")
            return torch.tanh(self.lin(x))

    layers = _toy_stack(Limited)

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.sum().backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "recompile limit" in why


def test_abort_restores_the_rng_state():
    """Draws made in an abandoned window (NEFTune noise) are not consumed: the retry draws the same numbers."""
    layers = _toy_stack()

    def fn(m):
        torch.rand(3)
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.sum().backward()
        layers[0](x)   # a second forward: protocol break in the loss part

    torch.manual_seed(11)
    expect = torch.rand(3)
    torch.manual_seed(11)
    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None
    assert torch.equal(torch.rand(3), expect)


def test_tied_trainable_embedding_and_head_are_refused():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
                                   num_attention_heads = 2, num_key_value_heads = 1, vocab_size = 64,
                                   tie_word_embeddings = True)
    model = transformers.LlamaForCausalLM(cfg)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    losses, why = lm.run_layer_major(model.model.layers, [lambda: None, lambda: None], model = model)
    assert losses is None and "share a trainable weight" in why


def test_input_hooks_do_not_outlive_the_window():
    layers = _toy_stack()
    x0 = torch.randn(2, 5, 16, requires_grad = True)

    def fn(m):
        x = x0
        for layer in layers:
            x = layer(x)
        x.sum().backward()

    for _ in range(3):
        losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
        assert why is None
    assert not (x0._backward_hooks or {})


def test_forward_input_keeps_requires_grad():
    """A reentrant checkpoint hands its no-grad forward an input that requires grad; compiled guards see the same."""
    layers = _toy_stack()
    seen = []
    h = layers[1].register_forward_pre_hook(lambda mod, args: seen.append((torch.is_grad_enabled(), args[0].requires_grad)))

    def fn(m):
        x = _toy_x(m).requires_grad_(True)
        for layer in layers:
            x = layer(x)
        x.sum().backward()

    try:
        losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    finally:
        h.remove()
    assert why is None
    assert (False, True) in seen


# ---------------------------------------------------------------- round 2 review fixes
@pytest.mark.parametrize("arch", sorted(FAMILIES))
def test_dense_families_bitwise(arch):
    """Every standard dense decoder family the trainer engages automatically: losses and LoRA gradients bitwise."""
    model = _model(False, arch = arch)
    batches, items = _batches()
    base_losses, base_grads = _standard(model, batches, items)
    losses, grads = _layer_major(model, batches, items)
    assert all(torch.equal(a, b) for a, b in zip(base_losses, losses))
    _assert_same(base_grads, grads)


def test_kv_shared_layers_are_refused():
    """Gemma 4 style: a consumer layer reads the key / value states a producer wrote (no-grad in the window)."""
    layers = _toy_stack()
    layers[2].attn = torch.nn.Module()
    layers[2].attn.is_kv_shared_layer = True

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.sum().backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "share key / value states" in why
    assert all(p.grad is None for p in layers.parameters())


def test_gemma4_kv_sharing_is_refused(monkeypatch):
    """The real case: Gemma 4's KV-sharing layers (transformers 5.x). Refused up front, and the runtime mapping
    check catches it without the up-front check too."""
    if not hasattr(transformers, "Gemma4ForCausalLM"):
        pytest.skip(f"transformers {transformers.__version__} has no Gemma 4")
    torch.manual_seed(0)
    cfg = transformers.Gemma4TextConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4,
                                        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
                                        vocab_size = 256, vocab_size_per_layer_input = 256,
                                        hidden_size_per_layer_input = 8, num_kv_shared_layers = 2,
                                        layer_types = ["sliding_attention", "full_attention"] * 2,
                                        sliding_window = 8, max_position_embeddings = 256)
    cfg._attn_implementation = "eager"
    model = transformers.Gemma4ForCausalLM(cfg)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    model = peft.get_peft_model(model, peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = TARGETS,
                                                       lora_dropout = 0.0, init_lora_weights = False))
    model.enable_input_require_grads()
    model.train()
    layers = model.get_base_model().model.layers
    assert any(getattr(m, "is_kv_shared_layer", False) is True for m in layers.modules())
    batches, items = _batches(2)
    fns = [lambda b = b: _step(model, b, items) for b in batches]
    model.zero_grad(set_to_none = True)
    losses, why = lm.run_layer_major(layers, fns, model = model)
    assert losses is None and "share key / value states" in why
    monkeypatch.setattr(lm, "_kv_shared_layers", lambda layers: False)
    losses, why = lm.run_layer_major(layers, fns, model = model)
    # The shared KV mapping, or the per-layer inputs that need gradients here, whichever the window meets first.
    assert losses is None and ("mapping" in why or "needs gradients" in why), why
    assert _grads(model) == {}


def test_state_shared_through_a_mapping_aborts():
    """A mapping passed to every layer (shared KV states without the flag): a producer's tensors would reach a
    later layer detached, dropping that path's gradient."""
    class Shared(_Toy):
        def forward(self, x, shared = None, **kwargs):
            if "kv" in shared:
                x = x + shared["kv"]
            h = torch.tanh(self.lin(x))
            shared.setdefault("kv", h * 0.5)
            return h

    layers = _toy_stack(Shared)

    def fn(m):
        x, shared = _toy_x(m), {}
        for layer in layers:
            x = layer(x, shared = shared)
        x.sum().backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "mapping" in why
    assert all(p.grad is None for p in layers.parameters())


def test_loss_part_writing_a_decoder_weight_gradient_aborts():
    """A regulariser on a decoder weight: its gradients would all land before any recompute contribution."""
    layers = _toy_stack()
    weight = layers[1].lin.weight

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        (x.sum() + 1e-3 * weight.square().sum()).backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "decoder layer weight" in why
    assert all(p.grad is None for p in layers.parameters())


def test_recompute_with_other_values_rolls_back():
    """A layer whose math depends on grad mode: same structure, other values, so the gradient is of another
    function."""
    class Flips(_Toy):
        def forward(self, x, **kwargs):
            h = torch.tanh(self.lin(x))
            return h if torch.is_grad_enabled() else -h

    layers = _toy_stack(Flips)

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.square().sum().backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "changed the hidden states" in why
    assert all(p.grad is None for p in layers.parameters())


@pytest.mark.parametrize("reentrant", [True, None, False])
def test_grad_mode_dependent_layers_under_a_checkpoint(reentrant):
    """A reentrant checkpoint runs standard training's forward without gradients too: the window is then that
    exact computation (no value check). Non-reentrant runs it under grad mode: the window must refuse."""
    class Ckpt(_Toy):
        def __init__(self):
            super().__init__()
            self.gradient_checkpointing = True
            kw = {} if reentrant is None else {"use_reentrant": reentrant}
            self._gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, **kw)

        def _inner(self, x):
            h = torch.tanh(self.lin(x))
            return h if torch.is_grad_enabled() else h * 0.5

        def forward(self, x, **kwargs):
            if self.gradient_checkpointing and self.training:
                return self._gradient_checkpointing_func(self._inner, x)
            return self._inner(x)

    layers = _toy_stack(Ckpt)

    def fn(m):
        x = _toy_x(m).requires_grad_(True)
        for layer in layers:
            x = layer(x)
        loss = x.square().sum()
        loss.backward()
        return loss.detach()

    for p in layers.parameters():
        p.grad = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = [fn(m) for m in range(2)]
        base_grads = [p.grad.clone() for p in layers.parameters()]
        losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    if reentrant is False:
        assert losses is None and "changed the hidden states" in why
        assert all(p.grad is None for p in layers.parameters())
    else:
        assert why is None
        assert all(torch.equal(a, b) for a, b in zip(base, losses))
        assert all(torch.equal(a, p.grad) for a, p in zip(base_grads, layers.parameters()))


def test_a_raising_window_restores_gradients_rng_and_swapper():
    layers = _toy_stack()
    head = torch.nn.Parameter(torch.ones(16))
    owner = torch.nn.ModuleList(list(layers))
    owner.head = torch.nn.Module()
    owner.head.w = head

    class Swap(_Holds):
        resets = 0

        def reset(self):
            Swap.resets += 1

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        (x * head).sum().backward()
        torch.rand(3)
        if m == 1:
            raise ValueError("loss part failed")

    head.grad = torch.full((16,), 7.0)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match = "loss part failed"):
        lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], swapper = Swap(), model = owner)
    assert torch.equal(head.grad, torch.full((16,), 7.0))
    assert torch.equal(torch.get_rng_state(), rng)
    assert Swap.resets == 1
    assert all(p.grad is None for p in layers.parameters())


def test_out_of_memory_ending_the_step_keeps_the_window():
    """The step's end (next step's prefetch, pool growth) running out of memory: the gradients are complete."""
    layers = _toy_stack()

    class Swap(_Holds):
        resets = 0

        @contextlib.contextmanager
        def layer_major_window(self):
            yield

        def _end_step(self):
            raise torch.OutOfMemoryError("simulated")

        def reset(self):
            Swap.resets += 1

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.sum().backward()
        return m

    for p in layers.parameters():
        p.grad = None
    losses, why = lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], swapper = Swap(),
                                     model = layers)
    assert why is None and losses == [0, 1]
    assert Swap.resets == 1
    assert all(p.grad is not None for p in layers.parameters())


# ---------------------------------------------------------------- round 3 review fixes
class _CkptToy(_Toy):
    """A decoder layer with its own checkpoint wrapper, as transformers' GradientCheckpointingLayer."""
    def __init__(self, func = None):
        super().__init__()
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = func or functools.partial(torch.utils.checkpoint.checkpoint,
                                                                     use_reentrant = True)

    def _inner(self, x, past_key_value = None):
        if past_key_value is not None:
            past_key_value.n += 1
        return torch.tanh(self.lin(x))

    def forward(self, x, **kwargs):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(functools.partial(self._inner, **kwargs), x)
        return self._inner(x, **kwargs)


def _ckpt_fns(layers, n = 2, **call):
    def fn(m):
        x = _toy_x(m).requires_grad_(True)
        for layer in layers:
            x = layer(x, **call)
        x.square().sum().backward()
    return [functools.partial(fn, m) for m in range(n)]


@pytest.mark.parametrize("how", ["context_fn", "unrecognised"])
def test_checkpoint_wrappers_the_window_cannot_reproduce_are_refused(how):
    """A checkpoint context_fn (autocast in the checkpointed region) or a wrapper the window does not know: the
    window bypasses the wrapper, so it would train without what the wrapper does."""
    if how == "context_fn":
        def ctx():
            return contextlib.nullcontext(), torch.autocast("cpu", dtype = torch.bfloat16)
        func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant = False, context_fn = ctx)
    else:
        def func(f, *args):
            with torch.autocast("cpu", dtype = torch.bfloat16):
                return torch.utils.checkpoint.checkpoint(f, *args, use_reentrant = True)
    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_CkptToy(func) for _ in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        losses, why = _toy_run(layers, _ckpt_fns(layers))
    assert losses is None and "checkpoint" in why, why
    assert all(p.grad is None for p in layers.parameters())


def test_unsloth_checkpoint_shims_are_refused(monkeypatch):
    """They replay MoE routing / saved-attention frames around the recompute; the window would bypass them."""
    import types

    def unsloth_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
        return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant = True)
    shims = sys.modules.get("unsloth_zoo.gradient_checkpointing")
    if shims is None:
        shims = types.ModuleType("unsloth_zoo.gradient_checkpointing")
        monkeypatch.setitem(sys.modules, "unsloth_zoo.gradient_checkpointing", shims)
    monkeypatch.setattr(shims, "unsloth_gradient_checkpoint", unsloth_gradient_checkpoint, raising = False)
    for func in (unsloth_gradient_checkpoint, functools.partial(unsloth_gradient_checkpoint, use_reentrant = False)):
        layer = _CkptToy(func)
        no_grad_forward, problem = lm._checkpoint_mode(layer)
        assert no_grad_forward is False and "routing" in problem
    layer.gradient_checkpointing = False
    assert lm._checkpoint_mode(layer) == (False, None)


def test_live_cache_under_a_cleared_keyword_aborts():
    """A cache under a keyword the checkpoint wrapper clears (the deprecated past_key_value alias, which the decoder
    still reads): checked on the call as made, before clearing."""
    class DynamicCache:
        n = 0

    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_CkptToy() for _ in range(3)])
    cache = DynamicCache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        losses, why = _toy_run(layers, _ckpt_fns(layers, past_key_value = cache))
    assert losses is None and "live KV cache" in why
    assert all(p.grad is None for p in layers.parameters())


def test_trainable_weights_shared_between_decoder_layers_are_refused():
    layers = _toy_stack()
    layers[2].lin = layers[0].lin

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.sum().backward()

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "share a trainable weight" in why
    assert all(p.grad is None for p in layers.parameters())


def test_an_error_ending_the_step_restores_gradients_rng_and_swapper():
    layers = _toy_stack()
    head = torch.nn.Parameter(torch.ones(16))
    owner = torch.nn.ModuleList(list(layers))
    owner.head = torch.nn.Module()
    owner.head.w = head

    class Swap(_Holds):
        resets = 0

        @contextlib.contextmanager
        def layer_major_window(self):
            yield

        def _end_step(self):
            raise RuntimeError("prefetch failed")

        def reset(self):
            Swap.resets += 1

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        (x * head).sum().backward()
        torch.rand(3)

    head.grad = torch.full((16,), 7.0)
    rng = torch.get_rng_state()
    with pytest.raises(RuntimeError, match = "prefetch failed"):
        lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], swapper = Swap(), model = owner)
    assert torch.equal(head.grad, torch.full((16,), 7.0))
    assert torch.equal(torch.get_rng_state(), rng)
    assert Swap.resets == 1
    assert all(p.grad is None for p in layers.parameters())


def test_an_interrupted_submission_leaves_no_worker_waiting():
    layers = _toy_stack()

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.sum().backward()
        return m

    fns = [functools.partial(fn, m) for m in range(2)]
    assert _toy_run(layers, fns)[1] is None   # two workers exist

    class Interrupting:
        def __init__(self, real):
            self.real, self.fired = real, False

        def put(self, job):
            if not self.fired:
                self.fired = True
                raise KeyboardInterrupt
            self.real.put(job)

    real = lm._WORKERS.queues[1]
    lm._WORKERS.queues[1] = Interrupting(real)
    try:
        with pytest.raises(KeyboardInterrupt):
            _toy_run(layers, fns)
    finally:
        lm._WORKERS.queues[1] = real
    assert all(p.grad is None for p in layers.parameters())
    out = []
    t = threading.Thread(target = lambda: out.append(_toy_run(layers, fns)), daemon = True)
    t.start()
    t.join(30)
    assert out and out[0][1] is None and out[0][0] == [0, 1]


# ---------------------------------------------------------------- round 4 review fixes
def test_wrapped_checkpoint_functions_are_refused():
    """functools.wraps copies torch's checkpoint name, module and __wrapped__: identity decides."""
    @functools.wraps(torch.utils.checkpoint.checkpoint)
    def scaled(function, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(lambda *a: function(*a) * 2, *args, use_reentrant = True)

    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_CkptToy(scaled) for _ in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        losses, why = _toy_run(layers, _ckpt_fns(layers))
    assert losses is None and "checkpoint wrapper" in why, why
    assert all(p.grad is None for p in layers.parameters())


def test_reentrant_checkpoint_without_a_grad_input_is_refused():
    """A reentrant checkpoint gives the layers no gradient when no positional input needs one; the window would."""
    torch.manual_seed(0)
    layers = torch.nn.ModuleList([_CkptToy() for _ in range(3)])
    head = torch.nn.Parameter(torch.ones(16))
    owner = torch.nn.ModuleList(list(layers))
    owner.head = torch.nn.Module()
    owner.head.w = head

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        (x * head).sum().backward()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        losses, why = lm.run_layer_major(layers, [functools.partial(fn, m) for m in range(2)], model = owner)
    assert losses is None and "reentrant checkpoint" in why, why
    assert head.grad is None and all(p.grad is None for p in layers.parameters())


class _Scale(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args = (), kwargs = None):
        out = func(*args, **(kwargs or {}))
        return out * 2 if func is torch.nn.functional.linear else out


@pytest.mark.parametrize("where", ["caller", "micro-batch"])
def test_operator_modes_are_refused(where):
    layers = _toy_stack()

    def fn(m):
        with _Scale() if where == "micro-batch" else contextlib.nullcontext():
            x = _toy_x(m)
            for layer in layers:
                x = layer(x)
            x.sum().backward()

    fns = [functools.partial(fn, m) for m in range(2)]
    with _Scale() if where == "caller" else contextlib.nullcontext():
        losses, why = _toy_run(layers, fns)
    assert losses is None and "TorchFunctionMode" in why, why
    assert all(p.grad is None for p in layers.parameters())


def test_a_micro_batch_raising_layer_major_abort_rolls_back():
    layers = _toy_stack()

    def fn(m):
        x = _toy_x(m)
        for layer in layers:
            x = layer(x)
        x.sum().backward()
        if m == 1:
            raise lm.LayerMajorAbort("raised by the micro-batch")
        return m

    losses, why = _toy_run(layers, [functools.partial(fn, m) for m in range(2)])
    assert losses is None and "raised by the micro-batch" in why
    assert all(p.grad is None for p in layers.parameters())


def test_failed_interceptor_setup_restores_every_class():
    """An interrupt while patching the second decoder class: the first one's __call__ is put back too."""
    class Meta(type(_Toy)):
        calls = 0

        def __setattr__(cls, name, value):
            if name == "__call__":
                Meta.calls += 1
                if Meta.calls == 2:
                    raise KeyboardInterrupt
            super().__setattr__(name, value)

    A = Meta("A", (_Toy,), {})
    B = Meta("B", (_Toy,), {})
    layers = torch.nn.ModuleList([A(), B(), A()])
    with pytest.raises(KeyboardInterrupt):
        with lm._intercept(None, layers):
            pass
    assert Meta.calls >= 2
    assert "__call__" not in A.__dict__ and "__call__" not in B.__dict__


def test_boundary_bytes():
    assert lm.boundary_bytes([2 * 12, 2 * 17], 64, 4, 2) == (24 + 34) * 64 * 5 * 2


def test_mode_switch(monkeypatch):
    for v, want in (("0", "off"), ("1", "on"), ("auto", "auto"), ("", "auto")):
        monkeypatch.setenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", v)
        assert lm.layer_major_mode() == want


# ---------------------------------------------------------------- GPU: a real BlockSwap
def _block_swap():
    spec = importlib.util.find_spec("unsloth_zoo.block_swap")
    if spec is None or not spec.origin.startswith(os.path.join(_HERE, "unsloth_zoo")):
        for name in ("block_swap_codec", "block_swap_disk"):
            s = importlib.util.spec_from_file_location(f"unsloth_zoo.{name}", os.path.join(_HERE, "unsloth_zoo",
                                                                                          f"{name}.py"))
            mod = importlib.util.module_from_spec(s)
            s.loader.exec_module(mod)
            sys.modules.setdefault(f"unsloth_zoo.{name}", mod)
        s = importlib.util.spec_from_file_location("block_swap_lm_under_test",
                                                   os.path.join(_HERE, "unsloth_zoo", "block_swap.py"))
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        return mod
    import unsloth_zoo.block_swap as mod
    return mod


@pytest.fixture
def _spill_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_OFFLOAD_DISK_DIR", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_OFFLOAD_DISK", raising = False)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_HOST_LIMIT", raising = False)
    yield


def _gpu_run(layer_major, depth, disk, steps = 2, n = 3, reentrant = True, arch = "llama"):
    bs = _block_swap()
    model = _model(reentrant, layers = 6, dtype = torch.bfloat16, device = "cuda", arch = arch)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    layers = _layers(model)
    if disk:
        d = bs._disk()
        d._SPILL[:] = []
        per = bs._Block(layers[1], {}, torch.device("cuda", torch.cuda.current_device())).nbytes()
        # The staging ring, a block of release lag and one layer in RAM: the other four go to disk.
        os.environ["UNSLOTH_OFFLOAD_HOST_LIMIT"] = str(2 * d._round_up(per) + per + per + 1024)
    try:
        sw = bs.BlockSwap(layers, 5, prefetch_depth = depth, placement = "spread", model = model)
    finally:
        os.environ.pop("UNSLOTH_OFFLOAD_HOST_LIMIT", None)
    batches, items = _batches(n, device = "cuda")
    out = []
    try:
        for s in range(steps):
            model.zero_grad(set_to_none = True)
            sw.stats()
            if layer_major:
                losses, why = lm.run_layer_major(layers, [lambda b = b: _step(model, b, items) for b in batches],
                                                 swapper = sw, model = model)
                assert why is None, why
            else:
                losses = [_step(model, b, items) for b in batches]
            torch.cuda.synchronize()
            st = sw.stats()
            out.append((losses, _grads(model), st["copies"], st.get("disk_reads", 0), st.get("disk_layers", 0)))
    finally:
        sw.remove()
    return out


@_cuda
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("disk", [False, True])
def test_gpu_block_swap_bitwise_and_fewer_fetches(_spill_dir, depth, disk):
    base = _gpu_run(False, depth, disk)
    head = _gpu_run(True, depth, disk)
    if disk:
        assert base[0][4] == head[0][4] == 4, (base[0][4], head[0][4])
    for (bl, bg, bc, br, _), (hl, hg, hc, hr, _) in zip(base, head):
        assert all(torch.equal(a, b) for a, b in zip(bl, hl))
        _assert_same(bg, hg)
    # Steady step: 2 x 5 - 2 x depth fetches per micro-batch (the first and last depth blocks stay on the card
    # across the turnarounds) standard, the same per step layer-major: 3 micro-batches, 3x fewer.
    (_, _, bc, br, _), (_, _, hc, hr, _) = base[-1], head[-1]
    assert bc == 3 * (2 * 5 - 2 * depth) and hc == 2 * 5 - 2 * depth, (bc, hc)
    if disk:
        assert hr > 0 and br == 3 * hr, (br, hr)


@_cuda
def test_gpu_worker_uses_the_callers_stream_on_the_model_device():
    """Model and inputs on card 1 produced on a side stream while card 0 is current: the workers run on that
    stream, so they read the inputs after they are written (a race on card 1's default stream otherwise)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    dev = torch.device("cuda", 1)
    model = _model(device = dev)
    src, items = _batches(2, device = dev)
    side = torch.cuda.Stream(dev)

    def produce():
        out = []
        with torch.cuda.device(dev):
            for b in src:
                torch.cuda._sleep(200_000_000)
                out.append({k: v.clone() for k, v in b.items()})
        return out

    with torch.cuda.stream(side), torch.cuda.device(0):
        base_losses, base_grads = _standard(model, produce(), items)
        model.zero_grad(set_to_none = True)
        batches = produce()
        losses, why = lm.run_layer_major(_layers(model), [lambda b = b: _step(model, b, items) for b in batches],
                                         model = model)
    torch.cuda.synchronize(dev)
    assert why is None
    assert all(torch.equal(a, b) for a, b in zip(base_losses, losses))
    _assert_same(base_grads, _grads(model))


@_cuda
def test_gpu_block_swap_step_ends_once_per_window_and_resets_after_an_abort(_spill_dir):
    bs = _block_swap()
    model = _model(True, layers = 6, dtype = torch.bfloat16, device = "cuda")
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    layers = _layers(model)
    sw = bs.BlockSwap(layers, 5, prefetch_depth = 1, placement = "spread", model = model)
    batches, items = _batches(3, device = "cuda")
    try:
        steps = []
        for layer_major in (False, True):
            model.zero_grad(set_to_none = True)
            before = sw._steps
            if layer_major:
                assert lm.run_layer_major(layers, [lambda b = b: _step(model, b, items) for b in batches],
                                          swapper = sw, model = model)[1] is None
            else:
                for b in batches:
                    _step(model, b, items)
            steps.append(sw._steps - before)
        assert steps == [3, 1], steps

        def oom(mod, args, out):
            if torch.is_grad_enabled():
                raise torch.OutOfMemoryError("simulated")

        h = layers[3].register_forward_hook(oom)
        try:
            model.zero_grad(set_to_none = True)
            losses, why = lm.run_layer_major(layers, [lambda b = b: _step(model, b, items) for b in batches],
                                             swapper = sw, model = model)
        finally:
            h.remove()
        assert losses is None and "out of memory" in why
        assert not any(b.pending for b in sw.blocks)
        assert _grads(model) == {}
        model.zero_grad(set_to_none = True)
        for b in batches:
            _step(model, b, items)
    finally:
        sw.remove()


@_cuda
@pytest.mark.parametrize("arch", sorted(FAMILIES))
@pytest.mark.parametrize("disk", [False, True])
def test_gpu_block_swap_dense_families_bitwise(_spill_dir, arch, disk):
    base = _gpu_run(False, 1, disk, arch = arch)
    head = _gpu_run(True, 1, disk, arch = arch)
    for (bl, bg, bc, br, _), (hl, hg, hc, hr, _) in zip(base, head):
        assert all(torch.equal(a, b) for a, b in zip(bl, hl))
        _assert_same(bg, hg)
    bc, hc = base[-1][2], head[-1][2]
    assert bc == 3 * hc and hc > 0, (bc, hc)


@_cuda
def test_gpu_pool_growth_out_of_memory_keeps_the_pool(monkeypatch):
    bs = _block_swap()
    model = _model(True, layers = 6, dtype = torch.bfloat16, device = "cuda")
    layers = _layers(model)
    sw = bs.BlockSwap(layers, 5, prefetch_depth = 1, placement = "spread", model = model)
    try:
        before = {k: len(v) for k, v in sw.free.items()}
        calls = []

        def new_slot(sig):
            calls.append(sig)
            raise torch.cuda.OutOfMemoryError("simulated")

        monkeypatch.setattr(sw, "_new_slot", new_slot)
        monkeypatch.setattr(bs, "_free_device_bytes", lambda dv: 1 << 50)
        assert sw._grow_pool() is False
        assert calls and {k: len(v) for k, v in sw.free.items()} == before
    finally:
        monkeypatch.undo()
        sw.remove()
