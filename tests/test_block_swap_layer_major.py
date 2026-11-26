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

import copy
import importlib.util
import os
import sys

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


def _model(reentrant = True, layers = 4, dtype = torch.float32, device = "cpu", train_embed = False,
           train_head = False, seed = 0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = layers,
                                   num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 256,
                                   max_position_embeddings = 256, tie_word_embeddings = False)
    cfg._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(cfg).to(dtype = dtype, device = device)
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


@pytest.mark.skipif(not hasattr(torch._C._autograd, "_top_saved_tensors_default_hooks"),
                    reason = "torch without the saved-tensor hooks query")
def test_saved_tensor_hooks_are_refused():
    model = _model()
    batches, items = _batches(2)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        losses, why = lm.run_layer_major(_layers(model), [lambda b = b: _step(model, b, items) for b in batches],
                                         model = model)
    assert losses is None and "saved-tensor hooks" in why
    assert _grads(model) == {}


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


def _gpu_run(layer_major, depth, disk, steps = 2, n = 3, reentrant = True):
    bs = _block_swap()
    model = _model(reentrant, layers = 6, dtype = torch.bfloat16, device = "cuda")
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
