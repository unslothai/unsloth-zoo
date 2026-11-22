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

"""CPU torch.compile default: selection, the step-count gate, eager fallback, parity."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _tiny_model():
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(_TINY, dtype = torch.float32, attn_implementation = "sdpa")
    except Exception as error:  # offline without the tiny model cached
        pytest.skip(f"{_TINY} unavailable: {error}")


def test_cpu_compile_auto_selection(monkeypatch):
    from unsloth_zoo.cpu import loader
    import transformers
    monkeypatch.delenv("UNSLOTH_CPU_COMPILE", raising = False)
    dense = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2, num_attention_heads = 4, vocab_size = 64))
    assert loader._cpu_compile_wanted(dense) == "auto"
    monkeypatch.setenv("UNSLOTH_CPU_COMPILE", "0")
    assert loader._cpu_compile_wanted(dense) is None
    monkeypatch.setenv("UNSLOTH_CPU_COMPILE", "1")
    assert loader._cpu_compile_wanted(dense) == "explicit"
    monkeypatch.delenv("UNSLOTH_CPU_COMPILE")
    dense.config.quantization_config = {"quant_method": "bitsandbytes"}
    assert loader._cpu_compile_wanted(dense) is None  # quantized: eager
    moe = transformers.MixtralForCausalLM(transformers.MixtralConfig(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 1, num_attention_heads = 4, vocab_size = 64,
        num_local_experts = 2, num_key_value_heads = 2))
    assert loader._cpu_compile_wanted(moe) is None  # not on the allowlist


def test_cpu_compile_auto_compiles_and_matches_eager(monkeypatch):
    """Default (unset) marks a dense model for compile; compiled fp32 loss / grads equal eager."""
    from unsloth_zoo.cpu import loader
    if loader._cpu_compile_unavailable() is not None:
        pytest.skip("no C++ toolchain for Inductor here")
    from peft import LoraConfig, get_peft_model
    results = []
    for value in ("0", None):
        if value is None:
            monkeypatch.delenv("UNSLOTH_CPU_COMPILE", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_CPU_COMPILE", value)
        torch.manual_seed(0)
        model = _tiny_model()
        loader.FastCPUModel._prepare(model, False)
        # Auto mode only marks the model; the trainer's gate compiles once it has timed an eager step.
        assert not getattr(model, "_unsloth_cpu_compiled", False)
        assert bool(vars(model).get("_unsloth_cpu_compile_pending", False)) is (value is None)
        if value is None:
            assert loader.compile_cpu_layers(model) == len(model.model.layers)
        torch.manual_seed(1)
        lora = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj", "up_proj"], init_lora_weights = False))
        ids = torch.randint(3, 1000, (2, 17), generator = torch.Generator().manual_seed(0))
        loss = lora(input_ids = ids, labels = ids, use_cache = False).loss
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in lora.parameters() if p.requires_grad])
        results.append((loss.detach(), grad))
        torch._dynamo.reset()
    torch.testing.assert_close(results[0][0], results[1][0], rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(results[0][1], results[1][1], rtol = 1e-4, atol = 1e-6)


def test_compile_error_on_a_layer_falls_back_to_eager(monkeypatch, capsys):
    from unsloth_zoo.cpu import loader
    model = _tiny_model()
    model.config.use_cache = False
    ids = torch.randint(3, 1000, (1, 9), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        reference = model(input_ids = ids, use_cache = False).logits
    BackendCompilerFailed = type("BackendCompilerFailed", (RuntimeError,), {"__module__": "torch._dynamo.exc"})
    calls = []

    def broken_compile(self, *args, **kwargs):
        def fail(*a, **k):
            calls.append(1)
            raise BackendCompilerFailed("C++ build failed")
        self._compiled_call_impl = fail
    monkeypatch.setattr(torch.nn.Module, "compile", broken_compile)
    assert loader._compile_layers(model) == len(model.model.layers)
    with torch.no_grad():
        out = model(input_ids = ids, use_cache = False).logits
    torch.testing.assert_close(out, reference)
    assert len(calls) == 1  # only the first layer tried; the rest run eagerly
    assert all(layer._compiled_call_impl is None for layer in model.model.layers)
    assert "training eagerly" in capsys.readouterr().out
    # A genuine model error is not swallowed: the eager rerun raises it.
    def model_bug(self, *args, **kwargs):
        def fail(*a, **k):
            raise ValueError("shape bug")
        self._compiled_call_impl = fail
    monkeypatch.setattr(torch.nn.Module, "compile", model_bug)
    loader._compile_layers(model)
    layer_cls = type(model.model.layers[0])

    def eager_bug(self, *a, **k):
        raise ValueError("shape bug")
    monkeypatch.setattr(layer_cls, "forward", eager_bug)
    with pytest.raises(ValueError, match = "shape bug"):
        with torch.no_grad():
            model(input_ids = ids, use_cache = False)


def test_compile_fallback_lets_checkpoint_control_flow_through(monkeypatch):
    """Non-reentrant checkpointing stops recomputation with _StopRecomputationError: never a reason to go eager."""
    import torch.utils.checkpoint as checkpoint
    from unsloth_zoo.cpu import loader
    stop = getattr(checkpoint, "_StopRecomputationError", None)
    if stop is None:
        pytest.skip("this torch has no _StopRecomputationError")
    model = _tiny_model()

    def raises_stop(self, *args, **kwargs):
        def fail(*a, **k):
            raise stop()
        self._compiled_call_impl = fail
    monkeypatch.setattr(torch.nn.Module, "compile", raises_stop)
    loader._compile_layers(model)
    with pytest.raises(stop):
        model.model.layers[0](torch.zeros(1, 2, model.config.hidden_size))
    assert all(layer._compiled_call_impl is not None for layer in model.model.layers)


def test_compiled_layers_survive_checkpointed_training(monkeypatch, capsys):
    """Compile + non-reentrant checkpointing + backward: no fallback message, layers stay compiled, grads = eager."""
    from unsloth_zoo.cpu import loader
    if loader._cpu_compile_unavailable() is not None:
        pytest.skip("no C++ toolchain for Inductor here")
    from peft import LoraConfig, get_peft_model
    grads = []
    for value in ("0", "1"):
        monkeypatch.setenv("UNSLOTH_CPU_COMPILE", value)
        torch.manual_seed(0)
        model = _tiny_model()
        model.config.use_cache = False
        loader.FastCPUModel._prepare(model, True)  # checkpointing on, then compile (explicit: no gate)
        torch.manual_seed(1)
        lora = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj"], init_lora_weights = False))
        lora.enable_input_require_grads()
        lora.train()
        ids = torch.randint(3, 1000, (2, 17), generator = torch.Generator().manual_seed(0))
        for _ in range(2):
            lora.zero_grad()
            lora(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in lora.parameters() if p.requires_grad]))
        if value == "1":
            assert all(layer._compiled_call_impl is not None for layer in model.model.layers)
        torch._dynamo.reset()
    assert "training eagerly" not in capsys.readouterr().out
    torch.testing.assert_close(grads[0], grads[1], rtol = 1e-4, atol = 1e-6)


def test_compile_probe_reports_dynamo_disabled(monkeypatch):
    from unsloth_zoo.cpu import loader
    loader._cpu_compile_unavailable.cache_clear()
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    try:
        assert loader._cpu_compile_unavailable() == "TORCHDYNAMO_DISABLE=1"
    finally:
        loader._cpu_compile_unavailable.cache_clear()


@pytest.mark.parametrize("step_s, steps_left, expected", [
    (8.0, 58, False),    # Qwen3-0.6B notebook (60 steps) on this host: 93 s saved < 150 s
    (8.0, 98, True),     # 100 steps: 157 s saved
    (2.0, 28, False),    # a 30-step smoke test
    (30.0, 0, False),    # nothing left
])
def test_compile_gate_arithmetic(monkeypatch, step_s, steps_left, expected):
    from unsloth_zoo.cpu import trainer
    monkeypatch.delenv("UNSLOTH_CPU_COMPILE_COST_S", raising = False)
    monkeypatch.delenv("UNSLOTH_CPU_COMPILE_SAVING", raising = False)
    worth, saved, cost = trainer._compile_worth_it(step_s, steps_left)
    assert worth is expected and cost == 150.0 and saved == pytest.approx(0.2 * step_s * max(0, steps_left))
    monkeypatch.setenv("UNSLOTH_CPU_COMPILE_COST_S", "10")
    assert trainer._compile_worth_it(step_s, steps_left)[0] is (steps_left > 0 and 0.2 * step_s * steps_left > 10)


def _gate_run(monkeypatch, tmp_path, max_steps, cost):
    from datasets import Dataset
    from transformers import TrainingArguments
    from unsloth_zoo.cpu import loader, trainer
    monkeypatch.delenv("UNSLOTH_CPU_COMPILE", raising = False)
    monkeypatch.setenv("UNSLOTH_CPU_COMPILE_COST_S", str(cost))
    compiled = []
    monkeypatch.setattr(loader, "compile_cpu_layers", lambda model, explicit = False: compiled.append(model) or 1)
    model = _tiny_model()
    loader.FastCPUModel._prepare(model, False)
    assert vars(model).get("_unsloth_cpu_compile_pending")
    from transformers import Trainer
    ids = [list(range(5, 21))] * 8
    data = Dataset.from_dict({"input_ids": ids, "labels": ids})
    args = TrainingArguments(output_dir = str(tmp_path), max_steps = max_steps, per_device_train_batch_size = 2,
                             report_to = "none", use_cpu = True, save_strategy = "no", logging_strategy = "no")
    t = Trainer(model = model, args = args, train_dataset = data)
    assert trainer._install_compile_gate(t) is True
    t.train()
    return compiled, model


def test_compile_gate_short_run_stays_eager(monkeypatch, tmp_path, capsys):
    compiled, model = _gate_run(monkeypatch, tmp_path, max_steps = 4, cost = 150)
    assert compiled == [] and not vars(model).get("_unsloth_cpu_compile_pending")
    assert "staying eager" in capsys.readouterr().out


def test_compile_gate_long_enough_run_compiles(monkeypatch, tmp_path):
    compiled, model = _gate_run(monkeypatch, tmp_path, max_steps = 4, cost = 0)
    assert compiled == [model]


def test_compile_gate_not_installed_when_overridden(monkeypatch):
    from unsloth_zoo.cpu import loader, trainer
    for value in ("0", "1"):
        monkeypatch.setenv("UNSLOTH_CPU_COMPILE", value)
        model = _tiny_model()
        monkeypatch.setattr(loader, "_compile_layers", lambda m: 0)
        loader.FastCPUModel._prepare(model, False)
        assert not vars(model).get("_unsloth_cpu_compile_pending")

        class T:
            pass
        t = T()
        t.model = model
        assert trainer._install_compile_gate(t) is None


def test_backward_only_compile_error_falls_back_to_eager(monkeypatch, capsys):
    """A layer whose forward lowers but whose backward cannot: AOTAutograd would defer that failure to
    loss.backward(), past the fallback. The guarded call lowers the backward inside the forward instead."""
    import torch._functorch.config as functorch_config
    from unsloth_zoo.cpu import loader
    if not hasattr(functorch_config, "force_non_lazy_backward_lowering"):
        pytest.skip("this torch always lowers the backward lazily")
    if loader._cpu_compile_unavailable() is not None:
        pytest.skip("no C++ toolchain for Inductor here")
    import torch._inductor.compile_fx as compile_fx
    original = compile_fx._compile_fx_inner

    def backward_fails(*args, **kwargs):
        if kwargs.get("is_backward"):
            raise RuntimeError("backward lowering failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(compile_fx, "_compile_fx_inner", backward_fails)
    # Importing unsloth's GPU model utilities (as other tests in this process may) sets suppress_errors = True, and
    # Dynamo then swallows the injected failure before the fallback sees it; a CPU-only Unsloth process leaves it False.
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    from peft import LoraConfig, get_peft_model
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = _tiny_model()
    model.config.use_cache = False
    assert loader._compile_layers(model) == len(model.model.layers)
    lora = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj"]))
    ids = torch.randint(3, 1000, (2, 11), generator = torch.Generator().manual_seed(0))
    try:
        lora(input_ids = ids, labels = ids, use_cache = False).loss.backward()
    finally:
        torch._dynamo.reset()
    assert all(layer._compiled_call_impl is None for layer in model.model.layers)
    out = capsys.readouterr().out
    assert "training eagerly" in out and "backward lowering failed" in out


def test_compile_error_on_a_later_layer_keeps_checkpoint_recompute_compiled(monkeypatch, capsys):
    """Non-reentrant checkpointing: layer 0's compiled forward already ran when the last layer fails to compile.
    Recomputing layer 0 eagerly in backward saves different tensors (CheckpointError); it replays compiled once,
    then every layer is eager. Grads equal an all-eager run."""
    from unsloth_zoo.cpu import loader
    if loader._cpu_compile_unavailable() is not None:
        pytest.skip("no C++ toolchain for Inductor here")
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    BackendCompilerFailed = type("BackendCompilerFailed", (RuntimeError,), {"__module__": "torch._dynamo.exc"})
    real_compile = torch.nn.Module.compile
    ids = torch.randint(3, 1000, (2, 17), generator = torch.Generator().manual_seed(0))
    grads = []
    for compile_layers in (False, True):
        torch.manual_seed(0)
        model = _tiny_model()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        layers = list(model.model.layers)
        if compile_layers:
            def compile_all_but_last(self, *args, **kwargs):
                if self is layers[-1]:
                    def fail(*a, **k):
                        raise BackendCompilerFailed("C++ build failed")
                    self._compiled_call_impl = fail
                else:
                    real_compile(self, *args, **kwargs)
            monkeypatch.setattr(torch.nn.Module, "compile", compile_all_but_last)
            loader._compile_layers(model)
            monkeypatch.setattr(torch.nn.Module, "compile", real_compile)
        try:
            for _ in range(2):  # the failing step, then a fully eager one
                model.zero_grad()
                model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        finally:
            torch._dynamo.reset()
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
        if compile_layers:
            assert all(layer._compiled_call_impl is None for layer in layers)
    assert "training eagerly" in capsys.readouterr().out
    torch.testing.assert_close(grads[0], grads[1], rtol = 1e-4, atol = 1e-6)


def test_compile_error_after_a_graph_break_restarts_the_checkpoint_eagerly(monkeypatch, capsys):
    """A layer with a graph break runs its first compiled segment (Linear / GELU), then the second segment fails to
    compile. Rerunning eagerly inside the same non-reentrant checkpoint would leave the first segment's saved tensors
    in it and backward raises CheckpointError; the whole checkpointed call restarts eagerly instead."""
    from functools import partial
    from transformers.modeling_layers import GradientCheckpointingLayer
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)

    class Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.up, self.down = torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)

        def forward(self, x):
            x = torch.nn.functional.gelu(self.up(x))
            torch._dynamo.graph_break()
            return self.down(x) * 2.0

    graphs = []

    def first_segment_only(gm, example_inputs):
        graphs.append(gm)
        if len(graphs) > 1:
            raise RuntimeError("C++ build failed")
        return gm.forward

    x = torch.randn(4, 8, generator = torch.Generator().manual_seed(0))
    grads = []
    for compiled in (False, True):
        torch.manual_seed(0)
        blocks = torch.nn.ModuleList([Block(), Block()]).train()
        for block in blocks:
            block.gradient_checkpointing = True
            block._gradient_checkpointing_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant = False)
        if compiled:
            for block in blocks:
                block.compile(backend = first_segment_only)
            loader._fall_back_to_eager_on_compile_error(blocks)
        try:
            for _ in range(2):  # the failing step, then a fully eager one
                blocks.zero_grad()
                h = x.clone().requires_grad_(True)
                for block in blocks:
                    h = block(h)
                h.square().sum().backward()
        finally:
            torch._dynamo.reset()
        grads.append(torch.cat([p.grad.flatten() for p in blocks.parameters()]))
        if compiled:
            assert len(graphs) >= 2  # the first segment really ran compiled before the second failed
            assert all(block._compiled_call_impl is None for block in blocks)
    assert "training eagerly" in capsys.readouterr().out
    torch.testing.assert_close(grads[0], grads[1])


def test_cached_forward_never_appends_twice_after_a_compile_failure(monkeypatch, capsys):
    """Cached inference through compiled layers: a segment that updates the KV cache, a graph break, then a segment
    that fails to compile. A retry of the layer call would append the same tokens to the cache again; cached calls
    run eagerly, so each layer's cache grows by exactly the tokens of the forward."""
    from transformers.modeling_layers import GradientCheckpointingLayer
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)

    class Block(GradientCheckpointingLayer):
        def __init__(self, index):
            super().__init__()
            self.index = index
            self.up, self.down = torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)

        def forward(self, x, past_key_values = None):
            x = torch.nn.functional.gelu(self.up(x))
            if past_key_values is not None:
                past_key_values[self.index].append(x.detach())
            torch._dynamo.graph_break()
            return self.down(x) * 2.0

    graphs = []

    def first_segment_only(gm, example_inputs):
        graphs.append(gm)
        if len(graphs) > 1:
            raise RuntimeError("C++ build failed")
        return gm.forward

    x = torch.randn(1, 3, 8, generator = torch.Generator().manual_seed(0))
    outs = []
    for compiled in (False, True):
        torch.manual_seed(0)
        blocks = torch.nn.ModuleList([Block(0), Block(1)]).eval()
        if compiled:
            for block in blocks:
                block.compile(backend = first_segment_only)
            loader._fall_back_to_eager_on_compile_error(blocks)
        cache = [[], []]
        try:
            with torch.no_grad():
                h = x
                for block in blocks:
                    h = block(h, past_key_values = cache)
        finally:
            torch._dynamo.reset()
        assert [len(c) for c in cache] == [1, 1], [len(c) for c in cache]
        outs.append(h)
    torch.testing.assert_close(outs[0], outs[1])


@pytest.mark.parametrize("reentrant", [False, True])
def test_two_outstanding_checkpoints_of_one_layer_replay_in_their_own_mode(monkeypatch, capsys, reentrant):
    """Two checkpointed forwards through the same layer before one backward, the second shape failing to compile:
    the first (compiled) checkpoint recomputes compiled, the second (retried eagerly) recomputes eagerly instead of
    retrying the failed compile in backward."""
    from functools import partial
    from transformers.modeling_layers import GradientCheckpointingLayer
    from unsloth_zoo.cpu import loader
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)

    class Block(GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.up, self.down = torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)

        def forward(self, x):
            return self.down(torch.nn.functional.gelu(self.up(x)))

    def first_shape_only(gm, example_inputs):
        if any(isinstance(t, torch.Tensor) and t.shape[0] == 3 for t in example_inputs):
            raise RuntimeError("C++ build failed")
        return gm.forward

    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(4, 8, generator = g), torch.randn(3, 8, generator = g)]
    grads = []
    for compiled in (False, True):
        torch.manual_seed(0)
        block = Block().train()
        block.gradient_checkpointing = True
        block._gradient_checkpointing_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant = reentrant)
        if compiled:
            block.compile(backend = first_shape_only, dynamic = False)
            loader._fall_back_to_eager_on_compile_error([block])
        try:
            outs = [block(x.clone().requires_grad_(True)) for x in xs]
            sum(o.square().sum() for o in outs).backward()
        finally:
            torch._dynamo.reset()
        grads.append(torch.cat([p.grad.flatten() for p in block.parameters()]))
        if compiled:
            assert block._compiled_call_impl is None
    assert "training eagerly" in capsys.readouterr().out
    torch.testing.assert_close(grads[0], grads[1])
