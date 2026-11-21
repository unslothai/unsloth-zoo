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

"""CPU memory defaults: checkpointing off when activations fit, grouped K / V in masked SDPA, failed 4-bit load released."""

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


class _GcArgs:
    def __init__(self, gc = True, length = 128, batch = 2):
        self.gradient_checkpointing = gc
        self.max_length = length
        self.per_device_train_batch_size = batch


def _gc_model(choice):
    model = _tiny_model()
    model._unsloth_cpu_gradient_checkpointing = choice
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    return model


@pytest.mark.parametrize("choice, ram, env, expect_off", [
    ("unsloth", 1 << 40, None, True),   # fits: off
    ("unsloth", 1 << 20, None, False),  # 1 MB free: keep recomputation
    (True, 1 << 40, None, False),       # the user's explicit True
    ("unsloth", 1 << 40, "0", False),   # UNSLOTH_CPU_AUTO_GC=0
])
def test_auto_gradient_checkpointing(monkeypatch, choice, ram, env, expect_off):
    from unsloth_zoo.cpu import trainer
    monkeypatch.setattr(trainer, "available_ram", lambda: ram)
    if env is not None:
        monkeypatch.setenv("UNSLOTH_CPU_AUTO_GC", env)
    model = _gc_model(choice)
    args = _GcArgs()
    result = trainer._auto_cpu_gradient_checkpointing(model, args)
    assert bool(result) is expect_off
    assert args.gradient_checkpointing is (not expect_off)
    assert model.is_gradient_checkpointing is (not expect_off)
    # Idempotent: a second trainer on the same model / args changes nothing more.
    trainer._auto_cpu_gradient_checkpointing(model, args)
    assert args.gradient_checkpointing is (not expect_off)


def test_gradient_checkpointing_off_keeps_loss_and_grads():
    model = _tiny_model()
    model.config.use_cache = False
    model.train()
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 1000, (2, 33), generator = generator)
    results = []
    for gc in (True, False):
        if gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        model.zero_grad()
        loss = model(input_ids = ids, labels = ids, use_cache = False).loss
        loss.backward()
        results.append((loss.detach(), model.model.layers[0].mlp.up_proj.weight.grad.clone()))
    torch.testing.assert_close(results[0][0], results[1][0], rtol = 0, atol = 0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol = 1e-6, atol = 1e-7)


def test_use_gradient_checkpointing_false_beats_trl_default(monkeypatch):
    """SFTConfig defaults gradient_checkpointing = True; the model's use_gradient_checkpointing = False wins."""
    from unsloth_zoo.cpu import trainer
    monkeypatch.setattr(trainer, "available_ram", lambda: 0)
    model = _tiny_model()
    model._unsloth_cpu_gradient_checkpointing = False
    args = _GcArgs()
    assert trainer._auto_cpu_gradient_checkpointing(model, args) is True
    assert args.gradient_checkpointing is False


def test_cpu_sdpa_keeps_kv_grouped_with_a_mask(monkeypatch):
    """Padded batches: K / V stay grouped (enable_gqa) instead of repeat_kv copies; fp32 logits / grads match."""
    import transformers
    from transformers.integrations import sdpa_attention
    from unsloth_zoo.cpu import ops
    original = sdpa_attention.use_gqa_in_sdpa
    # Another test may already have installed the patch (install_cpu_training_defaults): start from transformers' own.
    sdpa_attention.use_gqa_in_sdpa = getattr(original, "__wrapped__", original)
    config = transformers.LlamaConfig(hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                                      num_attention_heads = 8, num_key_value_heads = 2, vocab_size = 128,
                                      attn_implementation = "sdpa")
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    ids = torch.randint(0, 128, (2, 11), generator = torch.Generator().manual_seed(0))
    mask = torch.ones_like(ids)
    mask[1, 7:] = 0
    calls = []
    real_sdpa = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        calls.append((q.shape[1], k.shape[1], kwargs.get("enable_gqa", False)))
        return real_sdpa(q, k, v, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    results = []
    try:
        for patched in (False, True):
            if patched:
                assert ops.patch_cpu_sdpa_gqa() and ops.patch_cpu_sdpa_gqa()  # idempotent
            calls.clear()
            model.zero_grad()
            out = model(input_ids = ids, attention_mask = mask, use_cache = False).logits
            out.float().pow(2).mean().backward()
            results.append((out.detach(), model.model.layers[0].self_attn.k_proj.weight.grad.clone()))
            assert calls and all(c == ((8, 2, True) if patched else (8, 8, False)) for c in calls), calls
    finally:
        sdpa_attention.use_gqa_in_sdpa = original
    torch.testing.assert_close(results[0][0], results[1][0], rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(results[0][1], results[1][1], rtol = 1e-4, atol = 1e-6)


def test_failed_4bit_load_is_released_before_the_16bit_retry(monkeypatch):
    """The dense retry runs outside the except block, so the failed load's objects are already unreachable."""
    import gc
    import weakref
    import transformers
    from unsloth_zoo.cpu import loader
    refs, attempts = [], []

    class HalfBuilt:
        pass

    def fake_load(name, **kwargs):
        attempts.append("quantization_config" in kwargs)
        if "quantization_config" in kwargs:
            partial = HalfBuilt()
            refs.append(weakref.ref(partial))
            raise RuntimeError("bitsandbytes packing rejects this layer")
        gc.collect()
        attempts.append(refs[0]() is None)  # freed before the retry allocates
        raise _StopLoad
    class _StopLoad(Exception):
        pass
    monkeypatch.setattr(loader, "_adapter_base_model", lambda *a, **k: None)
    monkeypatch.setattr(loader, "cpu_bnb_4bit_supported", lambda: True)
    monkeypatch.setattr(loader, "available_ram", lambda: 0)
    monkeypatch.setattr(loader, "_checkpoint_quant_method", lambda *a, **k: None)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_load)
    with pytest.raises(_StopLoad):
        loader.FastCPUModel.from_pretrained("org/model", load_in_4bit = True)
    assert attempts == [True, False, True]


def _config_llama(attention):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size = 1000, hidden_size = 64, intermediate_size = 128, num_hidden_layers = 2,
                         num_attention_heads = 4, num_key_value_heads = 2, max_position_embeddings = 2048,
                         attn_implementation = attention)
    model = LlamaForCausalLM(config)
    model._unsloth_cpu_gradient_checkpointing = "unsloth"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    return model


@pytest.mark.parametrize("attention, expect_off", [("eager", False), ("sdpa", True)])
def test_auto_gradient_checkpointing_counts_eager_attention_scores(monkeypatch, attention, expect_off):
    """2 layers x 4 heads, batch 2 x 2048: eager keeps ~295 MB of scores / softmax for backward (the linear terms are
    ~74 MB), so 512 MB free (a 128 MB budget) must keep checkpointing for eager; SDPA keeps no scores and turns it off."""
    from unsloth_zoo.cpu import trainer
    monkeypatch.setattr(trainer, "available_ram", lambda: 512 << 20)
    monkeypatch.delenv("UNSLOTH_CPU_AUTO_GC", raising = False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising = False)
    model = _config_llama(attention)
    args = _GcArgs(length = 2048, batch = 2)
    assert bool(trainer._auto_cpu_gradient_checkpointing(model, args)) is expect_off
    assert args.gradient_checkpointing is (not expect_off)


def test_available_ram_honours_a_cgroup_v1_container_limit(monkeypatch, tmp_path):
    """A cgroup v1 container (Docker on RHEL 8 / Amazon Linux 2) sees the host's MemAvailable in /proc/meminfo: the
    16-bit and no-checkpointing decisions must use the container's own limit, or a model that fits the host OOMs."""
    import types
    import psutil
    import unsloth_zoo.hf_xet_tuning as xet
    from unsloth_zoo.cpu import runtime
    gib = 1024 ** 3
    (tmp_path / "memory.limit_in_bytes").write_text(str(8 * gib))
    (tmp_path / "memory.usage_in_bytes").write_text(str(1 * gib))
    monkeypatch.delenv("UNSLOTH_HOST_BUDGET_CGROUP", raising = False)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(total = 512 * gib, available = 256 * gib))
    monkeypatch.setattr(xet, "_cgroup_v2_dirs", lambda: [])
    monkeypatch.setattr(xet, "_cgroup_v1_dirs", lambda controller: [tmp_path] if controller == "memory" else [])
    real_open = open

    def fake_open(path, *args, **kwargs):  # the host view a v1 container gets
        if str(path) == "/proc/meminfo":
            import io
            return io.StringIO(f"MemTotal: {512 * gib // 1024} kB\nMemAvailable: {256 * gib // 1024} kB\n")
        if str(path) == "/proc/self/cgroup":
            import io
            return io.StringIO("4:memory:/docker/abc\n1:name=systemd:/docker/abc\n")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", fake_open)
    assert runtime.available_ram() <= 7 * gib

