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

"""Opt-in merged-weight GRPO rollouts (UNSLOTH_VLLM_MERGED_ROLLOUT), CPU only.

The HF base weights are row views of a fake fused vLLM qkv tensor, exactly the aliasing
fast_inference sets up, and the vLLM engine is a stub whose generate records what the
weights held while it ran. No vLLM engine is built.
"""

import copy
import inspect
import sys
import threading
import time
import types

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model

from unsloth_zoo import vllm_utils
from unsloth_zoo.vllm_utils import (
    _MergedLoRARequest,
    _MergedRolloutState,
    _merged_rollout_targets,
    install_merged_rollout_engine_wrapper,
    load_lora,
    merged_rollout,
    prepare_merged_rollout,
)

H, Q, KV, I = 64, 64, 32, 96
_ADAPTER_DIR = None
DTYPE = torch.bfloat16


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, Q, bias = False)
        self.k_proj = nn.Linear(H, KV, bias = False)
        self.v_proj = nn.Linear(H, KV, bias = False)
        self.o_proj = nn.Linear(Q, H, bias = False)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(I, H, bias = False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(50, H)
        self.layers = nn.ModuleList([_Layer(), _Layer()])


class _Tiny(nn.Module):
    def __init__(self, extra = None):
        super().__init__()
        self.config = types.SimpleNamespace(model_type = "tiny", tie_word_embeddings = False)
        self.model = _Inner()
        self.lm_head = nn.Linear(H, 50, bias = False)
        if extra == "visual":
            self.visual = nn.Module()
            self.visual.proj = nn.Linear(H, H, bias = False)
        if extra == "experts":
            self.model.layers[0].mlp.experts = nn.ModuleList([nn.Linear(H, I, bias = False)])

    def forward(self, x):
        return x


class _FakeVllmModel(nn.Module):
    """vLLM's own parameters: fused qkv, o_proj and down_proj per layer."""
    def __init__(self, n_layers):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.qkv = nn.ParameterList([
            nn.Parameter((torch.randn(Q + 2 * KV, H, generator = g) * 0.02).to(DTYPE), requires_grad = False)
            for _ in range(n_layers)
        ])
        self.o = nn.ParameterList([
            nn.Parameter((torch.randn(H, Q, generator = g) * 0.02).to(DTYPE), requires_grad = False)
            for _ in range(n_layers)
        ])
        self.down = nn.ParameterList([
            nn.Parameter((torch.randn(H, I, generator = g) * 0.02).to(DTYPE), requires_grad = False)
            for _ in range(n_layers)
        ])


class _FakeLLM:
    """Stands in for vllm.LLM: generate / chat record the live weights they would read."""
    def __init__(self, vllm_model, fail = False):
        runner = types.SimpleNamespace(model = vllm_model)
        worker = types.SimpleNamespace(model_runner = runner)
        self.llm_engine = types.SimpleNamespace(model_executor = types.SimpleNamespace(driver_worker = worker))
        self.vllm_model = vllm_model
        self.fail = fail
        self.calls = []
        self.resets = []

    def reset_prefix_cache(self):
        # Record whether the weights were folded when the cache was dropped.
        self.resets.append([p.detach().clone() for p in self.vllm_model.parameters()])
        return True

    def generate(self, prompts, sampling_params = None, *, use_tqdm = True, lora_request = None):
        self.calls.append(("generate", lora_request, [p.detach().clone() for p in self.vllm_model.parameters()]))
        if self.fail: raise RuntimeError("engine died mid-generate")
        return ["out"]

    def chat(self, messages, sampling_params = None, use_tqdm = True, lora_request = None):
        self.calls.append(("chat", lora_request, [p.detach().clone() for p in self.vllm_model.parameters()]))
        return self.generate(messages, sampling_params, lora_request = lora_request)


def _make_model(
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "down_proj"),
    extra = None, alias = True, wrap = True, base_dtype = DTYPE, adapter_name = "default",
    fail = False, randomize_B = True, **lora_kwargs,
):
    torch.manual_seed(0)
    base = _Tiny(extra = extra)
    n_layers = len(base.model.layers)
    vllm_model = _FakeVllmModel(n_layers)
    for i, layer in enumerate(base.model.layers):
        a = layer.self_attn
        if alias:
            qkv = vllm_model.qkv[i]
            a.q_proj.weight = nn.Parameter(qkv[:Q], requires_grad = False)
            a.k_proj.weight = nn.Parameter(qkv[Q:Q + KV], requires_grad = False)
            a.v_proj.weight = nn.Parameter(qkv[Q + KV:], requires_grad = False)
            a.o_proj.weight = nn.Parameter(vllm_model.o[i], requires_grad = False)
            layer.mlp.down_proj.weight = nn.Parameter(vllm_model.down[i], requires_grad = False)
        else:
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, layer.mlp.down_proj):
                lin.weight = nn.Parameter(lin.weight.detach().to(base_dtype).clone(), requires_grad = False)
    for module in base.modules():
        if isinstance(module, nn.Linear) and module.weight.dtype != base_dtype:
            module.weight = nn.Parameter(module.weight.detach().to(base_dtype), requires_grad = False)

    config = LoraConfig(r = 8, lora_alpha = 16, target_modules = list(target_modules), lora_dropout = 0.0, **lora_kwargs)
    model = get_peft_model(base, config, adapter_name = adapter_name)
    if randomize_B:
        g = torch.Generator().manual_seed(1)
        for module in model.modules():
            lora_B = getattr(module, "lora_B", None)
            if isinstance(lora_B, nn.ModuleDict):
                for lin in lora_B.values():
                    lin.weight.data = torch.randn(lin.weight.shape, generator = g) * 0.05
    llm = _FakeLLM(vllm_model, fail = fail)
    if wrap: install_merged_rollout_engine_wrapper(llm)
    model.vllm_engine = llm
    return model, llm, vllm_model


@pytest.fixture(autouse = True)
def _isolate(monkeypatch, tmp_path):
    # load_lora writes the adapter config on its first call: keep it out of the CWD.
    monkeypatch.setattr(sys.modules[__name__], "_ADAPTER_DIR", str(tmp_path / "adapter"))
    monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", "1")
    monkeypatch.setattr(vllm_utils, "_MERGED_ROLLOUT_LOGGED", set())
    monkeypatch.setattr(vllm_utils, "LORA_REQUEST_ID", None, raising = False)

    class LoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path = "", lora_tensors = None, lora_config = None):
            self.lora_name, self.lora_int_id, self.lora_path = lora_name, lora_int_id, lora_path
            self.lora_tensors, self.lora_config = lora_tensors, lora_config

    fake = types.ModuleType("vllm.lora.request")
    fake.LoRARequest = LoRARequest
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake)
    monkeypatch.setattr(vllm_utils, "get_peft_config", lambda d: {"r": 8})
    monkeypatch.setenv("UNSLOTH_DISABLE_LORA_NAME_CHECK", "1")
    yield fake


def _lora_modules(model):
    return [m for m in model.modules() if isinstance(getattr(m, "lora_A", None), nn.ModuleDict) and len(m.lora_A)]


def _snapshot(vllm_model):
    return [p.detach().clone() for p in vllm_model.parameters()]


def _assert_bitwise(before, vllm_model):
    for a, b in zip(before, vllm_model.parameters()):
        assert torch.equal(a.view(torch.int16), b.detach().view(torch.int16))


def _perturb_adapter(model, step):
    g = torch.Generator().manual_seed(100 + step)
    for m in _lora_modules(model):
        for d in (m.lora_A, m.lora_B):
            w = d["default"].weight
            w.data += torch.randn(w.shape, generator = g) * 0.02


def _naive_cycle(model):
    """PEFT merge/unmerge style: W += sBA, then W -= sBA, both in bf16."""
    with torch.no_grad():
        for m in _lora_modules(model):
            W = m.base_layer.weight
            delta = (m.lora_B["default"].weight @ m.lora_A["default"].weight * m.scaling["default"]).to(W.dtype)
            W += delta
            W -= delta


def _safe_cycle(model):
    state = prepare_merged_rollout(model)
    assert state is not None
    with merged_rollout(state):
        pass


def _run_cycles(model, vllm_model, cycle, n = 100):
    before = _snapshot(vllm_model)
    for step in range(n):
        _perturb_adapter(model, step)
        cycle(model)
    _assert_bitwise(before, vllm_model)


# 1. Restore is bit-exact over many cycles; the naive fold control must fail the same check.
def test_fold_restore_is_bit_exact_over_100_cycles():
    model, llm, vllm_model = _make_model()
    _run_cycles(model, vllm_model, _safe_cycle)


def test_naive_inplace_fold_control_drifts():
    model, llm, vllm_model = _make_model()
    with pytest.raises(AssertionError):
        _run_cycles(model, vllm_model, _naive_cycle)


# 2. The folded value is one rounding of P + s * B @ A, with rsLoRA scaling read off PEFT.
def test_folded_value_matches_single_rounding_with_rslora():
    model, llm, vllm_model = _make_model(use_rslora = True)
    module = _lora_modules(model)[0]
    s = module.scaling["default"]
    assert s == pytest.approx(16 / 8 ** 0.5)
    state = prepare_merged_rollout(model)
    P = module.base_layer.weight.detach().clone()
    A = module.lora_A["default"].weight.detach().to(DTYPE).float()
    B = module.lora_B["default"].weight.detach().to(DTYPE).float()
    expected = (P.float() + s * (B @ A)).to(DTYPE)
    with merged_rollout(state):
        got = module.base_layer.weight.detach().clone()
    ulp = (expected.float().abs() * 2 ** -7).clamp_min(2 ** -133)
    assert ((got.float() - expected.float()).abs() <= ulp).all()
    assert not torch.equal(got, P)
    assert torch.equal(module.base_layer.weight, P)


# 3. Folding q only leaves the k / v rows of the fused vLLM tensor untouched.
def test_q_fold_leaves_fused_kv_rows_untouched():
    model, llm, vllm_model = _make_model(target_modules = ("q_proj",))
    before = [t.clone() for t in vllm_model.qkv]
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    assert isinstance(request, _MergedLoRARequest)
    llm.generate(["hi"], lora_request = request)
    seen = llm.calls[-1][2][:len(before)]
    for b, s in zip(before, seen):
        assert torch.equal(s[Q:], b[Q:])
        assert not torch.equal(s[:Q], b[:Q])
    for b, t in zip(before, vllm_model.qkv):
        assert torch.equal(t, b)


# 4. Eligibility matrix: every case returns a reason and load_lora returns a real request.
def _ineligible_cases():
    def bnb(m): m.base_model.model.is_loaded_in_4bit = True
    def fp8(m): m.base_model.model.config.quantization_config = {"quant_method": "fp8"}
    def fan(m): _lora_modules(m)[0].fan_in_fan_out = True
    def merged(m): _lora_modules(m)[0].merged_adapters = ["default"]
    def second(m): m.add_adapter("other", LoraConfig(r = 4, target_modules = ["q_proj"]))
    return {
        "bitsandbytes": (dict(), bnb, "is quantized"),
        "fp8_quant_config": (dict(), fp8, "is quantized"),
        "moe_experts": (dict(extra = "experts", target_modules = ("q_proj", "experts.0")), None, "is a MoE expert LoRA"),
        "vision_tower": (dict(extra = "visual", target_modules = ("q_proj", "proj")), None, "vision / audio tower"),
        "embed_tokens": (dict(target_modules = ("q_proj", "embed_tokens")), None, "has an embedding LoRA"),
        "lm_head": (dict(target_modules = ("q_proj", "lm_head")), None, "is an embedding or lm_head"),
        "modules_to_save": (dict(modules_to_save = ["lm_head"]), None, "modules_to_save"),
        "dora": (dict(use_dora = True), None, "uses DoRA"),
        "lora_bias": (dict(lora_bias = True), None, "uses lora_bias"),
        "fan_in_fan_out": (dict(), fan, "uses fan_in_fan_out"),
        "already_merged": (dict(), merged, "already has a PEFT-merged adapter"),
        "two_adapters": (dict(), second, "the adapters are ['default', 'other']"),
        "non_default_adapter": (dict(adapter_name = "other"), None, "the adapters are ['other']"),
        "fp32_base": (dict(alias = False, base_dtype = torch.float32), None, "not 16-bit"),
        "not_aliased": (dict(alias = False), None, "does not share storage with the vLLM engine"),
        "adapters_disabled": (dict(), lambda m: setattr(_lora_modules(m)[0], "_disable_adapters", True), "has its adapters disabled"),
        "trainable_base": (dict(), lambda m: _lora_modules(m)[0].base_layer.weight.requires_grad_(True), "base weight is trainable"),
        "tensor_scaling": (dict(), lambda m: _lora_modules(m)[0].scaling.__setitem__("default", torch.tensor(2.0)), "non-scalar LoRA scaling"),
        "custom_lora_A": (dict(), lambda m: _lora_modules(m)[0].lora_A.__setitem__("default", type("L", (nn.Linear,), {})(H, 8, bias = False)), "not a plain Linear LoRA"),
        "trainable_tokens": (dict(), lambda m: setattr(m.peft_config["default"], "trainable_token_indices", [1]), "trainable_token_indices"),
        "target_parameters": (dict(), lambda m: setattr(m.peft_config["default"], "target_parameters", ["x"]), "target_parameters"),
        "engine_not_wrapped": (dict(wrap = False), None, "not wrapped"),
        "engine_unreachable": (dict(), lambda m: setattr(m, "vllm_engine", types.SimpleNamespace(_unsloth_merged_rollout_wrapped = True)), "could not be inspected"),
    }


def _load_lora_outcome(model):
    try:
        request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    except Exception as error:
        return ("raises", type(error), str(error)), None
    tensors = getattr(request, "lora_tensors", None)
    return ("returns", type(request), sorted(tensors) if tensors is not None else None), request


@pytest.mark.parametrize("case", sorted(_ineligible_cases()))
def test_ineligible_falls_back_to_todays_path(case, capsys, monkeypatch):
    kwargs, mutate, needle = _ineligible_cases()[case]
    model, llm, vllm_model = _make_model(**kwargs)
    if mutate is not None: mutate(model)
    before = _snapshot(vllm_model)

    monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", "0")
    today, _ = _load_lora_outcome(model)
    monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", "1")
    capsys.readouterr()
    outcome, request = _load_lora_outcome(model)
    assert outcome == today
    assert outcome[1] is not _MergedLoRARequest
    out = capsys.readouterr().out
    assert "falling back to vLLM LoRA (punica)" in out and needle in out, out
    assert prepare_merged_rollout(model) is None
    assert "falling back" not in capsys.readouterr().out   # logged once
    if request is not None:
        assert request.lora_tensors and all(".lora_" in k for k in request.lora_tensors)
        llm.generate(["hi"], lora_request = request)
        assert llm.calls[-1][1] is request
    _assert_bitwise(before, vllm_model)


def test_eligible_baseline_has_no_reason():
    model, llm, vllm_model = _make_model()
    assert _merged_rollout_targets(model)[1] is None
    assert isinstance(load_lora(model, _ADAPTER_DIR, load_tensors = True), _MergedLoRARequest)


def test_only_1_turns_it_on(monkeypatch):
    for value, on in (("1", True), ("0", False), ("", False), ("true", False), ("auto", False)):
        monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", value)
        assert vllm_utils._merged_rollout_enabled() is on


# 5. An exception inside generate still restores the weights.
def test_exception_during_generate_restores():
    model, llm, vllm_model = _make_model(fail = True)
    before = _snapshot(vllm_model)
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    with pytest.raises(RuntimeError, match = "mid-generate"):
        llm.generate(["hi"], lora_request = request)
    seen = llm.calls[-1][2]
    assert any(not torch.equal(a, b) for a, b in zip(before, seen))
    _assert_bitwise(before, vllm_model)
    assert model._unsloth_merged_rollout_state.depth == 0


# 6. The wrapper: sentinel -> lora_request = None inside a fold; real requests pass through.
def test_wrapper_folds_for_sentinel_and_passes_real_requests_through():
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    llm.generate(["hi"], lora_request = request)
    kind, seen_request, seen = llm.calls[-1]
    assert seen_request is None
    assert any(not torch.equal(a, b) for a, b in zip(before, seen))
    _assert_bitwise(before, vllm_model)

    real = object()
    llm.generate(["hi"], lora_request = real)
    assert llm.calls[-1][1] is real
    assert all(torch.equal(a, b) for a, b in zip(before, llm.calls[-1][2]))

    # chat: positional sentinel, and the nested generate must not restore early
    llm.calls.clear()
    llm.chat(["hi"], None, True, request)
    (_, chat_req, chat_seen), (_, gen_req, gen_seen) = llm.calls
    assert chat_req is None and gen_req is None
    assert all(torch.equal(a, b) for a, b in zip(chat_seen, gen_seen))
    assert any(not torch.equal(a, b) for a, b in zip(before, gen_seen))
    _assert_bitwise(before, vllm_model)


def test_prefix_cache_is_reset_before_the_fold_and_after_restore():
    """Merged generate hashes its KV as base blocks: stale prefixes must not survive."""
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    llm.chat(["hi"], None, True, request)
    assert len(llm.resets) == 2
    folded = llm.calls[-1][2]
    assert any(not torch.equal(a, b) for a, b in zip(before, folded))
    # Reset 0 runs on the base just before the fold, reset 1 after the restore.
    assert all(torch.equal(a, b) for a, b in zip(before, llm.resets[0]))
    assert all(torch.equal(a, b) for a, b in zip(before, llm.resets[1]))
    llm.generate(["hi"], lora_request = object())
    assert len(llm.resets) == 2


def test_wrapper_keeps_the_real_signature_for_unsloth_binding_check():
    model, llm, vllm_model = _make_model()
    sig = inspect.signature(llm.chat)
    assert "lora_request" in sig.bind_partial(["m"], None, True, "x").arguments
    install_merged_rollout_engine_wrapper(llm)
    assert not hasattr(llm.generate.__wrapped__, "__wrapped__")


def test_kill_switch_off_returns_a_real_lora_request(monkeypatch, _isolate):
    monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", "0")
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    r1 = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    r2 = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    assert type(r1) is _isolate.LoRARequest and type(r2) is _isolate.LoRARequest
    assert (r1.lora_int_id, r2.lora_int_id) == (1, 2)
    expected = {k.replace(".default", "") for k in model.state_dict() if ".lora_A." in k or ".lora_B." in k}
    assert set(r1.lora_tensors) == expected
    assert getattr(model, "_unsloth_merged_rollout_state", None) is None
    _assert_bitwise(before, vllm_model)


def test_kill_switch_defaults_off(monkeypatch, _isolate):
    monkeypatch.delenv("UNSLOTH_VLLM_MERGED_ROLLOUT", raising = False)
    assert not vllm_utils._merged_rollout_enabled()
    model, llm, vllm_model = _make_model()
    assert type(load_lora(model, _ADAPTER_DIR, load_tensors = True)) is _isolate.LoRARequest


def test_nested_contexts_restore_only_at_the_outermost_exit():
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    state = prepare_merged_rollout(model)
    with merged_rollout(state):
        folded = _snapshot(vllm_model)
        with merged_rollout(state):
            pass
        assert all(torch.equal(a, b) for a, b in zip(folded, vllm_model.parameters()))
    _assert_bitwise(before, vllm_model)


def test_state_is_cached_and_rebuilt_when_adapters_change():
    model, llm, vllm_model = _make_model()
    s1 = prepare_merged_rollout(model)
    assert prepare_merged_rollout(model) is s1
    model.add_adapter("other", LoraConfig(r = 4, target_modules = ["q_proj"]))
    assert prepare_merged_rollout(model) is None


# 7. Compatibility refusals (PEFT variants, shared base weights, MLA, multimodal, vLLM 0.11).
def test_compat_lora_variant_refused():
    # aLoRA is a PEFT lora_variant (0.18+): the adapter is only live after the invocation tokens.
    model, llm, _ = _make_model(target_modules = ("q_proj",), alora_invocation_tokens = [1, 2])
    reason = _merged_rollout_targets(model)[1]
    assert reason is not None and "variant" in reason, reason


def test_compat_shared_base_weight_refused():
    model, llm, vm = _make_model(target_modules = ("q_proj", "o_proj"))
    layers = model.base_model.model.model.layers
    # layer_replication style: layer 1's o_proj base is layer 0's.
    layers[1].self_attn.o_proj.base_layer.weight = layers[0].self_attn.o_proj.base_layer.weight
    reason = _merged_rollout_targets(model)[1]
    assert reason is not None and "shares its base weight" in reason, reason


def test_compat_kv_b_proj_refused():
    model, llm, vm = _make_model(target_modules = ("q_proj",))
    attn = model.base_model.model.model.layers[0].self_attn
    attn.kv_b_proj = attn.q_proj
    del attn.q_proj
    reason = _merged_rollout_targets(model)[1]
    assert reason is not None and "kv_b_proj" in reason, reason


def test_compat_storage_limited_to_language_model():
    model, llm, vm = _make_model(target_modules = ("q_proj",))
    # A multimodal vLLM model whose language model holds none of the HF storages.
    vm.get_language_model = lambda: nn.Linear(2, 2)
    reason = _merged_rollout_targets(model)[1]
    assert reason is not None and "does not share storage" in reason, reason
    vm.get_language_model = lambda: vm
    assert _merged_rollout_targets(model)[1] is None


def test_compat_state_invalidated_when_base_becomes_trainable():
    model, llm, vm = _make_model(target_modules = ("q_proj",))
    state = prepare_merged_rollout(model)
    assert state is not None
    _lora_modules(model)[0].base_layer.weight.requires_grad_(True)
    assert not state.is_valid_for(model)


def test_compat_reset_none_falls_back_to_scheduler():
    calls = []
    class Sched:
        def __init__(self, result): self.result = result
        def reset_prefix_cache(self):
            calls.append(1); return self.result
    class OldLLM:  # vLLM 0.11.x: LLM.reset_prefix_cache returns None
        def __init__(self, result):
            core = types.SimpleNamespace(engine_core = types.SimpleNamespace(scheduler = Sched(result)))
            self.llm_engine = types.SimpleNamespace(engine_core = core)
        def reset_prefix_cache(self): return None
    assert vllm_utils._reset_merged_rollout_prefix_cache(OldLLM(False)) is False
    assert vllm_utils._reset_merged_rollout_prefix_cache(OldLLM(True)) is True
    assert calls == [1, 1]


# 8. Prefix-cache reset failures: before the fold -> this call takes the LoRA path, no fold.
def _failing_reset(llm, how, when):
    real, n = llm.reset_prefix_cache, {"n": 0}
    def reset():
        n["n"] += 1
        if n["n"] in when:
            if how == "raise": raise RuntimeError("engine busy")
            return False
        return real()
    llm.reset_prefix_cache = reset
    return n


@pytest.mark.parametrize("how", ["false", "raise"])
def test_reset_failure_before_fold_uses_a_real_lora_request(how, capsys, _isolate):
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    assert isinstance(request, _MergedLoRARequest)
    _failing_reset(llm, how, when = {1, 4})   # call 2 resets twice (before + after)
    capsys.readouterr()
    for _ in range(2):
        llm.generate(["hi"], lora_request = request)
        kind, seen_request, seen = llm.calls[-1]
        assert type(seen_request) is _isolate.LoRARequest
        expected = {k.replace(".default", "") for k in model.state_dict() if ".lora_A." in k or ".lora_B." in k}
        assert set(seen_request.lora_tensors) == expected
        assert all(torch.equal(a, b) for a, b in zip(before, seen))   # never folded
        _assert_bitwise(before, vllm_model)
        # Recovered on the next call: reset works again, so it folds.
        llm.generate(["hi"], lora_request = request)
        assert llm.calls[-1][1] is None
        assert any(not torch.equal(a, b) for a, b in zip(before, llm.calls[-1][2]))
    # Fresh LoRA ids each fallback, like load_lora today.
    ids = [c[1].lora_int_id for c in llm.calls if c[1] is not None]
    assert len(set(ids)) == len(ids) == 2
    assert capsys.readouterr().out.count("did not reset its prefix cache before the fold") == 1
    _assert_bitwise(before, vllm_model)


@pytest.mark.parametrize("how", ["false", "raise"])
def test_reset_failure_after_restore_logs_once_and_next_call_resets(how, capsys):
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    n = _failing_reset(llm, how, when = {2, 4})
    capsys.readouterr()
    llm.generate(["hi"], lora_request = request)
    _assert_bitwise(before, vllm_model)
    llm.generate(["hi"], lora_request = request)
    assert n["n"] == 4   # the second call reset again before folding
    assert all(c[1] is None for c in llm.calls)
    assert any(not torch.equal(a, b) for a, b in zip(before, llm.calls[-1][2]))
    _assert_bitwise(before, vllm_model)
    assert capsys.readouterr().out.count("after a merged rollout") == 1


def test_missing_reset_takes_the_lora_path(_isolate):
    model, llm, vllm_model = _make_model()
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    llm.reset_prefix_cache = None
    llm.generate(["hi"], lora_request = request)
    assert type(llm.calls[-1][1]) is _isolate.LoRARequest


# 9. Ineligible verdicts are cached and invalidated by the same checks as the fold state.
def test_ineligible_verdict_is_cached_until_a_layer_changes(monkeypatch):
    model, llm, vllm_model = _make_model()
    module = _lora_modules(model)[0]
    module.merged_adapters = ["default"]
    calls = {"n": 0}
    real = vllm_utils._merged_rollout_targets
    def counting(m):
        calls["n"] += 1
        return real(m)
    monkeypatch.setattr(vllm_utils, "_merged_rollout_targets", counting)
    for _ in range(5): assert prepare_merged_rollout(model) is None
    assert calls["n"] == 1
    module.merged_adapters = []
    assert isinstance(prepare_merged_rollout(model), _MergedRolloutState)
    assert calls["n"] == 2
    for _ in range(3): prepare_merged_rollout(model)
    assert calls["n"] == 2   # eligible state is cached too
    module.base_layer.weight.requires_grad_(True)
    assert prepare_merged_rollout(model) is None
    assert calls["n"] == 3


@pytest.mark.parametrize("change", ["data_ptr", "dtype", "shape", "stride", "version", "requires_grad", "disabled", "merged"])
def test_state_invalidated_by_each_layer_check(change):
    model, llm, vllm_model = _make_model()
    state = prepare_merged_rollout(model)
    assert state.is_valid_for(model)
    module = _lora_modules(model)[0]
    W = module.base_layer.weight
    if change == "data_ptr": W.data = W.data.clone()
    if change == "dtype": W.data = W.data.view(torch.float16)   # same bytes and address
    if change == "shape": W.data = W.data[:-1]
    if change == "stride": W.data = W.data.t()   # square q_proj: same address and shape
    if change == "version":
        with torch.no_grad(): W.add_(0)
    if change == "requires_grad": W.requires_grad_(True)
    if change == "disabled": module._disable_adapters = True
    if change == "merged": module.merged_adapters = ["default"]
    assert not state.is_valid_for(model)


def test_state_stays_valid_across_rollouts():
    # The restore itself bumps _version: it is re-recorded, so P is not rebuilt every step.
    model, llm, vllm_model = _make_model()
    request = _req(model)
    for _ in range(3):
        llm.generate(["hi"], lora_request = request)
        assert request.state.is_valid_for(model)
        assert prepare_merged_rollout(model) is request.state


def test_non_linear_base_layer_refused():
    model, llm, vllm_model = _make_model(target_modules = ("q_proj",))
    _lora_modules(model)[0].base_layer.__class__ = type("NotLinear", (nn.Module,), {})
    reason = _merged_rollout_targets(model)[1]
    assert reason is not None and "is not a dense Linear" in reason, reason


def test_direct_merged_rollout_is_locked():
    model, llm, vllm_model = _make_model()
    state = prepare_merged_rollout(model)
    inside, release, entered = threading.Event(), threading.Event(), []
    def outer():
        with merged_rollout(state):
            inside.set(); release.wait(5)
    def other():
        with merged_rollout(state):
            entered.append(state.depth)
    ta = threading.Thread(target = outer); ta.start(); inside.wait(5)
    tb = threading.Thread(target = other); tb.start(); time.sleep(0.3)
    assert entered == []   # waits for the first thread's restore
    release.set(); ta.join(); tb.join()
    assert entered == [1] and state.depth == 0


# 10. Safety: lock, restore retry, unusable state, stale sentinels.
def _req(model):
    return load_lora(model, _ADAPTER_DIR, load_tensors = True)


def _folded(before, seen):
    return any(not torch.equal(a, b) for a, b in zip(before, seen))


def test_concurrent_generate_serialises():
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    release_a, in_a, release_b = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    class SlowLLM(_FakeLLM):
        def generate(self, prompts, sampling_params = None, *, use_tqdm = True, lora_request = None):
            if prompts[0] == "A": in_a.set(); release_a.wait(5)
            if prompts[0] == "B": release_b.wait(5)
            seen[prompts[0]] = _snapshot(vllm_model)
            return ["out"]
    llm = SlowLLM(vllm_model)
    install_merged_rollout_engine_wrapper(llm)
    model.vllm_engine = llm
    req = _req(model)
    ta = threading.Thread(target = lambda: llm.generate(["A"], lora_request = req)); ta.start(); in_a.wait(5)
    tb = threading.Thread(target = lambda: llm.generate(["B"], lora_request = req)); tb.start(); time.sleep(0.3)
    assert "B" not in seen
    release_a.set(); ta.join(); release_b.set(); tb.join()
    assert _folded(before, seen["A"]) and _folded(before, seen["B"])
    assert req.state.depth == 0
    # Each thread ran as its own outermost fold: 2 resets each.
    assert len(llm.resets) == 4
    _assert_bitwise(before, vllm_model)


def test_interrupted_restore_is_redone(monkeypatch):
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    req = _req(model)
    real_im, n = torch.inference_mode, {"n": 0}
    def im(*a, **k):
        n["n"] += 1
        if n["n"] == 2: raise KeyboardInterrupt()
        return real_im(*a, **k)
    monkeypatch.setattr(vllm_utils.torch, "inference_mode", im)
    with pytest.raises(KeyboardInterrupt):
        llm.generate(["hi"], lora_request = req)
    monkeypatch.setattr(vllm_utils.torch, "inference_mode", real_im)
    _assert_bitwise(before, vllm_model)
    assert req.state.depth == 0 and req.state.poisoned is None
    llm.generate(["hi"], lora_request = _req(model))
    assert _folded(before, llm.calls[-1][2])


def test_persistent_restore_failure_is_loud(monkeypatch):
    model, llm, vllm_model = _make_model()
    req = _req(model)
    real_im, n = torch.inference_mode, {"n": 0}
    def im(*a, **k):
        n["n"] += 1
        if n["n"] >= 2: raise RuntimeError("CUDA error: an illegal memory access")
        return real_im(*a, **k)
    monkeypatch.setattr(vllm_utils.torch, "inference_mode", im)
    with pytest.raises(RuntimeError, match = "could not restore"):
        llm.generate(["hi"], lora_request = req)
    assert n["n"] == 3   # fold, restore, one retry
    monkeypatch.setattr(vllm_utils.torch, "inference_mode", real_im)
    assert req.state.depth == 0
    with pytest.raises(RuntimeError, match = "could not restore"):
        _req(model)
    with pytest.raises(RuntimeError, match = "unusable"):
        llm.generate(["hi"], lora_request = req)


def test_nested_depth_counts_down_and_only_outermost_restores():
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    state = prepare_merged_rollout(model)
    with merged_rollout(state):
        with merged_rollout(state):
            with merged_rollout(state):
                assert state.depth == 3
            assert state.depth == 2
        assert state.depth == 1
        assert _folded(before, _snapshot(vllm_model))
    assert state.depth == 0
    _assert_bitwise(before, vllm_model)


def test_merge_adapter_detected():
    model, llm, vllm_model = _make_model()
    _req(model)
    class _Cfg(dict):
        def __getattr__(self, k):
            if k in self: return self[k]
            raise AttributeError(k)
    model.base_model.model.config = _Cfg(model_type = "tiny", tie_word_embeddings = False)
    model.merge_adapter()
    assert prepare_merged_rollout(model) is None


def test_inplace_base_reload_rebuilds_P():
    model, llm, vllm_model = _make_model()
    req = _req(model)
    with torch.no_grad():
        for p in vllm_model.parameters(): p.add_(1.0)
    new_base = _snapshot(vllm_model)
    # The held sentinel is re-validated at generate time: no fold from the stale P.
    llm.generate(["hi"], lora_request = req)
    assert llm.calls[-1][1] is None
    _assert_bitwise(new_base, vllm_model)
    assert model._unsloth_merged_rollout_state is not req.state


def test_stale_sentinel_uses_the_lora_path_when_no_longer_eligible(_isolate):
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    req = _req(model)
    _lora_modules(model)[0].base_layer.weight.requires_grad_(True)
    llm.generate(["hi"], lora_request = req)
    assert type(llm.calls[-1][1]) is _isolate.LoRARequest
    assert all(torch.equal(a, b) for a, b in zip(before, llm.calls[-1][2]))
    _assert_bitwise(before, vllm_model)


def test_synchronize_runs_after_fold_and_before_restore(monkeypatch):
    model, llm, vllm_model = _make_model()
    events = []
    monkeypatch.setattr(vllm_utils, "_merged_rollout_synchronize", lambda state: events.append("sync"))
    state = prepare_merged_rollout(model)
    with merged_rollout(state):
        events.append("generate")
    # The last sync lands the restore before the lock is released.
    assert events == ["sync", "generate", "sync", "sync"]


def test_sentinel_keeps_the_probe_marker():
    model, llm, vllm_model = _make_model()
    assert getattr(_req(model), "_unsloth_merged", False) is True


# 11. Kill switch unset: load_lora and load_vllm behave exactly as origin/main.
def test_kill_switch_unset_matches_build_lora_request(monkeypatch, _isolate):
    monkeypatch.delenv("UNSLOTH_VLLM_MERGED_ROLLOUT", raising = False)
    model, llm, vllm_model = _make_model(wrap = False)
    real_generate = llm.generate
    r1 = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    r2 = load_lora(model, _ADAPTER_DIR)
    assert (type(r1), type(r2)) == (_isolate.LoRARequest, _isolate.LoRARequest)
    assert (r1.lora_int_id, r2.lora_int_id, vllm_utils.LORA_REQUEST_ID) == (1, 2, 3)
    assert r2.lora_path == _ADAPTER_DIR
    assert getattr(model, "_unsloth_merged_rollout_state", None) is None
    assert not getattr(llm, "_unsloth_merged_rollout_wrapped", False)
    assert llm.generate == real_generate


def test_load_vllm_only_wraps_when_enabled():
    import ast
    source = inspect.getsource(vllm_utils.load_vllm)
    guards = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If) and any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "install_merged_rollout_engine_wrapper"
            for n in ast.walk(node)
        )
    ]
    assert any(
        isinstance(n.test, ast.Call) and getattr(n.test.func, "id", None) == "_merged_rollout_enabled" and
        len(n.body) == 1 and getattr(n.body[0].value.func, "id", None) == "install_merged_rollout_engine_wrapper"
        for n in guards
    )
    assert source.count("install_merged_rollout_engine_wrapper") == 1
    assert source.count("_merged_rollout") == 2


# 12. Round 2: untracked reloads, per-prompt lists, one lock, current modules, engine, dirty cache.
def _expected_fold(module, base):
    A = module.lora_A["default"].weight.to(DTYPE)
    B = module.lora_B["default"].weight.to(DTYPE)
    return torch.addmm(base, B, A, alpha = module.scaling["default"])


def test_untracked_base_reload_refreshes_P(capsys):
    model, llm, vllm_model = _make_model(target_modules = ("q_proj",))
    req = _req(model)
    module = _lora_modules(model)[0]
    # vLLM load_weights / .data writes do not bump the view's _version.
    with torch.no_grad():
        for p in vllm_model.parameters(): p.data.add_(1.0)
    assert req.state.is_valid_for(model)
    new_base = _snapshot(vllm_model)
    new_q = module.base_layer.weight.detach().clone()
    llm.generate(["hi"], lora_request = req)
    assert llm.calls[-1][1] is None
    folded_q = llm.calls[-1][2][0][:Q]
    assert torch.equal(folded_q, _expected_fold(module, new_q))
    _assert_bitwise(new_base, vllm_model)
    assert "refreshed the merged-rollout pristine copy" in capsys.readouterr().out


def test_refresh_check_is_a_no_op_when_unchanged(monkeypatch):
    model, llm, vllm_model = _make_model()
    state = prepare_merged_rollout(model)
    copies = []
    real = vllm_utils._merged_rollout_refresh_pristine
    monkeypatch.setattr(vllm_utils, "_merged_rollout_refresh_pristine", lambda s: copies.append(real(s)))
    for _ in range(3):
        with merged_rollout(state): pass
    assert copies == [False, False, False]


def test_per_prompt_list_of_one_state_folds_once(_isolate):
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    req = _req(model)
    llm.generate(["a", "b"], lora_request = [req, _req(model)])
    kind, seen_request, seen = llm.calls[-1]
    assert seen_request is None and _folded(before, seen)
    assert len(llm.resets) == 2
    _assert_bitwise(before, vllm_model)


@pytest.mark.parametrize("container", [list, tuple])
def test_per_prompt_mixed_list_converts_sentinels(container, _isolate):
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    req = _req(model)
    other = _isolate.LoRARequest("x", 99)
    llm.generate(["a", "b", "c"], lora_request = container([req, None, other]))
    kind, seen_request, seen = llm.calls[-1]
    assert type(seen_request) is container
    assert type(seen_request[0]) is _isolate.LoRARequest and seen_request[0].lora_tensors
    assert seen_request[1] is None and seen_request[2] is other
    assert not any(isinstance(x, _MergedLoRARequest) for x in seen_request)
    assert not _folded(before, seen)
    _assert_bitwise(before, vllm_model)


def test_plain_call_waits_for_a_fold_on_another_thread():
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    in_a, release_a, seen = threading.Event(), threading.Event(), {}

    class SlowLLM(_FakeLLM):
        def generate(self, prompts, sampling_params = None, *, use_tqdm = True, lora_request = None):
            if prompts[0] == "A": in_a.set(); release_a.wait(5)
            seen[prompts[0]] = _snapshot(vllm_model)
            return ["out"]
    llm = SlowLLM(vllm_model)
    install_merged_rollout_engine_wrapper(llm)
    model.vllm_engine = llm
    req = _req(model)
    ta = threading.Thread(target = lambda: llm.generate(["A"], lora_request = req)); ta.start(); in_a.wait(5)
    tb = threading.Thread(target = lambda: llm.generate(["B"], lora_request = None)); tb.start(); time.sleep(0.3)
    assert "B" not in seen
    release_a.set(); ta.join(); tb.join()
    assert _folded(before, seen["A"])
    assert not _folded(before, seen["B"])


def test_prepare_never_snapshots_folded_weights():
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    state = prepare_merged_rollout(model)
    inside, release, built = threading.Event(), threading.Event(), {}
    def outer():
        with merged_rollout(state):
            inside.set(); release.wait(5)
    def other():
        model._unsloth_merged_rollout_state = None
        built["state"] = prepare_merged_rollout(model)
    ta = threading.Thread(target = outer); ta.start(); inside.wait(5)
    tb = threading.Thread(target = other); tb.start(); time.sleep(0.3)
    assert "state" not in built
    release.set(); ta.join(); tb.join()
    _assert_bitwise(before, vllm_model)
    for _, W, P in built["state"].entries:
        assert torch.equal(W, P)


def test_replaced_lora_layer_invalidates_the_state():
    model, llm, vllm_model = _make_model(target_modules = ("q_proj",))
    state = prepare_merged_rollout(model)
    attn = model.base_model.model.model.layers[0].self_attn
    old = attn.q_proj
    # A new LoRA layer object on the same base weight, with a new B.
    new = copy.copy(old)
    new._modules = dict(old._modules)
    new.lora_B = nn.ModuleDict({"default": nn.Linear(8, Q, bias = False)})
    nn.init.normal_(new.lora_B["default"].weight, std = 0.5)
    attn.q_proj = new
    assert not state.is_valid_for(model)
    base_q = new.base_layer.weight.detach().clone()
    llm.generate(["hi"], lora_request = _MergedLoRARequest(state, _ADAPTER_DIR))
    assert llm.calls[-1][1] is None
    assert torch.equal(llm.calls[-1][2][0][:Q], _expected_fold(new, base_q))


def test_replaced_lora_B_parameter_invalidates_the_state():
    model, llm, vllm_model = _make_model(target_modules = ("q_proj",))
    state = prepare_merged_rollout(model)
    _lora_modules(model)[0].lora_B["default"] = nn.Linear(8, Q, bias = False)
    assert not state.is_valid_for(model)


def test_sentinel_on_another_engine_uses_a_real_lora_request(_isolate):
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    req = _req(model)
    other_model = _FakeVllmModel(2)
    other = install_merged_rollout_engine_wrapper(_FakeLLM(other_model))
    other.generate(["hi"], lora_request = req)
    assert type(other.calls[-1][1]) is _isolate.LoRARequest
    assert llm.calls == [] and llm.resets == []
    _assert_bitwise(before, vllm_model)


@pytest.mark.parametrize("how", ["false", "raise"])
def test_failed_reset_after_restore_blocks_plain_calls_until_it_resets(how):
    model, llm, vllm_model = _make_model()
    req = _req(model)
    n = _failing_reset(llm, how, when = {2, 3})
    llm.generate(["hi"], lora_request = req)
    assert n["n"] == 2
    calls = len(llm.calls)
    with pytest.raises(RuntimeError, match = "prefix cache still holds"):
        llm.generate(["plain"], lora_request = None)
    assert len(llm.calls) == calls and n["n"] == 3
    # The retry works: the plain call runs after the reset, with no extra one.
    llm.generate(["plain"], lora_request = None)
    assert n["n"] == 4
    assert llm.calls[-1][1] is None
    llm.generate(["plain"], lora_request = None)
    assert n["n"] == 4
