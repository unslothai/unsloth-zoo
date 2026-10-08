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

import ast
import inspect
import sys
import types

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model

from unsloth_zoo import vllm_utils
from unsloth_zoo.vllm_utils import (
    _MergedLoRARequest,
    install_merged_rollout_engine_wrapper,
    load_lora,
    merged_rollout,
    merged_rollout_ineligible_reason,
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
    for name, module in base.named_modules():
        if isinstance(module, nn.Linear) and module.weight.dtype != base_dtype:
            module.weight = nn.Parameter(module.weight.detach().to(base_dtype), requires_grad = False)
    if base_dtype != DTYPE:
        for module in base.modules():
            if isinstance(module, nn.Linear):
                module.weight = nn.Parameter(module.weight.detach().to(base_dtype), requires_grad = False)
    for p in base.parameters(): p.requires_grad_(False)

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
        "bitsandbytes": (dict(), bnb, "bitsandbytes"),
        "fp8_quant_config": (dict(), fp8, "quantized"),
        "moe_experts": (dict(extra = "experts", target_modules = ("q_proj", "experts.0")), None, "is a MoE expert LoRA"),
        "vision_tower": (dict(extra = "visual", target_modules = ("q_proj", "proj")), None, "vision / audio tower"),
        "embed_tokens": (dict(target_modules = ("q_proj", "embed_tokens")), None, "has an embedding LoRA"),
        "lm_head": (dict(target_modules = ("q_proj", "lm_head")), None, "is an embedding or lm_head"),
        "modules_to_save": (dict(modules_to_save = ["lm_head"]), None, "modules_to_save"),
        "dora": (dict(use_dora = True), None, "DoRA is not a plain low-rank delta"),
        "lora_bias": (dict(lora_bias = True), None, "lora_bias adds a bias term"),
        "fan_in_fan_out": (dict(), fan, "uses fan_in_fan_out"),
        "already_merged": (dict(), merged, "already has a PEFT-merged adapter"),
        "two_adapters": (dict(), second, "more than one adapter"),
        "non_default_adapter": (dict(adapter_name = "other"), None, "adapter"),
        "fp32_base": (dict(alias = False, base_dtype = torch.float32), None, "not 16-bit"),
        "dora_module_only": (dict(), lambda m: _lora_modules(m)[0].use_dora.__setitem__("default", True), "uses DoRA"),
        "not_aliased": (dict(alias = False), None, "does not share storage with the vLLM engine"),
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
    assert merged_rollout_ineligible_reason(model) is None
    assert isinstance(load_lora(model, _ADAPTER_DIR, load_tensors = True), _MergedLoRARequest)


def test_auto_mode_is_also_on_and_unknown_values_mean_on(monkeypatch):
    for value in ("auto", "1", "true"):
        monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", value)
        assert vllm_utils._merged_rollout_mode() != "0"
    for value in ("0", "", "false", "off"):
        monkeypatch.setenv("UNSLOTH_VLLM_MERGED_ROLLOUT", value)
        assert vllm_utils._merged_rollout_mode() == "0"


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


def test_prefix_cache_is_reset_inside_the_fold_and_after_restore():
    """Merged generate hashes its KV as base blocks: stale prefixes must not survive."""
    model, llm, vllm_model = _make_model()
    before = _snapshot(vllm_model)
    request = load_lora(model, _ADAPTER_DIR, load_tensors = True)
    llm.chat(["hi"], None, True, request)
    assert len(llm.resets) == 2
    folded = llm.calls[-1][2]
    assert all(torch.equal(a, b) for a, b in zip(folded, llm.resets[0]))
    assert all(torch.equal(a, b) for a, b in zip(before, llm.resets[1]))
    llm.generate(["hi"], lora_request = object())
    assert len(llm.resets) == 2


def test_wrapper_keeps_the_real_signature_for_unsloth_binding_check():
    model, llm, vllm_model = _make_model()
    sig = inspect.signature(llm.chat)
    assert "lora_request" in sig.bind_partial(["m"], None, True, "x").arguments
    install_merged_rollout_engine_wrapper(llm)
    assert not hasattr(llm.generate.__wrapped__, "_unsloth_merged_rollout_wrapper")


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
    assert vllm_utils._merged_rollout_mode() == "0"
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


def test_kill_switch_off_never_wraps_the_engine():
    """load_vllm installs the wrapper only behind `_merged_rollout_mode() != "0"`."""
    tree = ast.parse(inspect.getsource(vllm_utils.load_vllm))
    guarded = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "_merged_rollout_mode" in ast.unparse(node.test):
            guarded += [ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Call)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and "install_merged_rollout_engine_wrapper" in ast.unparse(n.func)]
    assert len(calls) == 1
    assert any("install_merged_rollout_engine_wrapper" in g for g in guarded)
    assert "!= '0'" in ast.unparse(next(n.test for n in ast.walk(tree) if isinstance(n, ast.If) and "_merged_rollout_mode" in ast.unparse(n.test)))


def test_state_is_cached_and_rebuilt_when_adapters_change():
    model, llm, vllm_model = _make_model()
    s1 = prepare_merged_rollout(model)
    assert prepare_merged_rollout(model) is s1
    model.add_adapter("other", LoraConfig(r = 4, target_modules = ["q_proj"]))
    assert prepare_merged_rollout(model) is None
