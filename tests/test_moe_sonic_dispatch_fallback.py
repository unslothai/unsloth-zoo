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

"""UNSLOTH_MOE_BACKEND=sonic_moe: the bnb4bit, FP8 and Qwen3-VL dispatchers take grouped_mm, not the eager loop."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import unsloth_zoo.temporary_patches.moe_utils as M
import unsloth_zoo.temporary_patches.moe_routed as R
import unsloth_zoo.temporary_patches.moe_utils_bnb4bit as B
import unsloth_zoo.temporary_patches.moe_utils_fp8 as F


E, H, I, T, K = 4, 16, 8, 6, 2


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = E
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(E, 2 * I, H), requires_grad = False)
        self.down_proj = torch.nn.Parameter(torch.zeros(E, H, I), requires_grad = False)
        self.act_fn = torch.nn.SiLU()


def _inputs():
    hidden = torch.randn(T, H)
    top_k_index = torch.randint(0, E, (T, K))
    top_k_weights = torch.rand(T, K)
    return hidden, top_k_index, top_k_weights


def _spies(monkeypatch, targets):
    calls = []

    def make(name):
        def spy(self, hidden_states, *args, **kwargs):
            calls.append(name)
            return torch.zeros_like(hidden_states)
        return spy

    for module in targets:
        for name in ("forward_native_grouped_mm", "forward_native_moe_loop", "forward_triton_grouped_gemm"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, make(name))
    return calls


@pytest.fixture
def selected(monkeypatch):
    def select(backend):
        monkeypatch.setattr(M, "_SELECTED_MOE_BACKEND", backend)
    yield select


@pytest.fixture
def no_routed(monkeypatch):
    monkeypatch.setattr(R, "routed_moe_forward", lambda *args, **kwargs: None)


def _call_through_swap(self, gate_up, down, forward_fn, *args, **kwargs):
    return forward_fn(self, *args, **kwargs)


@pytest.mark.parametrize("backend, expected", [
    ("sonic_moe", "forward_native_grouped_mm"),
    ("grouped_mm", "forward_native_grouped_mm"),
    ("native_torch", "forward_native_moe_loop"),
])
def test_bnb4bit_dequant_route(monkeypatch, selected, no_routed, backend, expected):
    selected(backend)
    calls = _spies(monkeypatch, [M])
    monkeypatch.setattr(B, "_is_bnb4bit_param", lambda p: True)
    monkeypatch.setattr(M, "_moe_recompute_enabled", lambda *a, **k: False)
    monkeypatch.setattr(B, "_dequantize_bnb4bit_expert_weights", lambda w, dtype: torch.zeros(1))
    monkeypatch.setattr(M, "swap_moe_weights_for_call", _call_through_swap)
    B.forward_moe_backend_bnb4bit(_Experts(), *_inputs())
    assert calls == [expected]


@pytest.mark.parametrize("backend, recompute", [("sonic_moe", True), ("grouped_mm", True), ("native_torch", False)])
def test_bnb4bit_recompute_route_keeps_packed_weights(monkeypatch, selected, no_routed, backend, recompute):
    selected(backend)
    calls = _spies(monkeypatch, [M])
    dequantized = []
    monkeypatch.setattr(B, "_is_bnb4bit_param", lambda p: True)
    monkeypatch.setattr(M, "_moe_recompute_enabled", lambda *a, **k: True)
    monkeypatch.setattr(B, "_dequantize_bnb4bit_expert_weights", lambda w, dtype: dequantized.append(1) or torch.zeros(1))
    monkeypatch.setattr(M, "swap_moe_weights_for_call", _call_through_swap)
    B.forward_moe_backend_bnb4bit(_Experts(), *_inputs())
    if recompute:
        assert calls == ["forward_native_grouped_mm"] and dequantized == []
    else:
        assert calls == ["forward_native_moe_loop"] and dequantized == [1, 1]


def _fp8_stubs(monkeypatch, scaled_ok):
    scaled = []
    monkeypatch.setattr(F, "_fp8_experts_own_gate", lambda m: False)
    monkeypatch.setattr(F, "_check_torch_scaled_grouped_mm_supported", lambda device = None: scaled_ok)

    def fake_scaled(self, hidden_states, top_k_index, top_k_weights):
        scaled.append(1)
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(F, "_forward_scaled_grouped_mm_fp8", fake_scaled)
    monkeypatch.setattr(F, "_get_moe_weight_and_quant_info", lambda m, name: (torch.zeros(1), None, None))
    monkeypatch.setattr(F, "_refuse_fp4_with_an_unapplied_gate", lambda *a: None)
    monkeypatch.setattr(F, "_dequantize_full_expert_weights", lambda *a, **k: torch.zeros(1))
    monkeypatch.setattr(F, "_release_dequantized_banks_until_backward", lambda *a: None)
    monkeypatch.setattr(M, "swap_moe_weights_for_call", _call_through_swap)
    return scaled


@pytest.mark.parametrize("backend, takes_scaled", [("sonic_moe", True), ("grouped_mm", True), ("native_torch", False)])
def test_fp8_scaled_grouped_mm_route(monkeypatch, selected, backend, takes_scaled):
    selected(backend)
    calls = _spies(monkeypatch, [M])
    scaled = _fp8_stubs(monkeypatch, scaled_ok = True)
    F.forward_moe_backend_fp8(_Experts(), *_inputs())
    if takes_scaled:
        assert scaled == [1] and calls == []
    else:
        assert scaled == [] and calls == ["forward_native_moe_loop"]


@pytest.mark.parametrize("backend, expected", [
    ("sonic_moe", "forward_native_grouped_mm"),
    ("grouped_mm", "forward_native_grouped_mm"),
    ("native_torch", "forward_native_moe_loop"),
])
def test_fp8_dequant_route(monkeypatch, selected, backend, expected):
    selected(backend)
    calls = _spies(monkeypatch, [M])
    _fp8_stubs(monkeypatch, scaled_ok = False)
    F.forward_moe_backend_fp8(_Experts(), *_inputs())
    assert calls == [expected]


@pytest.mark.parametrize("backend", ["grouped_mm", "unsloth_triton", "native_torch"])
def test_without_sonic_leaves_other_backends(monkeypatch, backend):
    monkeypatch.setattr(M, "select_moe_backend", lambda: backend)
    assert M.select_moe_backend_without_sonic() == backend


def test_without_sonic_maps_sonic_to_grouped_mm(monkeypatch):
    monkeypatch.setattr(M, "select_moe_backend", lambda: "sonic_moe")
    assert M.select_moe_backend_without_sonic() == "grouped_mm"


@pytest.mark.parametrize("backend, expected", [
    ("sonic_moe", "forward_sonic_moe"),
    ("grouped_mm", "forward_native_grouped_mm"),
    ("native_torch", "forward_native_moe_loop"),
])
def test_qwen3_vl_moe_experts_forward(monkeypatch, selected, backend, expected):
    modeling = pytest.importorskip("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe")
    experts_cls = getattr(modeling, "Qwen3VLMoeTextExperts", None)
    if experts_cls is None:
        pytest.skip("transformers has no Qwen3VLMoeTextExperts")
    import unsloth_zoo.temporary_patches.qwen3_vl_moe as Q
    import unsloth_zoo.temporary_patches.moe_utils_sonic as S

    # Restore every class attribute the patch replaces.
    for cls in (experts_cls, modeling.Qwen3VLMoeTextSparseMoeBlock):
        for name, value in list(vars(cls).items()):
            if not name.startswith("__") or name == "__init__":
                monkeypatch.setattr(cls, name, value)

    selected(backend)
    calls = _spies(monkeypatch, [Q, M])
    monkeypatch.setattr(Q, "_bnb4bit_experts_forward_or_none", lambda *a, **k: None)

    def fake_sonic(self, hidden_states, *args, **kwargs):
        calls.append("forward_sonic_moe")
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(S, "forward_sonic_moe", fake_sonic)
    Q.patch_qwen3_vl_moe()
    experts_cls.forward(_Experts(), *_inputs())
    assert calls == [expected]


def test_without_sonic_takes_the_loop_on_a_card_without_grouped_mm(monkeypatch):
    # Mixed host: sonic_moe was selected on a grouped_mm card, the expert runs on one without it.
    monkeypatch.setattr(M, "select_moe_backend", lambda: "sonic_moe")
    monkeypatch.setattr(M, "_check_torch_grouped_mm_supported", lambda device = None: False)
    assert M.select_moe_backend_without_sonic(torch.device("cuda", 1)) == "native_torch"
    assert M.select_moe_backend_without_sonic(torch.device("cpu")) == "grouped_mm"
    monkeypatch.setattr(M, "_check_torch_grouped_mm_supported", lambda device = None: True)
    assert M.select_moe_backend_without_sonic(torch.device("cuda", 1)) == "grouped_mm"


def test_compiled_qwen3_vl_moe_sonic_forward_resolves_dispatcher(tmp_path, monkeypatch):
    # The Qwen3-VL experts forward is copied into the compiled cache: forward_sonic_moe must be imported.
    from unsloth_zoo import compiler
    from unsloth_zoo.temporary_patches import moe_utils_sonic

    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    source = "def _probe(self, x):\n    return forward_sonic_moe(self, x, None, None)\n"
    module = compiler.create_new_function(
        "UnslothSonicDispatchImportProbe", source,
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", [], overwrite = True,
    )
    assert module.forward_sonic_moe is moe_utils_sonic.forward_sonic_moe
