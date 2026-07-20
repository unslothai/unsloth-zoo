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

"""Regression tests for OLMoE fused-experts routing (unslothai/unsloth-zoo#850).

Transformers v5 stores OLMoE experts as fused 3D nn.Parameters, so the generic MoE
bnb-4bit quantizer matches and quantizes them — but with no per-arch forward patch
the native OlmoeExperts.forward received the packed 2D uint8 Params4bit storage and
crashed on the first forward (IndexError / 'expected mat_a ... got Byte'). These
tests assert patch_olmoe_moe routes OlmoeExperts.forward through the Unsloth MoE
backend, registers the explicit weight preprocessor (OLMoE's shipping gate_up is
square — 2*1024 == 2048 == hidden — so grouped_mm layout cannot be inferred from
shape, unslothai/unsloth-zoo#849), and that the patched forward still matches stock
OLMoE expert math on CPU. The bnb-4bit path itself needs CUDA and is exercised by
the end-to-end run in the PR.
"""

import types

import pytest
import torch

from unsloth_zoo.temporary_patches.olmoe import (
    patch_olmoe_moe,
    _olmoe_weight_preprocessor,
)
from unsloth_zoo.temporary_patches.moe_utils import (
    get_weight_preprocessor,
    select_moe_backend,
)

try:
    from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
except ImportError:  # transformers < 5: ModuleList experts, the patch is a strict no-op
    OlmoeExperts = None
    OlmoeConfig = None

requires_fused_olmoe = pytest.mark.skipif(
    OlmoeExperts is None, reason="transformers < 5 has no fused OlmoeExperts"
)


# --- 1. The registered preprocessor: F.linear layout -> grouped_mm layout, both projections. ---

def test_preprocessor_transposes_flinear_to_grouped():
    E, H, I = 4, 64, 32  # 2I == H: OLMoE's ambiguous square gate_up shape
    gate_up_flinear = torch.randn(E, 2 * I, H)
    down_flinear = torch.randn(E, H, I)

    out_gu = _olmoe_weight_preprocessor(gate_up_flinear, "gate_up", H)
    out_dn = _olmoe_weight_preprocessor(down_flinear, "down", H)
    assert torch.equal(out_gu, gate_up_flinear.transpose(-2, -1))  # (E, H, 2I)
    assert torch.equal(out_dn, down_flinear.transpose(-2, -1))     # (E, I, H)


# --- 2. Patch wiring: forward replaced, model_type + registry installed, idempotent. ---

@requires_fused_olmoe
def test_patch_installs_backend_and_registry():
    patch_olmoe_moe()

    assert getattr(OlmoeExperts, "_unsloth_already_patched", False)
    assert getattr(OlmoeExperts, "_unsloth_model_type", None) == "olmoe"
    assert get_weight_preprocessor("olmoe") is _olmoe_weight_preprocessor
    assert getattr(OlmoeExperts, "_unsloth_lora_extractor_fn", None) is not None
    # The dispatcher replaced the native expert loop.
    assert OlmoeExperts.forward.__name__ == "forward_moe_backend"

    patch_olmoe_moe()  # second call must be a no-op, not a double-patch
    assert OlmoeExperts.forward.__name__ == "forward_moe_backend"


# --- 3. Patched forward == stock OLMoE expert math on CPU (square gate_up config). ---

@requires_fused_olmoe
def test_patched_forward_matches_stock_math_cpu(monkeypatch):
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "native_torch")
    # select_moe_backend is @lru_cache(maxsize=1): drop any choice cached by an
    # earlier test so the env pin above actually takes effect, and drop ours on
    # the way out so it can't leak into later tests (e.g. on a GPU host whose
    # uncached choice would be grouped_mm).
    select_moe_backend.cache_clear()
    try:
        patch_olmoe_moe()

        E, H, I, T, K = 8, 64, 32, 16, 2  # 2I == H -> the #849-ambiguous shape
        cfg = OlmoeConfig(
            hidden_size=H, intermediate_size=I, num_local_experts=E,
            num_experts_per_tok=K, hidden_act="silu",
        )
        torch.manual_seed(0)
        experts = OlmoeExperts(cfg).float().eval()
        with torch.no_grad():
            experts.gate_up_proj.normal_(0.0, 0.2)
            experts.down_proj.normal_(0.0, 0.2)

        hs = torch.randn(T, H)
        idx = torch.randint(0, E, (T, K))
        w = torch.softmax(torch.rand(T, K), dim=-1)

        with torch.no_grad():
            out = experts(hs, idx, w)

            # Stock OlmoeExperts math: F.linear over the (E, 2I, H)/(E, H, I) weights.
            ref = torch.zeros(T, H)
            for t in range(T):
                for k in range(K):
                    e = idx[t, k].item()
                    gate, up = torch.nn.functional.linear(hs[t], experts.gate_up_proj[e]).chunk(2, dim=-1)
                    ref[t] += w[t, k] * torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, experts.down_proj[e])

        assert out.shape == ref.shape
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5), (
            f"max abs diff {(out - ref).abs().max().item()}"
        )
    finally:
        select_moe_backend.cache_clear()


# --- 4. The compiled-cache registry split (Codex review on #915): get_forward_moe_backend
#        may resolve the forward from the unsloth_compiled_cache copy of moe_utils — a
#        distinct module object whose _WEIGHT_PREPROCESSORS starts empty, where a package-
#        only registration is invisible and OLMoE's square gate_up falls back to shape
#        inference (#849). The patch must land the preprocessor in BOTH namespaces. ---

_SPLIT_NAMESPACE_SRC = """
import torch

_WEIGHT_PREPROCESSORS = {}

def register_weight_preprocessor(model_type, preprocessor_fn):
    _WEIGHT_PREPROCESSORS[model_type] = preprocessor_fn

def get_weight_preprocessor(model_type):
    return _WEIGHT_PREPROCESSORS.get(model_type)

# The signature (names + annotations) must match the real dispatcher: patch_function's
# strict fingerprint check compares them before installing the replacement.
def forward_moe_backend(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("routing stand-in; never called in this test")
"""


@requires_fused_olmoe
def test_preprocessor_registered_in_executing_namespace(monkeypatch):
    import unsloth_zoo.temporary_patches.olmoe as olmoe_mod

    # Stand-in for the cache-loaded moe_utils: a real module object with its own empty
    # registry, its own register/get API, and a forward bound to its namespace.
    split = types.ModuleType("unsloth_cached_moe_utils_standin")
    exec(compile(_SPLIT_NAMESPACE_SRC, split.__name__, "exec"), split.__dict__)
    assert split.forward_moe_backend.__globals__ is split.__dict__

    # Re-run the patch with the stand-in as the resolved backend; register the current
    # forward with monkeypatch so it is restored for the other tests afterwards.
    monkeypatch.setattr(OlmoeExperts, "forward", OlmoeExperts.forward)
    monkeypatch.setattr(OlmoeExperts, "_unsloth_already_patched", False, raising=False)
    monkeypatch.setattr(
        olmoe_mod, "get_forward_moe_backend", lambda: split.forward_moe_backend
    )

    olmoe_mod.patch_olmoe_moe()

    # The package registry has the preprocessor ...
    assert get_weight_preprocessor("olmoe") is _olmoe_weight_preprocessor
    # ... and so does the namespace the installed forward actually reads.
    installed = OlmoeExperts.forward
    assert installed is split.forward_moe_backend
    assert installed.__globals__["_WEIGHT_PREPROCESSORS"].get("olmoe") is _olmoe_weight_preprocessor
    assert split.get_weight_preprocessor("olmoe") is _olmoe_weight_preprocessor
