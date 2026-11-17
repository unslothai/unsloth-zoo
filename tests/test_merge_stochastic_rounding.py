# SPDX-License-Identifier: AGPL-3.0-or-later
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

"""Stochastic merge rounding keeps sub-ulp LoRA deltas in expectation."""

from __future__ import annotations

import os

import pytest
import torch

import _merge_e2e_helpers as H
from unsloth_zoo import saving_utils as S


DTYPES = [torch.bfloat16, torch.float16]
DELTA = 2.0 ** -16


def _gap_up(r):
    bits = r.view(torch.int16).to(torch.int32)
    return ((bits + 1).to(torch.int16).view(r.dtype).float() - r.float())


@pytest.mark.parametrize("dtype", DTYPES)
def test_representable_values_are_unchanged(dtype):
    x = (torch.randn(4096) * 0.05).to(dtype)
    x[:4] = torch.tensor([0.0, -0.0, 1.0, -2.0], dtype=dtype)
    out = S._stochastic_round(x.float(), dtype, "k")
    assert out.dtype == dtype
    assert torch.equal(out.view(torch.int16), x.view(torch.int16))


@pytest.mark.parametrize("dtype", DTYPES)
def test_rounds_to_a_neighbour_and_is_unbiased(dtype):
    base = torch.full((1 << 20,), 1.0, dtype=dtype)
    ulp = float(_gap_up(base[:1]))
    for frac in (0.1, 0.25, 0.5, 0.8):
        x = base.float() + frac * ulp
        out = S._stochastic_round(x, dtype, f"unbiased.{frac}").float()
        assert set(out.unique().tolist()) <= {1.0, 1.0 + ulp}
        se = (frac * (1 - frac) / x.numel()) ** 0.5
        assert abs(float((out - 1.0).mean()) / ulp - frac) < 6 * se
    # Below zero, across a binade edge (the gap below 1.0 is half the gap above).
    x = torch.full((1 << 20,), -1.0) - 0.3 * ulp
    out = S._stochastic_round(x, dtype, "neg").float()
    assert set(out.unique().tolist()) <= {-1.0, -1.0 - ulp}
    x = torch.full((1 << 20,), 1.0) - 0.2 * ulp  # 0.4 of the lower gap
    out = S._stochastic_round(x, dtype, "edge").float()
    assert abs(float(out.mean()) - float(x.mean())) < 6 * (0.24 / x.numel()) ** 0.5 * ulp / 2


@pytest.mark.parametrize("dtype", DTYPES)
def test_deterministic_per_key(dtype):
    x = torch.randn(3, 1000) * 0.02 + torch.randn(3, 1000) * 1e-6
    a = S._stochastic_round(x, dtype, "model.layers.0.q_proj.weight")
    b = S._stochastic_round(x, dtype, "model.layers.0.q_proj.weight")
    c = S._stochastic_round(x, dtype, "model.layers.1.q_proj.weight")
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert not torch.equal(a.view(torch.int16), c.view(torch.int16))
    assert a.shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", DTYPES)
def test_gpu_deterministic_neighbour_and_unbiased(dtype):
    x16 = (torch.randn(1 << 20, device="cuda") * 0.02).to(dtype)
    assert torch.equal(S._stochastic_round(x16.float(), dtype, "g").view(torch.int16), x16.view(torch.int16))
    base = torch.full((1 << 20,), 1.0, dtype=dtype, device="cuda")
    ulp = float(_gap_up(base[:1]))
    x = base.float() + 0.25 * ulp
    a = S._stochastic_round(x, dtype, "g")
    b = S._stochastic_round(x, dtype, "g")
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    assert set(a.float().unique().tolist()) <= {1.0, 1.0 + ulp}
    assert abs(float((a.float() - 1.0).mean()) / ulp - 0.25) < 6 * (0.1875 / x.numel()) ** 0.5


def test_mode_switch(monkeypatch):
    x = torch.full((4096,), 1.0) + 2.0 ** -10  # a quarter of a bf16 ulp above 1.0
    monkeypatch.delenv("UNSLOTH_MERGE_ROUNDING", raising=False)
    assert S._merge_rounding_mode() == S._MERGE_ROUNDING_DEFAULT
    monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", "nearest")
    assert S._round_merged_weight(x, torch.bfloat16, "k") is x
    monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", "stochastic")
    out = S._round_merged_weight(x, torch.bfloat16, "k")
    assert out.dtype == torch.bfloat16 and bool((out.float() != 1.0).any())
    assert S._round_merged_weight(x, torch.float32, "k") is x
    monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", "bogus")
    with pytest.warns(UserWarning):
        assert S._merge_rounding_mode() == "nearest"


def _merge_tiny_delta(tmp_path, tag, mode, monkeypatch):
    """Merge a bf16 tiny llama whose LoRA delta is DELTA everywhere: below half a bf16 ulp wherever |W| >= 2**-7."""
    H.set_offline_cpu_env()
    spec = H.make_spec("llama")
    base_dir = os.path.join(str(tmp_path), "base")
    if not os.path.isdir(base_dir):
        H.build_and_save_base(spec, base_dir, dtype=torch.bfloat16)
    model = H.build_and_save_base(spec, os.path.join(str(tmp_path), f"scratch_{tag}"), dtype=torch.bfloat16)
    model.config._name_or_path = base_dir
    peft_model = H.attach_lora(model, spec, "full")
    deltas = {}
    with torch.no_grad():
        for name, mod in H._iter_lora_modules(peft_model):
            key = H._module_to_key(name, None)
            A = mod.lora_A["default"].weight; B = mod.lora_B["default"].weight; s = float(mod.scaling["default"])
            B.zero_(); A.zero_()
            B[:, :1] = 1.0; A[:1, :] = DELTA / s
            deltas[key] = s * (B.float() @ A.float())
    if mode is None:
        monkeypatch.delenv("UNSLOTH_MERGE_ROUNDING", raising=False)
    else:
        monkeypatch.setenv("UNSLOTH_MERGE_ROUNDING", mode)
    out_dir = os.path.join(str(tmp_path), f"merged_{tag}")
    H.run_merge(peft_model, base_dir, out_dir, save_dtype=torch.bfloat16)
    return H.read_safetensors_dir(base_dir), H.read_safetensors_dir(out_dir), deltas


def test_default_mode_is_stochastic(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MERGE_ROUNDING", raising=False)
    assert S._merge_rounding_mode() == "stochastic"


def test_sub_ulp_delta_vanishes_with_nearest_and_survives_by_default(tmp_path, monkeypatch):
    base, rtn, deltas = _merge_tiny_delta(tmp_path, "rtn", "nearest", monkeypatch)
    _, sr, _ = _merge_tiny_delta(tmp_path, "sr", None, monkeypatch)
    _, sr2, _ = _merge_tiny_delta(tmp_path, "sr2", "stochastic", monkeypatch)
    got_sum = want_sum = 0.0
    checked = 0
    for key, d in deltas.items():
        W = base[key].float()
        big = W.abs() >= 2.0 ** -7  # half a bf16 ulp there is >= 2**-15 > DELTA
        if not bool(big.any()):
            continue
        checked += int(big.sum())
        # Round to nearest: the whole delta is lost on those elements.
        assert torch.equal(rtn[key][big], base[key][big])
        got = (sr[key].float() - W)[big]
        assert bool(((got == 0) | (got > 0)).all())
        got_sum += float(got.double().sum()); want_sum += float(d[big].double().sum())
        assert torch.equal(sr[key].view(torch.int16), sr2[key].view(torch.int16))
    assert checked >= 10000, checked
    # The delta survives on average (binomial noise over >= 10k elements is a few percent).
    assert abs(got_sum / want_sum - 1.0) < 0.1, got_sum / want_sum
    for key in base:
        if key not in deltas:
            assert torch.equal(sr[key].view(torch.int16), base[key].view(torch.int16))
