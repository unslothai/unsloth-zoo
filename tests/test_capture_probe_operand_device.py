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
"""Card 0 current, card 1 (operand) capturing: self-checks must not probe or cache. CPU, faked CUDA device API."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

from unsloth_zoo.temporary_patches import gpt_oss_fused_glue as glue  # noqa: E402
from unsloth_zoo.temporary_patches import moe_fused_swiglu as sw  # noqa: E402
from unsloth_zoo.temporary_patches import moe_routing_triton as mrt  # noqa: E402

CARD_1 = torch.device("cuda", 1)


@pytest.fixture
def card_1_capturing(monkeypatch):
    state = {"current": 0}

    class _Device:
        def __init__(self, device):
            self.index = device.index if hasattr(device, "index") else int(device)

        def __enter__(self):
            self.prev, state["current"] = state["current"], self.index

        def __exit__(self, *exc):
            state["current"] = self.prev

    probes = []

    def _no_probe(*args, **kwargs):
        probes.append(args)
        raise RuntimeError("probe ran inside the operand card's capture")

    monkeypatch.setattr(torch.cuda, "device", _Device)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["current"] == 1)
    monkeypatch.setattr(torch, "Generator", _no_probe)
    state["probes"] = probes
    return state


def test_gpt_oss_glue_self_check_stays_out_of_the_operands_capture(monkeypatch, card_1_capturing):
    monkeypatch.setattr(glue, "_CHECKED", {})
    assert glue._self_check(CARD_1, torch.bfloat16, 1.702, 7.0) is False
    assert card_1_capturing["probes"] == [] and glue._CHECKED == {} and card_1_capturing["current"] == 0


def test_fused_swiglu_self_check_stays_out_of_the_operands_capture(monkeypatch, card_1_capturing):
    monkeypatch.setattr(sw, "_CACHE", {})
    assert sw.fused_swiglu_self_check(CARD_1, torch.bfloat16, sw.ACT_SILU, False) is False
    assert card_1_capturing["probes"] == [] and sw._CACHE == {} and card_1_capturing["current"] == 0


class _Ids:
    def __init__(self, device):
        self.device = device

    def numel(self):
        return 16


def test_routing_verification_stays_out_of_the_operands_capture(monkeypatch, card_1_capturing):
    if mrt.triton is None:
        pytest.skip("needs triton importable")
    verified = []
    monkeypatch.setattr(mrt, "_VERIFIED", {})
    monkeypatch.setattr(mrt, "_DISABLED_REASON", None)
    monkeypatch.setattr(mrt, "_KILLED", False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.delenv("TRITON_INTERPRET", raising = False)
    monkeypatch.setattr(mrt, "_verified_launch", lambda *a: verified.append(a) or (None, "ref"))
    monkeypatch.setattr(mrt, "torch_moe_routing_sort", lambda ids, E: "torch")
    assert mrt._routing_impl(_Ids(CARD_1), 8) == "torch"
    assert verified == [] and mrt._VERIFIED == {} and mrt._DISABLED_REASON is None
    assert card_1_capturing["current"] == 0
