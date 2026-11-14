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

"""Kept-packed GPT-OSS experts must load on transformers 5.23's shape-checking set_param_for_module."""

import os

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
core = pytest.importorskip("transformers.core_model_loading")
if not hasattr(core, "set_param_for_module"):
    pytest.skip("this transformers has no set_param_for_module", allow_module_level = True)
from transformers.utils.loading_report import LoadStateDictInfo  # noqa: E402

from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam  # noqa: E402
from unsloth_zoo.temporary_patches import mxfp4 as zoo_mxfp4  # noqa: E402

import inspect  # noqa: E402

_CHECKS_UNDER_QUANTIZER = "param_needs_quantization" in inspect.getsource(inspect.unwrap(core.set_param_for_module))
_needs_check = pytest.mark.skipif(not _CHECKS_UNDER_QUANTIZER, reason = "this transformers skips the shape check under a quantizer")

E, K, N = 4, 32, 64  # logical (E, K, N), packed blocks (E, N, K // 32, 16)


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(E, K, N, dtype = torch.bfloat16))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Experts()


class _DequantizeQuantizer:
    pre_quantized = True

    def param_needs_quantization(self, model, param_name, **kwargs):
        return False


def _info():
    fields = LoadStateDictInfo.__dataclass_fields__ if hasattr(LoadStateDictInfo, "__dataclass_fields__") else {}
    kwargs = {"missing_keys": {"experts.gate_up_proj"}, "unexpected_keys": set(), "mismatched_keys": set(),
              "error_msgs": [], "conversion_errors": {}, "skipped_pp_keys": set()}
    return LoadStateDictInfo(**{k: v for k, v in kwargs.items() if not fields or k in fields})


def _packed(n = N):
    blocks = torch.randint(0, 255, (E, n, K // 32, 16), dtype = torch.uint8)
    scales = torch.full((E, n, K // 32), 127, dtype = torch.uint8)
    return Mxfp4ExpertParam(blocks, mxfp4_scales = scales)


def _load(setter, param):
    model, info = _Model(), _info()
    setter(model, "experts.gate_up_proj", param, info, _DequantizeQuantizer())
    return model, info


@_needs_check
def test_the_stock_setter_rejects_a_packed_stack():
    _, info = _load(inspect.unwrap(core.set_param_for_module), _packed())
    assert {key for key, *_ in info.mismatched_keys} == {"experts.gate_up_proj"}


def test_packed_stack_loads_into_its_module():
    zoo_mxfp4.patch_mxfp4_packed_expert_shape_check()
    model, info = _load(core.set_param_for_module, _packed())
    assert not info.mismatched_keys
    assert isinstance(model.experts.gate_up_proj, Mxfp4ExpertParam)
    assert "experts.gate_up_proj" not in info.missing_keys


@_needs_check
def test_a_packed_stack_of_another_shape_is_still_mismatched():
    zoo_mxfp4.patch_mxfp4_packed_expert_shape_check()
    _, info = _load(core.set_param_for_module, _packed(n = 2 * N))
    assert {key for key, *_ in info.mismatched_keys} == {"experts.gate_up_proj"}


@_needs_check
def test_plain_tensors_keep_the_stock_shape_check():
    zoo_mxfp4.patch_mxfp4_packed_expert_shape_check()
    _, info = _load(core.set_param_for_module, torch.zeros(E, N, K, dtype = torch.bfloat16))
    assert {key for key, *_ in info.mismatched_keys} == {"experts.gate_up_proj"}
