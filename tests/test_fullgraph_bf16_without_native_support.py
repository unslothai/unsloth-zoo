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

"""bf16 fullgraph on sm < 80 runs eagerly; torch 2.13+ raises 'found no compiled frames' (CPU-faked)."""

import pytest
import torch

from unsloth_zoo.temporary_patches import utils as U

pytestmark = pytest.mark.usefixtures("compile_enabled")


def _region(calls):
    def swiglu_like(a):
        calls.append(a.dtype)
        return (a.float() * 2).to(a.dtype)
    return swiglu_like


def _skips_bf16(gm, inputs):
    from torch._dynamo.exc import SkipFrame
    if any(getattr(t, "dtype", None) == torch.bfloat16 for t in inputs):
        raise SkipFrame("BF16 is not supported")
    return gm.forward


@pytest.fixture
def no_native_bf16(monkeypatch):
    monkeypatch.setattr(U, "_NO_NATIVE_BF16", frozenset({("cpu", None)}))
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_bf16_runs_eagerly_and_later_dtypes_still_compile(no_native_bf16):
    calls, compiled = [], []
    region = U.torch_compile_with_fallback(fullgraph = True, backend = lambda gm, i: compiled.append(1) or _skips_bf16(gm, i))(_region(calls))
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        x = torch.randn(4, 8).to(dtype)
        assert torch.equal(region(x), (x.float() * 2).to(dtype))
    assert calls[0] == torch.bfloat16 and compiled, "bf16 went eager, fp16 / fp32 still compiled"
    assert hasattr(region, "_unsloth_compiled_func") and region.__wrapped__.__name__ == "swiglu_like"


def test_devices_with_native_bf16_keep_the_compiled_call(monkeypatch):
    monkeypatch.setattr(U, "_NO_NATIVE_BF16", frozenset())
    seen = []
    wrapper = U._eager_for_bf16_without_native_support(lambda x: seen.append("eager"), lambda x: seen.append("compiled"))
    wrapper(torch.zeros(2, dtype = torch.bfloat16))
    assert seen == ["compiled"]


def test_only_bf16_tensors_on_listed_devices_go_eager(no_native_bf16):
    devices = U._NO_NATIVE_BF16
    assert U._bf16_on(devices, (torch.zeros(1, dtype = torch.bfloat16),), {})
    assert U._bf16_on(devices, (), {"a": torch.zeros(1, dtype = torch.bfloat16)})
    assert not U._bf16_on(devices, (torch.zeros(1, dtype = torch.float16), 1.702, None), {})
    assert not U._bf16_on(frozenset({("cuda", 0)}), (torch.zeros(1, dtype = torch.bfloat16),), {})
