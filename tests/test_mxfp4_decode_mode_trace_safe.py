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

"""The compiled-region decode-mode resolver must be trace-safe. On sm_75 Inductor raises
``SkipFrame("BF16 is not supported")``, Dynamo then compiles ``_resolve_decode_mode`` as its own frame with a
symbolic index, and ``torch.device("cuda", SymInt)`` raised Unsupported under fullgraph. CPU only: the capability,
the hardware-convert probe and the skipping backend are faked."""

import pytest
import torch

import unsloth_zoo.mxfp4_gemm as mg


def _fake_gpu(monkeypatch, capability, cvt = False):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: capability)
    monkeypatch.setattr(mg, "_cvt_ok", lambda index, eager = False: cvt)
    monkeypatch.setattr(mg, "_ASM_OK", {})
    monkeypatch.delenv("UNSLOTH_MXFP4_ASM", raising = False)
    torch._dynamo.reset()


def _skipping_backend(gm, inputs):
    from torch._dynamo.exc import SkipFrame
    raise SkipFrame("BF16 is not supported")


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (12, 0)])
def test_a_skipped_caller_frame_does_not_break_fullgraph(monkeypatch, capability):
    _fake_gpu(monkeypatch, capability)

    @torch.compile(fullgraph = True, dynamic = True, backend = _skipping_backend)
    def region(x):
        mg._resolve_decode_mode(0)
        return x * 2

    try:
        assert torch.equal(region(torch.ones(3)), torch.full((3,), 2.0))
    except RuntimeError as error:
        # torch >= 2.13 rejects a fullgraph call whose every frame was skipped; the region still ran eagerly.
        assert "found no compiled frames" in str(error) and "device" not in str(error), error
    assert mg._ASM_OK[0] == (capability >= (8, 0))


@pytest.mark.parametrize("capability, cvt, want", [((7, 5), False, False), ((8, 0), False, True), ((12, 0), True, 2)])
def test_a_compiled_region_bakes_in_the_resolved_mode(monkeypatch, capability, cvt, want):
    _fake_gpu(monkeypatch, capability, cvt)

    @torch.compile(fullgraph = True, dynamic = True, backend = "eager")
    def region(x):
        mode = mg._resolve_decode_mode(0)
        return x * (3 if mode == 2 else 2 if mode else 1)

    scale = 3 if want == 2 else 2 if want else 1
    assert torch.equal(region(torch.ones(3)), torch.full((3,), float(scale)))
    assert mg._ASM_OK[0] is want or mg._ASM_OK[0] == want


def test_the_resolver_never_builds_a_device(monkeypatch):
    _fake_gpu(monkeypatch, (12, 0), cvt = True)
    seen = []
    real = mg._asm_ok
    monkeypatch.setattr(mg, "_asm_ok", lambda device, *a, **k: seen.append(device) or real(device, *a, **k))
    assert mg._resolve_decode_mode(0) == 2
    assert seen == [0] and not isinstance(seen[0], torch.device)
    assert mg._asm_ok(torch.device("cuda", 0)) == 2  # device and index share one cache entry
