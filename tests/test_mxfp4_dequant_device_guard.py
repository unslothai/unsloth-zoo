# SPDX-License-Identifier: AGPL-3.0-only
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

"""MXFP4 dequant on a non-current GPU; subprocess per case so a faulting kernel stays contained."""
import os
import subprocess
import sys
import textwrap

import pytest
import torch


_RUNNER = textwrap.dedent(
    """
    import sys, torch
    from unsloth_zoo.temporary_patches.mxfp4 import patch_convert_moe_packed_tensors
    patch_convert_moe_packed_tensors()
    import transformers.integrations.mxfp4 as M

    torch.manual_seed(0)
    E, D, G, B = 4, 64, 3, 16
    blocks = torch.randint(0, 256, (E, D, G, B), dtype = torch.uint8)
    scales = torch.randint(110, 140, (E, D, G), dtype = torch.uint8)

    torch.cuda.set_device(0)
    ref = M.convert_moe_packed_tensors(blocks.to("cuda:0"), scales.to("cuda:0")).cpu()
    torch.cuda.synchronize(0)

    # Operands on cuda:1 while cuda:0 stays current, as in the transformers loader threads.
    out = M.convert_moe_packed_tensors(blocks.to("cuda:1"), scales.to("cuda:1"))
    torch.cuda.synchronize(1)
    assert out.device == torch.device("cuda:1"), out.device
    assert torch.cuda.current_device() == 0, torch.cuda.current_device()
    assert torch.equal(out.cpu(), ref), "cuda:1 dequant differs from cuda:0"

    # The 5.x loader hook goes through dequantize_convertops.
    if hasattr(M, "dequantize_convertops"):
        import inspect
        args = (blocks.to("cuda:1"), scales.to("cuda:1"))
        if len(inspect.signature(M.dequantize_convertops).parameters) == 3:
            args = args + ("cuda:1",)
        w = M.dequantize_convertops(*args)
        torch.cuda.synchronize(1)
        assert torch.equal(w.detach().cpu(), ref.transpose(1, 2).contiguous())
    print("OK")
    """
)


def _two_gpus():
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


@pytest.mark.skipif(not _two_gpus(), reason = "needs two CUDA devices")
def test_dequant_on_non_current_gpu_matches_current_gpu():
    pytest.importorskip("transformers.integrations.mxfp4")
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER],
        capture_output = True,
        text = True,
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING = "1"),
        timeout = 600,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-4000:]


def test_device_guard_is_a_noop_off_accelerators():
    from unsloth_zoo.temporary_patches.mxfp4 import _device_guard

    with _device_guard(torch.zeros(1)):
        pass
    with _device_guard(torch.zeros(1, device = "meta")):
        pass


@pytest.mark.skipif(not _two_gpus(), reason = "needs two CUDA devices")
def test_device_guard_switches_and_restores_current_device():
    from unsloth_zoo.temporary_patches.mxfp4 import _device_guard

    torch.cuda.set_device(0)
    with _device_guard(torch.zeros(1, device = "cuda:1")):
        assert torch.cuda.current_device() == 1
    assert torch.cuda.current_device() == 0


_SPLIT_MM_RUNNER = textwrap.dedent(
    """
    import torch
    from unsloth_zoo.fp16_emulation import fp16_split_mm

    torch.manual_seed(0)
    A = torch.randn(64, 48, device = "cuda:1") * 0.01
    B = torch.randn(48, 32, device = "cuda:1") * 0.01
    torch.cuda.set_device(0)
    out = fp16_split_mm(A, B)
    torch.cuda.synchronize(1)
    want = A @ B
    assert out.device == A.device
    assert ((out - want).abs().max() / want.abs().max()).item() < 1e-4
    assert torch.cuda.current_device() == 0
    print("OK")
    """
)


@pytest.mark.skipif(not _two_gpus(), reason = "needs two CUDA devices")
def test_fp16_split_mm_on_non_current_gpu():
    proc = subprocess.run(
        [sys.executable, "-c", _SPLIT_MM_RUNNER],
        capture_output = True,
        text = True,
        env = dict(os.environ, CUDA_LAUNCH_BLOCKING = "1"),
        timeout = 600,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-4000:]
