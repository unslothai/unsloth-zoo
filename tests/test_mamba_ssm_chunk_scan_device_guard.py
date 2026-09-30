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

"""mamba_ssm `_chunk_scan_fwd` must launch on its inputs' device. Needs 2 GPUs."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason = "needs 2 CUDA GPUs",
)


def _apply_zoo_patch():
    from unsloth_zoo.temporary_patches import misc

    fn = getattr(misc, "patch_mamba_ssm_chunk_scan_device_guard", None)
    if fn is not None:
        fn()


def _inputs(device, seed = 0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator = g).to(device, torch.bfloat16)

    b, l, h, p, n = 2, 512, 8, 64, 64
    return dict(
        x = r(b, l, h, p),
        dt = r(b, l, h).abs(),
        A = -torch.rand(h, generator = g).to(device),
        B = r(b, l, 1, n),
        C = r(b, l, 1, n),
        D = torch.randn(h, generator = g).to(device),
        dt_bias = torch.randn(h, generator = g).to(device),
    )


def _scan(kw):
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    return mamba_chunk_scan_combined(
        kw["x"], kw["dt"], kw["A"], kw["B"], kw["C"], chunk_size = 128,
        D = kw["D"], dt_bias = kw["dt_bias"], dt_softplus = True,
    )


def test_chunk_scan_launches_on_input_device():
    pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    _apply_zoo_patch()
    from mamba_ssm.ops.triton import ssd_chunk_scan

    real = ssd_chunk_scan._chunk_scan_fwd_kernel
    seen = []

    class Spy:
        def __getitem__(self, grid):
            launch = real[grid]

            def run(*args, **kwargs):
                seen.append(torch.cuda.current_device())
                return launch(*args, **kwargs)
            return run

    ssd_chunk_scan._chunk_scan_fwd_kernel = Spy()
    try:
        torch.cuda.set_device(0)
        # Enable peer access like a device_map model; else the bug raises instead of racing.
        torch.ones(1, device = "cuda:0").to("cuda:1")
        out = _scan(_inputs("cuda:1"))
        torch.cuda.synchronize(1)
    finally:
        ssd_chunk_scan._chunk_scan_fwd_kernel = real

    assert seen, "_chunk_scan_fwd_kernel was not launched"
    assert seen == [1] * len(seen), \
        f"_chunk_scan_fwd_kernel launched on device(s) {seen} for inputs on cuda:1"
    ref = _scan(_inputs("cuda:0"))
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol = 0, atol = 0)


def test_patch_is_idempotent():
    pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")
    from unsloth_zoo.temporary_patches.misc import patch_mamba_ssm_chunk_scan_device_guard
    from mamba_ssm.ops.triton import ssd_chunk_scan, ssd_combined

    patch_mamba_ssm_chunk_scan_device_guard()
    first = ssd_chunk_scan._chunk_scan_fwd
    patch_mamba_ssm_chunk_scan_device_guard()
    assert ssd_chunk_scan._chunk_scan_fwd is first
    assert getattr(first, "_unsloth_device_guarded", False)
    assert ssd_combined._chunk_scan_fwd is first
