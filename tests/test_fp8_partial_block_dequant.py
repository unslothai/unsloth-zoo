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

"""FP8 weights with a partial last block dequantize with the configured block size (transformers#50603)."""

import types

import pytest
import torch

fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
if not hasattr(getattr(fp8, "Fp8Dequantize", None), "_dequantize_one"):
    pytest.skip("this transformers has no Fp8Dequantize._dequantize_one (no CPU FP8 dequant)", allow_module_level = True)


def _op(block):
    config = types.SimpleNamespace(weight_block_size = block)
    return fp8.Fp8Dequantize(types.SimpleNamespace(quantization_config = config))


def _case(rows, cols, block, lead = ()):
    torch.manual_seed(0)
    w = torch.randint(-4, 5, (*lead, rows, cols)).float().to(torch.float8_e4m3fn)  # exactly representable
    grid = (-(-rows // block[0]), -(-cols // block[1]))
    # Neighbouring blocks get distinct power-of-two scales, so a wrong block size cannot match by chance.
    s = 2.0 ** (torch.arange(int(torch.Size((*lead, *grid)).numel())) % 5 - 2).float().reshape(*lead, *grid)
    ref = w.float() * s.repeat_interleave(block[0], -2)[..., :rows, :].repeat_interleave(block[1], -1)[..., :cols]
    return w, s, ref


@pytest.fixture(autouse = True)
def _patched():
    import unsloth_zoo.cpu  # noqa: F401  (installs the device-independent guards)


def test_the_cpu_backend_installs_the_patch():
    assert getattr(fp8.Fp8Dequantize._dequantize_one, "_unsloth_partial_block_patched", False)


@pytest.mark.parametrize(
    "rows, cols, lead",
    [(200, 200, ()), (100, 200, ()), (130, 256, ()), (200, 384, (3,))],
    ids = ["both-partial", "cols-partial", "rows-partial", "experts"],
)
def test_partial_edge_blocks_use_the_configured_block(rows, cols, lead):
    w, s, ref = _case(rows, cols, [128, 128], lead)
    out = _op([128, 128])._dequantize_one(w, s, torch.float32)
    assert out.shape == ref.shape and torch.equal(out, ref)


def test_whole_blocks_and_other_grids_keep_the_stock_path():
    original = getattr(fp8.Fp8Dequantize._dequantize_one, "_unsloth_original", fp8.Fp8Dequantize._dequantize_one)
    w, s, ref = _case(256, 384, [128, 128])
    op = _op([128, 128])
    assert torch.equal(op._dequantize_one(w, s, torch.float32), original(op, w, s, torch.float32))
    assert torch.equal(op._dequantize_one(w, s, torch.float32), ref)
    # A grid the configured block does not explain (MXFP4-style [1, 32] experts) is left to the stock derivation.
    w, s, _ = _case(64, 64, [1, 32])
    assert torch.equal(op._dequantize_one(w, s, torch.float32), original(op, w, s, torch.float32))
