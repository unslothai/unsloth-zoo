"""nf4_dequant_triton must match bitsandbytes for 2**31..2**32-weight stacks (int32 2 * n_bytes wrapped)."""
import os

import pytest
import torch

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
bnb = pytest.importorskip("bitsandbytes")

from unsloth_zoo.temporary_patches import moe_utils_bnb4bit as M
from unsloth_zoo.temporary_patches.moe_triton_kernels import moe_triton_kernels_available, nf4_dequant_triton


def _free_gib():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


@pytest.mark.parametrize("shape", [
    (256, 4096, 2048),   # first wrapping size
    (384, 4096, 2048),
    (256, 4096, 4096),   # n_bytes arrives as int64
    (255, 4096, 2048),
])
def test_nf4_dequant_triton_matches_bitsandbytes_large_stacks(shape):
    if not moe_triton_kernels_available(torch.device("cuda")):
        pytest.skip("Triton MoE kernels unavailable")
    numel = shape[0] * shape[1] * shape[2]
    if _free_gib() < numel * 7 / 2**30 + 2:
        pytest.skip("not enough free GPU memory")
    torch.manual_seed(0)
    value = torch.randn(shape, device = "cuda", dtype = torch.bfloat16)
    param = M._make_expert_params4bit(value, requires_grad = False, blocksize = 64, quant_type = "nf4",
                                      compress_statistics = False)
    del value
    triton_out = nf4_dequant_triton(param.data, param.quant_state, getattr(param, "_original_shape", None))
    assert triton_out is not None
    reference = M._dequantize_4bit_in_slices(param)
    if reference is None:  # under 2**31: one bitsandbytes call
        reference = bnb.functional.dequantize_4bit(param.data, param.quant_state).view(shape)
    assert torch.equal(triton_out.view(shape), reference.view(shape))
