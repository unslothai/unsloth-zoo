"""weighted_unpermute must be bit-identical to the eager combine, or decline (None)."""
import pytest
import torch

from unsloth_zoo.temporary_patches import moe_triton_kernels as K
from unsloth_zoo.temporary_patches.moe_utils import combine_permuted_moe_outputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not K.moe_triton_kernels_available(torch.device("cuda")),
    reason = "needs the Triton MoE kernels on CUDA",
)


def _eager(y, s, w, T, k):
    return combine_permuted_moe_outputs(y * w.unsqueeze(-1), s, T, k, out_dtype = torch.bfloat16)


def test_hidden_one_report_case():
    # External report: fused 0.125 vs eager 0.25 at T=1, top_k=8, H=1.
    y = torch.tensor([1e8, 1, -1e8, 1, 0, 0, 0, 0], dtype = torch.bfloat16, device = "cuda").view(8, 1)
    w = torch.full((8,), 1 / 8, dtype = torch.bfloat16, device = "cuda")
    s = torch.arange(8, device = "cuda")
    out = K.weighted_unpermute(y, s, w, 1, 8, out_dtype = torch.bfloat16)
    assert out is None or torch.equal(out, _eager(y, s, w, 1, 8))


@pytest.mark.parametrize("H,k", [(1, 4), (1, 8), (513, 5), (513, 6), (2049, 6), (2048, 6), (2816, 8), (2880, 4)])
def test_fused_matches_eager_or_declines(H, k):
    g = torch.Generator(device = "cuda").manual_seed(0)
    T = 512 if H == 1 else 17
    for _ in range(4):
        N = T * k
        mag = torch.pow(10.0, torch.randint(-3, 9, (N, H), generator = g, device = "cuda").float())
        y = (torch.randn(N, H, generator = g, device = "cuda") * mag).to(torch.bfloat16)
        w = torch.rand(N, generator = g, device = "cuda").to(torch.bfloat16)
        s = torch.randint(0, 64, (N,), generator = g, device = "cuda").argsort(stable = True)
        out = K.weighted_unpermute(y, s, w[s], T, k, out_dtype = torch.bfloat16)
        if H % 2 == 0:
            assert out is not None, "even hidden sizes must keep the fused kernel"
        if out is not None:
            assert torch.equal(out, _eager(y, s, w[s], T, k))
