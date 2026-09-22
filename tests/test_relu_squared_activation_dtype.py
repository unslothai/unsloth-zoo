"""`relu2` keeps its input dtype under autocast.

transformers spells it `torch.square(relu(x))`; `square` is on autocast's float32 list, so
a bf16 Nemotron-H expert handed float32 to the down projection and to the routed-expert
combine. The Nemotron-H hub modeling code accumulates that combine in the router's dtype
with `index_add_`, which then fails with "self (BFloat16) and source (Float) must have the
same scalar type". The patched form is the same product rounded once at the width the down
projection would have cast it to.
"""
import pytest
import torch


def _patched_activation():
    from unsloth_zoo.temporary_patches.misc import patch_relu_squared_activation_dtype
    patch_relu_squared_activation_dtype()
    from transformers.activations import ReLUSquaredActivation, ACT2FN
    assert isinstance(ACT2FN["relu2"], ReLUSquaredActivation)
    return ACT2FN["relu2"]


def test_patch_is_idempotent_and_marks_the_class():
    act = _patched_activation()
    from unsloth_zoo.temporary_patches.misc import patch_relu_squared_activation_dtype
    forward = type(act).forward
    patch_relu_squared_activation_dtype()
    assert type(act).forward is forward
    assert getattr(type(act), "_unsloth_dtype_patched", False) is True


def test_keeps_dtype_outside_autocast_and_matches_square():
    act = _patched_activation()
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.randn(64, 32, dtype = dtype) * 4
        out = act(x)
        assert out.dtype == dtype
        assert torch.equal(out, torch.square(torch.relu(x)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "autocast on CUDA")
def test_bf16_under_autocast_stays_bf16_and_equals_the_rounded_reference():
    act = _patched_activation()
    x = (torch.randn(2048, 512, device = "cuda", dtype = torch.bfloat16) * 4).requires_grad_()
    with torch.autocast("cuda", dtype = torch.bfloat16):
        out = act(x)
    assert out.dtype == torch.bfloat16
    reference = torch.square(torch.relu(x.detach().float())).to(torch.bfloat16)
    assert torch.equal(out.detach(), reference)
    out.float().sum().backward()
    assert torch.equal(x.grad, (2 * torch.relu(x.detach())).to(x.grad.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "autocast on CUDA")
def test_nemotron_h_style_combine_no_longer_fails_under_autocast():
    act = _patched_activation()
    up = torch.nn.Linear(32, 64, bias = False).cuda().to(torch.bfloat16)
    down = torch.nn.Linear(64, 32, bias = False).cuda().to(torch.bfloat16)
    x = torch.randn(16, 32, device = "cuda", dtype = torch.bfloat16)
    router_w = torch.randn(16, 1, device = "cuda", dtype = torch.bfloat16)
    with torch.autocast("cuda", dtype = torch.bfloat16):
        out = torch.zeros_like(x, dtype = router_w.dtype)
        expert = down(act(up(x))) * router_w
        out.index_add_(0, torch.arange(16, device = "cuda"), expert)  # raised before the patch
    assert out.dtype == torch.bfloat16
