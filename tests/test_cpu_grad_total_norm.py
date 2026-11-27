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

"""CPU grad clipping: large float32 gradients get an accurate row-wise total norm; everything else is torch's."""

import pytest
import torch
from torch.nn.utils import clip_grad

ops = pytest.importorskip("unsloth_zoo.cpu.ops")

if not hasattr(clip_grad, "_get_total_norm"):
    pytest.skip("torch < 2.6 has no clip_grad._get_total_norm", allow_module_level = True)


@pytest.fixture
def patched(monkeypatch):
    original = getattr(clip_grad._get_total_norm, "_unsloth_original", clip_grad._get_total_norm)
    monkeypatch.setattr(clip_grad, "_get_total_norm", original)
    assert ops.patch_cpu_grad_total_norm() is True
    yield clip_grad._get_total_norm
    # monkeypatch restores torch's own function


def _grads(seed = 0):
    g = torch.Generator().manual_seed(seed)
    # one big tensor (the embedding-like case) plus small ones that stay on torch's path
    return [torch.randn(4096, 1024, generator = g) * 1e-3, torch.randn(1024, generator = g), torch.randn(64, 32, generator = g)]


def _ref(tensors):
    return torch.stack([t.double().norm() for t in tensors]).norm().item()


def test_installs_and_is_idempotent(patched):
    assert getattr(patched, "_unsloth_cpu", False)
    assert ops.patch_cpu_grad_total_norm() is True
    assert clip_grad._get_total_norm is patched


def test_large_float32_norm_is_accurate(patched):
    g = torch.Generator().manual_seed(1)
    big = torch.randn(151936 // 8, 1024, generator = g) * 1e-3  # 19M elements, keeps the test fast
    ref = _ref([big])
    got = patched([big]).item()
    assert abs(got - ref) / ref < 1e-6
    torch_err = abs(patched._unsloth_original([big]).item() - ref) / ref
    assert abs(got - ref) <= abs(torch_err * ref)  # never worse than torch's own norm


def test_matches_reference_on_mixed_sizes(patched):
    tensors = _grads()
    got = patched(tensors)
    assert got.dtype == torch.float32
    assert abs(got.item() - _ref(tensors)) / _ref(tensors) < 1e-6


def test_clip_grad_norm_end_to_end(patched):
    params = [torch.nn.Parameter(torch.zeros_like(t)) for t in _grads()]
    for p, t in zip(params, _grads()):
        p.grad = t.clone()
    total = torch.nn.utils.clip_grad_norm_(params, max_norm = 0.5)
    assert abs(total.item() - _ref(_grads())) / _ref(_grads()) < 1e-6
    clipped = _ref([p.grad for p in params])
    assert abs(clipped - 0.5) / 0.5 < 1e-5


@pytest.mark.parametrize("case", ["inf_norm", "one_norm", "bf16_only", "small_only", "one_dim_big"])
def test_other_inputs_keep_torch_path(patched, case):
    g = torch.Generator().manual_seed(2)
    big = torch.randn(2048, 1024, generator = g)
    norm_type = 2.0
    if case == "inf_norm":
        tensors, norm_type = [big], float("inf")
    elif case == "one_norm":
        tensors, norm_type = [big], 1.0
    elif case == "bf16_only":
        tensors = [big.bfloat16()]
    elif case == "small_only":
        tensors = [torch.randn(512, 512, generator = g)]
    else:
        tensors = [big.reshape(-1)]
    expected = patched._unsloth_original(tensors, norm_type)
    got = patched(tensors, norm_type)
    assert got.dtype == expected.dtype
    assert torch.equal(got, expected)


def test_foreach_true_is_torch_own_path(patched):
    try:
        expected = patched._unsloth_original(_grads(), 2.0, False, True)
    except RuntimeError:  # torch versions without CPU foreach norms refuse foreach=True
        with pytest.raises(RuntimeError):
            patched(_grads(), 2.0, False, True)
        return
    assert torch.equal(patched(_grads(), 2.0, False, True), expected)


def test_nonfinite_raises_when_asked(patched):
    tensors = _grads()
    tensors[0][0, 0] = float("nan")
    with pytest.raises(RuntimeError, match = "non-finite"):
        patched(tensors, 2.0, True)
    assert torch.isnan(patched(tensors, 2.0, False))


def test_mixed_dtypes_promote_like_torch(patched):
    tensors = [_grads()[0], torch.randn(64, dtype = torch.bfloat16)]
    assert patched(tensors).dtype == patched._unsloth_original(tensors).dtype


def test_non_contiguous_gradient(patched):
    big = _grads()[0].t()  # transposed view
    assert abs(patched([big]).item() - _ref([big])) / _ref([big]) < 1e-6


def test_no_autograd_graph(patched):
    big = _grads()[0].requires_grad_()
    assert not patched([big]).requires_grad


def test_installed_by_cpu_defaults_and_kill_switch():
    import subprocess, sys, os
    code = ("import torch\nfrom torch.nn.utils import clip_grad\nfrom unsloth_zoo.cpu import trainer\n"
            "trainer.install_cpu_training_defaults()\nprint(getattr(clip_grad._get_total_norm, '_unsloth_cpu', False))\n")
    pytest.importorskip("transformers")
    outs = {}
    for flag in ("1", "0"):
        env = dict(os.environ, UNSLOTH_CPU_GRAD_NORM = flag)
        r = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600)
        assert r.returncode == 0, r.stderr[-2000:]
        outs[flag] = r.stdout.strip().splitlines()[-1]
    assert outs == {"1": "True", "0": "False"}


def test_complex_gradient_keeps_torch_path(patched):
    tensors = [_grads()[0], torch.randn(64, dtype = torch.complex64)]
    expected = patched._unsloth_original(tensors)
    got = patched(tensors)
    assert not got.is_complex() and got.dtype == expected.dtype
    assert torch.equal(got, expected)
    params = [torch.nn.Parameter(torch.zeros_like(t)) for t in tensors]
    for p, t in zip(params, tensors):
        p.grad = t.clone()
    torch.nn.utils.clip_grad_norm_(params, max_norm = 0.5)  # clamps a real coefficient, as without the patch
