"""Remote Mamba2 torch_forward repairs (nvidia Nemotron-3-Nano-Omni modeling_nemotron_h.py).

The remote naive SSD path reduces the inter-chunk state over the wrong axis (sum(dim=2), must be the source
chunk, dim=3) and floors dt at time_step_min instead of clamping to time_step_limit. The fixture carries the
remote scan code verbatim (tests/_remote_mamba2_fixture.py.txt) and is imported as a transformers_modules
module, like trust_remote_code does. Reference: the exact sequential recurrence in float64. CPU only.
"""
import importlib.util
import os
import sys

import pytest
import torch

FIXTURE = os.path.join(os.path.dirname(__file__), "_remote_mamba2_fixture.py.txt")


def _load_fixture(name):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    source = open(FIXTURE).read()
    module.__file__ = FIXTURE
    sys.modules[name] = module
    exec(compile(source, FIXTURE, "exec"), module.__dict__)
    return module


def _sequential(mixer, x, B, C, dt, floor):
    H, P, G, N = mixer.num_heads, mixer.head_dim, mixer.n_groups, mixer.ssm_state_size
    b, L = x.shape[:2]
    dtp = torch.nn.functional.softplus(dt + mixer.dt_bias)
    if floor:
        dtp = torch.clamp(dtp, mixer.time_step_min)
    A = -torch.exp(mixer.A_log)
    Xh = x.reshape(b, L, H, P)
    Bh = B.reshape(b, L, G, N).repeat_interleave(H // G, 2)
    Ch = C.reshape(b, L, G, N).repeat_interleave(H // G, 2)
    st = torch.zeros(b, H, P, N, dtype=x.dtype)
    ys = []
    for t in range(L):
        st = st * torch.exp(dtp[:, t] * A)[..., None, None] + dtp[:, t][..., None, None] * Xh[:, t][..., None] * Bh[:, t][:, :, None, :]
        ys.append((st * Ch[:, t][:, :, None, :]).sum(-1) + Xh[:, t] * mixer.D[:, None])
    return torch.stack(ys, 1).reshape(b, L, -1)


def _inputs(mixer, L, seed=0):
    g = torch.Generator().manual_seed(seed)
    H, P, G, N = mixer.num_heads, mixer.head_dim, mixer.n_groups, mixer.ssm_state_size
    x = torch.randn(2, L, H * P, generator=g, dtype=torch.float64)
    B = torch.randn(2, L, G * N, generator=g, dtype=torch.float64)
    C = torch.randn(2, L, G * N, generator=g, dtype=torch.float64)
    dt = torch.randn(2, L, H, generator=g, dtype=torch.float64) * 2
    return x, B, C, dt


@pytest.fixture
def repaired():
    module = _load_fixture("transformers_modules.unsloth_test_remote_mamba2")
    cls = module.FakeRemoteMamba2Mixer
    original = cls.torch_forward
    try:
        from unsloth_zoo.temporary_patches.remote_mamba2 import repair_remote_mamba2_torch_forward
    except ImportError:
        repair_remote_mamba2_torch_forward = None
    changed = repair_remote_mamba2_torch_forward(cls) if repair_remote_mamba2_torch_forward else False
    yield cls, original, changed
    cls.torch_forward = original
    sys.modules.pop("transformers_modules.unsloth_test_remote_mamba2", None)


def test_fixture_reproduces_the_remote_defects():
    # Guards the fixture itself: the verbatim remote code is wrong after the first chunk.
    module = _load_fixture("transformers_modules.unsloth_test_remote_mamba2_raw")
    torch.manual_seed(0)
    mixer = module.FakeRemoteMamba2Mixer().double()
    x, B, C, dt = _inputs(mixer, 24)
    with torch.no_grad():
        y = mixer.torch_forward(x, B, C, dt).double()
        ref = _sequential(mixer, x, B, C, dt, floor=True)
    err = (y - ref).norm(dim=-1) / ref.norm(dim=-1)
    assert float(err[:, :mixer.chunk_size].max()) < 1e-5       # first chunk exact
    assert float(err[:, mixer.chunk_size:].max()) > 1e-2       # later chunks wrong
    sys.modules.pop("transformers_modules.unsloth_test_remote_mamba2_raw", None)


@pytest.mark.parametrize("L", [5, 8, 24, 29])
def test_repaired_matches_exact_recurrence(repaired, L):
    cls, _, changed = repaired
    assert changed
    torch.manual_seed(0)
    mixer = cls().double()
    x, B, C, dt = _inputs(mixer, L)
    with torch.no_grad():
        y = mixer.torch_forward(x, B, C, dt).double()
        ref = _sequential(mixer, x, B, C, dt, floor=False)   # clamp to time_step_limit (0, inf): no floor
    assert torch.allclose(y, ref, rtol=1e-5, atol=1e-6), float((y - ref).abs().max())


def test_repair_is_idempotent_and_skips_fixed_code(repaired):
    cls, _, changed = repaired
    from unsloth_zoo.temporary_patches.remote_mamba2 import repair_remote_mamba2_torch_forward
    assert changed and repair_remote_mamba2_torch_forward(cls) is False
    assert cls.torch_forward._unsloth_mamba2_fixed

    class Unrelated(torch.nn.Module):
        def torch_forward(self, x):
            return x.sum(dim=2)
    assert repair_remote_mamba2_torch_forward(Unrelated) is False


def test_kill_switch(monkeypatch):
    try:
        from unsloth_zoo.temporary_patches.remote_mamba2 import repair_remote_mamba2_torch_forward
    except ImportError:
        pytest.skip("repair not present")
    module = _load_fixture("transformers_modules.unsloth_test_remote_mamba2_ks")
    monkeypatch.setenv("UNSLOTH_REMOTE_MAMBA2_FIX", "0")
    assert repair_remote_mamba2_torch_forward(module.FakeRemoteMamba2Mixer) is False
    sys.modules.pop("transformers_modules.unsloth_test_remote_mamba2_ks", None)


def test_hook_repairs_remote_classes_loaded_later(monkeypatch):
    try:
        from unsloth_zoo.temporary_patches import remote_mamba2
    except ImportError:
        pytest.skip("repair not present")
    import transformers.dynamic_module_utils as dynamic_module_utils
    name = "transformers_modules.unsloth_test_remote_mamba2_hook"

    def fake_get_class_in_module(class_name, module_path, **kwargs):
        # What trust_remote_code does: import the remote file as transformers_modules.*, return the class.
        return getattr(_load_fixture(name), class_name)

    monkeypatch.setattr(dynamic_module_utils, "get_class_in_module", fake_get_class_in_module)
    remote_mamba2.patch_remote_mamba2_torch_forward()
    hooked = dynamic_module_utils.get_class_in_module
    assert hooked is not fake_get_class_in_module and hooked._unsloth_remote_mamba2
    remote_mamba2.patch_remote_mamba2_torch_forward()                 # idempotent: no second wrapper
    assert dynamic_module_utils.get_class_in_module is hooked
    try:
        cls = dynamic_module_utils.get_class_in_module("FakeRemoteMamba2Mixer", "unused")
        assert cls.torch_forward._unsloth_mamba2_fixed
        torch.manual_seed(0)
        mixer = cls().double()
        x, B, C, dt = _inputs(mixer, 24)
        with torch.no_grad():
            y = mixer.torch_forward(x, B, C, dt).double()
            ref = _sequential(mixer, x, B, C, dt, floor=False)
        assert torch.allclose(y, ref, rtol=1e-5, atol=1e-6)
    finally:
        sys.modules.pop(name, None)
