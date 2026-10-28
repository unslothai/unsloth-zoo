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

"""HIP branch of initialize_unsloth_gradient_checkpointing must ask a probe, not hardcode bf16 (RDNA 1/2 lack native
bf16 and die in LLVM, unslothai/unsloth issue 7922); inert until unslothai/unsloth 7682 since is_bf16_supported is
True for HIP before arch checks; CPU runs stub device_count to 0."""
import ast
import inspect

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import gradient_checkpointing as gc

_MISSING = object()


def _hip_branch_source():
    tree = ast.parse(inspect.getsource(gc.initialize_unsloth_gradient_checkpointing).lstrip())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and getattr(test.left, "id", None) == "DEVICE_TYPE"
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "hip"
        ):
            return node.body
    raise AssertionError("no `DEVICE_TYPE == \"hip\"` branch; this guard has gone vacuous")


def test_the_hip_branch_asks_rather_than_hardcoding_true():
    body = _hip_branch_source()

    assigned = [
        node.value
        for node in ast.walk(ast.Module(body = body, type_ignores = []))
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "SUPPORTS_BFLOAT16" for t in node.targets)
    ]
    assert assigned, "the HIP branch no longer sets SUPPORTS_BFLOAT16"
    assert not any(
        isinstance(value, ast.Constant) and value.value is True for value in assigned
    ), (
        "the HIP branch hardcodes bf16 support again, so an RDNA 1/2 card gets "
        "bfloat16 offload buffers and dies in the Triton bf16 dot intrinsic"
    )
    assert any(isinstance(value, ast.Call) for value in assigned), (
        "the HIP branch does not call anything, so it cannot be asking"
    )


def _globals_the_init_writes():
    """Read rather than list them, so the restore cannot fall behind a new one."""
    tree = ast.parse(inspect.getsource(gc.initialize_unsloth_gradient_checkpointing).lstrip())
    return sorted(
        {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names}
    )


@pytest.fixture
def run_init(monkeypatch):
    """Runs the real init on CPU, unwrapping torch.empty minus the CUDA pin so the dtype stays the module's choice;
    restores every global it writes (USE_UNSLOTH_GC would leak)."""
    real_empty = torch.empty
    saved = {name: getattr(gc, name, _MISSING) for name in _globals_the_init_writes()}

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    def run(device_type, probe, capability = (8, 0)):
        monkeypatch.setenv("UNSLOTH_DISABLE_DOUBLE_BUFFER", "1")
        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", probe)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
        monkeypatch.setattr(gc, "DEVICE_TYPE", device_type)
        monkeypatch.setattr(gc, "DEVICE_TYPE_TORCH", "cuda")
        gc.CPU_BUFFERS = []
        gc.initialize_unsloth_gradient_checkpointing()
        return gc.CPU_BUFFERS[0].dtype

    yield run

    for name, value in saved.items():
        if value is _MISSING:
            delattr(gc, name)
        else:
            setattr(gc, name, value)


def test_the_probe_is_called_with_no_arguments(run_init):
    """`_gpu_init.py` installs two wrappers; no-args is the only form both accept."""
    calls = []
    run_init("hip", lambda *args, **kwargs: calls.append((args, kwargs)) or True)
    assert calls and calls[0] == ((), {}), f"the probe was called as {calls[:1]}"


def test_a_stock_rocm_host_still_gets_bfloat16(run_init):
    """The inertness claim, which is what makes this safe to merge alone."""
    assert run_init("hip", lambda *a, **k: True) is torch.bfloat16, (
        "an unpatched AMD host no longer gets bfloat16, so this change is not "
        "inert and it moves the dtype under every current ROCm user"
    )


def test_a_patched_probe_saying_no_gets_float16(run_init):
    assert run_init("hip", lambda *a, **k: False) is torch.float16, (
        "the offload buffers are still bfloat16 on a card that cannot do bf16"
    )


@pytest.mark.parametrize(
    "capability,expected",
    [((8, 0), torch.bfloat16), ((7, 5), torch.float16)],
)
def test_cuda_still_decides_on_compute_capability_not_the_probe(
    run_init, capability, expected
):
    """The probe answers the opposite of compute capability in both rows, so a CUDA branch that consults the probe fails."""
    got = run_init("cuda", lambda *a, **k: capability[0] < 8, capability = capability)
    assert got is expected, (
        f"CUDA picked {got} on a compute-capability {capability[0]}.x card; the "
        "HIP change has leaked into the NVIDIA path"
    )


def test_xpu_is_untouched(run_init):
    assert run_init("xpu", lambda *a, **k: False) is torch.bfloat16
