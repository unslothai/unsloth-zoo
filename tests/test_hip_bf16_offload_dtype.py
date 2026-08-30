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

"""The HIP branch of ``initialize_unsloth_gradient_checkpointing`` must ASK.

It used to read ``SUPPORTS_BFLOAT16 = True`` and pin the CPU offload buffers in
bfloat16 on every AMD host. RDNA 1 and 2 (gfx101x, gfx103x) have no native bf16
arithmetic: Triton selects ``llvm.amdgcn.fdot2.bf16.bf16``, LLVM cannot lower it
for the target, and the process dies with no Python exception, which reaches the
user as "Training process exited unexpectedly" at step 0
(unslothai/unsloth issue 7922, an RX 6600 XT / gfx1032).

Two things this file has to keep straight, because they pull in opposite
directions:

* On a stock ROCm install the change is **inert**. ``torch.cuda.is_bf16_supported``
  returns True for ``torch.version.hip`` before it looks at the architecture at
  all, so an AMD user without unslothai/unsloth#7682 keeps bfloat16 exactly as
  before. A test that only proved "float16 now" would be proving a regression
  for every current AMD user.
* Once that PR patches the probe process-wide, this line inherits the same
  answer as every other bf16 decision instead of disagreeing with them.

``DEVICE_TYPE_TORCH`` maps "hip" to "cuda", so the HIP branch can be driven on
an NVIDIA box by spoofing one module global, and the buffers it allocates are
real ones.
"""

import ast
import inspect

import pytest

torch = pytest.importorskip("torch")

from unsloth_zoo import gradient_checkpointing as gc


def _hip_branch_source():
    """The body of the `elif DEVICE_TYPE == "hip"` arm, as source."""
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
    """Runs anywhere, including the CPU lane, since it only reads the source."""
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


def test_the_probe_is_called_with_no_arguments():
    """`unsloth` replaces this probe, and replacements differ in signature.

    `unsloth/_gpu_init.py` installs one wrapper taking `including_emulation` and
    another taking nothing, choosing by inspecting the original. Calling with no
    arguments is the only form both accept.
    """
    calls = []
    dtype = _run_hip_branch(lambda *args, **kwargs: calls.append((args, kwargs)) or True)
    if dtype is None:
        pytest.skip("needs a device to allocate the buffers")
    assert calls and calls[0] == ((), {}), f"the probe was called as {calls[:1]}"


def _run_hip_branch(probe):
    """Drive the real initializer down its HIP branch; None if no device."""
    if not torch.cuda.is_available():
        return None
    old_device_type = gc.DEVICE_TYPE
    old_probe = torch.cuda.is_bf16_supported
    gc.DEVICE_TYPE = "hip"
    torch.cuda.is_bf16_supported = probe
    try:
        gc.CPU_BUFFERS = []
        gc.initialize_unsloth_gradient_checkpointing()
        return gc.CPU_BUFFERS[0].dtype
    finally:
        gc.DEVICE_TYPE = old_device_type
        torch.cuda.is_bf16_supported = old_probe


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "allocates real device buffers")
def test_a_stock_rocm_host_still_gets_bfloat16():
    """The inertness claim, which is what makes this safe to merge alone.

    `torch.cuda.is_bf16_supported` short-circuits to True on `torch.version.hip`
    before any architecture check, so this is what every AMD host sees today.
    """
    assert _run_hip_branch(lambda *args, **kwargs: True) is torch.bfloat16, (
        "an unpatched AMD host no longer gets bfloat16, so this change is not "
        "inert and it moves the dtype under every current ROCm user"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "allocates real device buffers")
def test_a_patched_probe_saying_no_gets_float16():
    """The case the change exists for: gfx101x/gfx103x, once #7682 lands."""
    assert _run_hip_branch(lambda *args, **kwargs: False) is torch.float16, (
        "the offload buffers are still bfloat16 on a card that cannot do bf16"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "allocates real device buffers")
def test_cuda_still_decides_on_compute_capability_not_the_probe():
    """The other vendors must not move. CUDA reads the capability directly."""
    old_device_type = gc.DEVICE_TYPE
    old_probe = torch.cuda.is_bf16_supported
    gc.DEVICE_TYPE = "cuda"
    # A probe that would flip the answer if CUDA ever started consulting it.
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
    try:
        gc.CPU_BUFFERS = []
        gc.initialize_unsloth_gradient_checkpointing()
        got = gc.CPU_BUFFERS[0].dtype
    finally:
        gc.DEVICE_TYPE = old_device_type
        torch.cuda.is_bf16_supported = old_probe

    major, _ = torch.cuda.get_device_capability()
    expected = torch.bfloat16 if major >= 8 else torch.float16
    assert got is expected, (
        f"CUDA picked {got} on a compute-capability {major}.x card; the HIP "
        "change has leaked into the NVIDIA path"
    )
