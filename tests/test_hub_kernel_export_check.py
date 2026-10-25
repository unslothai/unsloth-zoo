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

"""Kernel-hub fallback wrappers keep their kernel under torch.compile and fall back only under torch.export."""
import sys
import types

import pytest
import torch

hub_kernels = pytest.importorskip("transformers.integrations.hub_kernels")
if not hasattr(hub_kernels, "use_kernel_func_from_hub_with_fallback"):
    pytest.skip("transformers without kernel-hub fallback wrappers", allow_module_level=True)

from unsloth_zoo.temporary_patches import hub_kernel_export_check as patch_module

CALLS = {"kernel": 0, "reference": 0}


@torch.compiler.disable
def _kernel(x):
    CALLS["kernel"] += 1
    return x * 2


def _build():
    package = types.ModuleType("_unsloth_fake_kernel_pkg")
    package.fake_kernel = _kernel
    sys.modules[package.__name__] = package

    @hub_kernels.use_kernel_func_from_hub_with_fallback("fake_kernel", package.__name__)
    def reference(x):
        CALLS["reference"] += 1
        return x * 2

    return reference


def _run(wrapper, compiled):
    for key in CALLS:
        CALLS[key] = 0
    torch._dynamo.reset()
    fn = (lambda x: wrapper(x + 1) - 1)
    if compiled:
        fn = torch.compile(fn, backend="eager", fullgraph=False)
    for _ in range(3):
        out = fn(torch.ones(4))
    assert torch.equal(out, torch.full((4,), 3.0))
    return dict(CALLS)


def test_patch_applies_only_where_dynamo_folds_is_exporting():
    applied = patch_module.patch_hub_kernel_export_check()
    folded = patch_module._exporting_folded_in_compile()
    assert applied == (folded and isinstance(getattr(torch.compiler, "_is_exporting_flag", None), bool))
    swapped = getattr(hub_kernels.is_torchdynamo_exporting, patch_module._MARK, False)
    assert swapped == applied
    # Idempotent.
    assert patch_module.patch_hub_kernel_export_check() == applied
    assert getattr(hub_kernels.is_torchdynamo_exporting, patch_module._MARK, False) == applied


def test_compiled_caller_reaches_the_kernel():
    patch_module.patch_hub_kernel_export_check()
    wrapper = _build()
    assert _run(wrapper, compiled=False) == {"kernel": 3, "reference": 0}
    assert _run(wrapper, compiled=True) == {"kernel": 3, "reference": 0}


@pytest.mark.parametrize("strict", [False, True])
def test_export_still_uses_the_reference(strict):
    patch_module.patch_hub_kernel_export_check()
    wrapper = _build()

    class M(torch.nn.Module):
        def forward(self, x):
            return wrapper(x + 1) - 1

    for key in CALLS:
        CALLS[key] = 0
    torch._dynamo.reset()
    exported = torch.export.export(M(), (torch.ones(4),), strict=strict)
    # Strict export traces the reference through Dynamo, so its Python counter need not tick.
    assert CALLS["kernel"] == 0 and (strict or CALLS["reference"] >= 1)
    assert torch.equal(exported.module()(torch.ones(4)), torch.full((4,), 3.0))
