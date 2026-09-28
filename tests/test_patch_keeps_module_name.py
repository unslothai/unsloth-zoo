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

"""Patches that exec `from X import (...)` into a module's globals must not import dunders.

`patch_merge_quantization_configs` imported every name of transformers.quantizers.auto that
appears in the rewritten source, and the source says `__class__.__name__`, so misc.py's own
`__name__` became "transformers.quantizers.auto". Every function defined there afterwards
carried that __module__, and torch.compile(fullgraph = True) through the SDPA wrapper died with

    InternalTorchDynamoError: AttributeError: module 'transformers.quantizers.auto' has no attribute 'torch'
"""

import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_merge_quantization_patch_keeps_misc_module_name():
    import unsloth_zoo.temporary_patches.misc as misc

    misc.patch_merge_quantization_configs()
    assert misc.__dict__["__name__"] == "unsloth_zoo.temporary_patches.misc"
    for name in ("__file__", "__spec__", "__loader__", "__package__"):
        value = misc.__dict__.get(name)
        assert value is None or "quantizers" not in str(getattr(value, "name", value)), name


def test_sdpa_wrapper_compiles_fullgraph():
    import unsloth_zoo.temporary_patches.misc as misc

    misc.patch_merge_quantization_configs()
    misc.patch_sdpa_bool_causal_mask()
    import transformers.integrations.sdpa_attention as sdpa_module

    forward = sdpa_module.sdpa_attention_forward
    assert forward.__module__ != "transformers.quantizers.auto"

    class Layer(torch.nn.Module):
        is_causal = True

    layer = Layer()
    torch._dynamo.reset()
    compiled = torch.compile(lambda q, k, m: forward(layer, q, k, k, m, scaling = 0.125)[0], fullgraph = True, backend = "eager")
    q = torch.randn(1, 2, 1, 64)
    k = torch.randn(1, 2, 16, 64)
    mask = torch.ones(1, 1, 1, 16, dtype = torch.bool)
    out = compiled(q, k, mask)
    torch.testing.assert_close(out, forward(layer, q, k, k, mask, scaling = 0.125)[0])
