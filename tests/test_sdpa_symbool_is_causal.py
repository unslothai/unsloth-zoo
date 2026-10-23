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

"""A SymBool `is_causal` must not graph break the patched SDPA forward.

Moonshine computes `is_causal = self.is_causal and attention_mask is None and q_len > 1`,
which under dynamic shapes is a SymBool; SDPA only takes a Python bool, so every
decoder attention call graph broke with "TypeError when making fake tensor call".
"""

import os

os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_symbool_is_causal_compiles_fullgraph(monkeypatch):
    import transformers.integrations.sdpa_attention as sdpa_module
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    import unsloth_zoo.temporary_patches.misc as misc

    # The patch is a no-op under UNSLOTH_COMPILE_DISABLE=1, which the unsloth repo's Core zoo
    # job sets for the whole suite; this test is about the patch, so it must not inherit that.
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    # Installed globally; restore both slots so the rest of the worker sees stock SDPA.
    monkeypatch.setattr(sdpa_module, "sdpa_attention_forward", sdpa_module.sdpa_attention_forward)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", ALL_ATTENTION_FUNCTIONS["sdpa"])
    misc.patch_sdpa_bool_causal_mask()

    forward = sdpa_module.sdpa_attention_forward
    assert getattr(forward, "__unsloth_bool_causal_mask_fix__", False), "the patch did not install"

    class Layer(torch.nn.Module):
        is_causal = True

    layer = Layer()

    def attend(q):
        is_causal = layer.is_causal and q.shape[2] > 1
        return forward(layer, q, q, q, None, scaling = 0.125, is_causal = is_causal)[0]

    torch._dynamo.reset()
    compiled = torch.compile(attend, fullgraph = True, dynamic = True, backend = "eager")
    for length in (5, 7, 1):
        q = torch.randn(1, 2, length, 16)
        torch.testing.assert_close(compiled(q), attend(q))
