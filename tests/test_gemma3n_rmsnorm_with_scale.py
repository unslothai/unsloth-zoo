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

import pytest
import torch

gemma3n = pytest.importorskip("transformers.models.gemma3n.modeling_gemma3n")
from unsloth_zoo.temporary_patches.gemma3n import Gemma3nRMSNorm_forward


def _norm(with_scale, legacy_buffer = False):
    norm = gemma3n.Gemma3nRMSNorm(16, eps = 1e-6, with_scale = with_scale)
    if with_scale:
        with torch.no_grad():
            norm.weight.normal_(1.0, 0.1)
    elif legacy_buffer and not hasattr(norm, "weight"):
        # transformers <= 5.4 registered this placeholder; 5.5.0 dropped it.
        norm.register_buffer("weight", torch.tensor(1.0), persistent = False)
    return norm


@pytest.mark.parametrize(
    "with_scale, legacy_buffer",
    [(True, False), (False, False), (False, True)],
    ids = ["with_scale", "no_scale", "no_scale_legacy_buffer"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_patched_norm_matches_transformers(with_scale, legacy_buffer, dtype):
    norm = _norm(with_scale, legacy_buffer)
    x = torch.randn(3, 5, 16).to(dtype)
    torch.testing.assert_close(Gemma3nRMSNorm_forward(norm, x), norm(x))


def test_the_embedder_post_projection_norm_has_no_scale():
    # The norm that crashed: built with with_scale=False by Gemma3nMultimodalEmbedder.
    source = open(gemma3n.__file__, encoding = "utf-8").read()
    assert "self.embedding_post_projection_norm = Gemma3nRMSNorm(" in source
    line = next(l for l in source.splitlines() if "self.embedding_post_projection_norm = Gemma3nRMSNorm(" in l)
    assert "with_scale=False" in line.replace(" ", "")
