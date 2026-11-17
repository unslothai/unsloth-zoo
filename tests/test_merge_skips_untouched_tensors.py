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

"""The in-place 16-bit merge must leave tensors nothing replaces unread and unwritten."""
from __future__ import annotations

import pytest

import _merge_e2e_helpers as H


@pytest.mark.parametrize("scenario", ["attn_only", "mlp_only"])
def test_merge_writes_only_adapted_tensors(scenario, tmp_path, monkeypatch):
    if not H.family_available("llama"):
        pytest.skip(reason = "tiny llama config cannot be built by the installed transformers")
    import unsloth_zoo.saving_utils as sv

    written = []
    real_write = sv._write_tensor_direct_torch

    def write(mm, header_metadata, length_of_header, output_key, tensor, output_dtype):
        written.append(output_key)
        return real_write(mm, header_metadata, length_of_header, output_key, tensor, output_dtype)

    monkeypatch.setattr(sv, "_write_tensor_direct_torch", write)
    n_adapted, n_pass = H.run_case("llama", scenario, str(tmp_path))
    assert n_adapted >= 1 and n_pass >= 1

    marker = ".self_attn." if scenario == "attn_only" else ".mlp."
    # Vocab layers carry the live embedding rows (new tokens), so they are rewritten.
    vocab = ("embed_tokens.weight", "lm_head.weight")
    untouched = sorted(k for k in written if marker not in k and not k.endswith(vocab))
    assert written, "the merge wrote nothing"
    assert untouched == [], f"untouched tensors were rewritten: {untouched}"
