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

import os, sys, pytest
os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
import torch
from unsloth_zoo import vllm_utils as vu

@pytest.mark.parametrize("fn", ["_saved_adapter_lora_keys", "_saved_adapter_lora_tensors"])
def test_bin_refused_on_old_torch(tmp_path, monkeypatch, fn):
    torch.save({"x.lora_A.weight": torch.zeros(1)}, tmp_path / "adapter_model.bin")
    calls = []
    monkeypatch.setattr(vu.torch, "__version__", "2.5.1")
    monkeypatch.setattr(vu.torch, "load", lambda *a, **k: calls.append(a) or {})
    with pytest.raises(ValueError, match="cannot load one safely"):
        getattr(vu, fn)(str(tmp_path))
    assert calls == []

def test_bin_still_loads_on_new_torch(tmp_path, monkeypatch):
    torch.save({"x.lora_A.weight": torch.zeros(1)}, tmp_path / "adapter_model.bin")
    monkeypatch.setattr(vu.torch, "__version__", "2.6.0")
    assert vu._saved_adapter_lora_keys(str(tmp_path)) == ["x.lora_A.weight"]
