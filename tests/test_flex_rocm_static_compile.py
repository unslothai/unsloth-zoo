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

"""ROCm: transformers' flex_attention is compiled with static shapes. Its default compile recompiles with a symbolic
sequence length at the second length, and that kernel returned inf / NaN gradients on gfx1151 (tiny Gemma-3 trained
at 24 tokens after a 32 token step). Runs on CPU: torch.compile is recorded, never called."""

import pytest
import torch

flex_module = pytest.importorskip("transformers.integrations.flex_attention")
pytest.importorskip("torch.nn.attention.flex_attention")


@pytest.fixture
def patched(monkeypatch):
    """Install the patch on a fresh transformers attribute with torch.compile recorded; yields (records, hip)."""
    from unsloth_zoo.temporary_patches import misc

    records = []

    def fake_compile(fn, **kwargs):
        records.append(kwargs)
        return lambda q, k, v, **kw: ("compiled", kwargs, kw)

    def install(hip):
        monkeypatch.setattr(torch.version, "hip", hip, raising = False)
        monkeypatch.setattr(flex_module, "compile_friendly_flex_attention", ORIGINAL)
        monkeypatch.setattr(torch, "compile", fake_compile)
        getattr(misc, "patch_flex_attention_static_shapes_on_rocm", lambda: None)()
        return flex_module.compile_friendly_flex_attention
    return records, install


ORIGINAL = flex_module.compile_friendly_flex_attention
if getattr(ORIGINAL, "_unsloth_rocm_static", False):  # already patched in this process (a ROCm host)
    ORIGINAL = ORIGINAL.__wrapped__


def _qkv():
    return torch.zeros(1, 1, 4, 8), torch.zeros(1, 1, 4, 8), torch.zeros(1, 1, 4, 8)


def test_rocm_compiles_flex_with_static_shapes(patched):
    records, install = patched
    fn = install("7.13.0")
    assert fn is not ORIGINAL, "no ROCm patch on transformers' compile_friendly_flex_attention"
    tag, kwargs, passed = fn(*_qkv(), training = True, scale = 0.5)
    assert tag == "compiled" and kwargs.get("dynamic") is False
    assert passed == {"scale": 0.5}
    fn(*_qkv(), training = True)
    fn(*_qkv(), training = False)
    assert len(records) == 2, "one compile per training flag, reused across calls"


def test_cuda_and_cpu_keep_transformers_compile(patched):
    _, install = patched
    assert install(None) is ORIGINAL


def test_patch_is_idempotent(patched):
    from unsloth_zoo.temporary_patches import misc

    _, install = patched
    first = install("7.13.0")
    misc.patch_flex_attention_static_shapes_on_rocm()
    assert flex_module.compile_friendly_flex_attention is first
