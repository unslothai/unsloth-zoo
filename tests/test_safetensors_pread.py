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

"""pread_backend_kwargs: CUDA unified-memory GPUs get pread, discrete GPUs (Linux and Windows) keep mmap, the env
switch wins, safetensors < 0.8.0 keeps mmap, and the unified-memory check never creates a CUDA context."""

import sys
import types

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from unsloth_zoo import safetensors_pread as sp  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.PREAD_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_UMA", raising = False)
    monkeypatch.setattr(sp, "_UMA", None)
    detect = sp.safetensors_supports_backend
    yield
    detect.cache_clear()


@pytest.mark.parametrize(
    "platform, env, uma, supported, device, expected",
    [
        ("linux", None, "0", True, "cpu", {}),
        ("linux", None, "1", True, "cpu", {"backend": "pread"}),
        ("linux", None, "1", False, "cpu", {}),
        ("linux", None, "1", True, "cuda:0", {}),
        ("linux", "0", "1", True, "cpu", {}),
        ("linux", "1", "0", True, "cpu", {"backend": "pread"}),
        # discrete Windows keeps mmap (pread is slower there)
        ("win32", None, "0", True, "cpu", {}),
        ("win32", None, "1", True, "cpu", {"backend": "pread"}),
        ("win32", "off", "1", True, "cpu", {}),
    ],
)
def test_pread_backend_kwargs(monkeypatch, platform, env, uma, supported, device, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("UNSLOTH_FORCE_UMA", uma)
    if env is not None:
        monkeypatch.setenv(sp.PREAD_ENV, env)
    monkeypatch.setattr(sp, "safetensors_supports_backend", lambda: supported)
    assert sp.pread_backend_kwargs(device) == expected


def test_safetensors_without_backend_keeps_mmap(monkeypatch):
    import safetensors._safetensors_rust as rust

    # safetensors < 0.8.0's extension class, as far as detection can see
    monkeypatch.setattr(rust, "safe_open", types.SimpleNamespace(__text_signature__ = "(filename, framework, device=...)"))
    sp.safetensors_supports_backend.cache_clear()
    assert sp.safetensors_supports_backend() is False
    monkeypatch.setenv(sp.PREAD_ENV, "1")
    assert sp.pread_backend_kwargs() == {}


def test_installed_safetensors_detection_matches_the_keyword(tmp_path):
    from safetensors.torch import save_file

    path = str(tmp_path / "x.safetensors")
    save_file({"w": torch.arange(6, dtype = torch.float32)}, path)
    try:
        safetensors.safe_open(path, framework = "pt", device = "cpu", backend = "pread")
        accepts = True
    except TypeError:
        accepts = False
    assert sp.safetensors_supports_backend() is accepts


def test_detection_sees_through_a_wrapped_safe_open(monkeypatch, tmp_path):
    """Another library wrapping safetensors.safe_open as (*args, **kwargs) must not hide the keyword."""
    from safetensors.torch import save_file

    real = safetensors.safe_open
    path = str(tmp_path / "x.safetensors")
    save_file({"w": torch.arange(4, dtype = torch.float32)}, path)
    try:
        real(path, framework = "pt", device = "cpu", backend = "pread")
        accepts = True
    except TypeError:
        accepts = False
    monkeypatch.setattr(safetensors, "safe_open", lambda *a, **k: real(*a, **k))
    sp.safetensors_supports_backend.cache_clear()
    assert sp.safetensors_supports_backend() is accepts


def test_unified_memory_check_never_initialises_cuda():
    if torch.cuda.is_initialized():
        pytest.skip("CUDA already initialised in this process")
    assert sp._unified_memory_now() is False
    assert not torch.cuda.is_initialized()
    assert sp._UMA is None
