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

"""FlashInfer needs to LINK, not merely compile.

The pre-existing guard checks that nvcc and ninja exist. Both do on plenty of
container images that still cannot build FlashInfer, because the link passes
`-lcuda` and therefore needs the driver STUB `libcuda.so` -- not the runtime
`libcuda.so.1` that every machine with a driver has.

Observed on Kaggle's GPU image: nvcc present, ninja present, every .cu file
compiled cleanly for sm_75, and the final step died on

    /usr/bin/ld: cannot find -lcuda

reported as `RuntimeError: Ninja build failed` after minutes of nvcc work. No
environment variable avoids it: `VLLM_USE_FLASHINFER_SAMPLER=0` merely moves
the build from the sampler kernels to the attention kernels, and vLLM exposes
no prefill-specific opt-out.
"""

import os
from unittest import mock

from unsloth_zoo import vllm_utils


def _stub(tmp_path, name = "libcuda.so"):
    path = tmp_path / name
    path.write_bytes(b"")
    return str(path)


def test_a_runtime_libcuda_without_the_stub_is_not_linkable(tmp_path):
    """The exact shape of the image that motivated this: libcuda.so.1 present,
    libcuda.so absent. Every other check passes and the link cannot."""
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    with mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(tmp_path),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is False


def test_the_stub_beside_the_runtime_is_linkable(tmp_path):
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    _stub(tmp_path)
    with mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(tmp_path),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_a_stub_supplied_through_library_path_counts(tmp_path):
    """LIBRARY_PATH is what the linker consults beyond its defaults, so a caller
    who has already supplied a stub there must not be told FlashInfer is
    unavailable. Unsloth's own Kaggle GRPO payload does exactly this, which is
    why that path works where a bare probe does not."""
    supplied = tmp_path / "shim"
    supplied.mkdir()
    _stub(supplied)
    empty = tmp_path / "cuda"
    empty.mkdir()
    with mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": str(supplied)}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_cuda_home_stubs_count(tmp_path):
    root = tmp_path / "cuda"
    stubs = root / "lib64" / "stubs"
    stubs.mkdir(parents = True)
    _stub(stubs)
    empty = tmp_path / "nothing"
    empty.mkdir()
    with mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": str(root), "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_an_empty_library_path_entry_is_not_a_directory(tmp_path):
    """`LIBRARY_PATH=""` splits to [""], and os.path.join("", "libcuda.so") is a
    RELATIVE path -- which exists whenever the process happens to be running in
    a directory that has one. Filtering empties keeps the answer from depending
    on the working directory."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _stub(cwd)
    with mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        here = os.getcwd()
        os.chdir(cwd)
        try:
            assert vllm_utils._can_link_libcuda() is False
        finally:
            os.chdir(here)
