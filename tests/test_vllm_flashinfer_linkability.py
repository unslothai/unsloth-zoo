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

The pre-existing guard checks nvcc and ninja exist. Both do on container images
that still cannot build FlashInfer, because the link passes `-lcuda` and so
needs the driver STUB `libcuda.so`, not the runtime `libcuda.so.1` that every
machine with a driver has.

Observed on Kaggle's GPU image: nvcc present, ninja present, every .cu file
compiled cleanly for sm_75, and the final step died on

    /usr/bin/ld: cannot find -lcuda

reported as `RuntimeError: Ninja build failed` after minutes of nvcc work. No
env var avoids it: `VLLM_USE_FLASHINFER_SAMPLER=0` merely moves the build from
the sampler kernels to the attention kernels, and vLLM exposes no
prefill-specific opt-out.
"""

import contextlib
import os
import platform
import subprocess
import sys
import sysconfig
from unittest import mock

import pytest

from unsloth_zoo import vllm_utils


def _stub(tmp_path, name = "libcuda.so"):
    path = tmp_path / name
    path.write_bytes(b"")
    return str(path)


@contextlib.contextmanager
def _linux_with_no_ambient_cuda():
    """Everything below asserts LINUX semantics, so pin the platform and
    neutralise the two HOST-derived sources: the nvcc-inferred CUDA root and
    the linker's own defaults.

    Both halves are load-bearing, and the platform pin was learned the hard
    way. On macOS and Windows `_can_link_libcuda` returns before any of this
    runs, so unpinned these tests silently answered a different question: on a
    macOS runner the two negative assertions went red (darwin returns True) and
    the positive ones went green for the wrong reason. A test that passes
    because the function short-circuited is worse than one that fails.
    """
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(vllm_utils, "_cuda_roots_from_nvcc", lambda: ()), \
         mock.patch.object(vllm_utils, "_linker_default_dirs", lambda: ()):
        yield


def test_a_runtime_libcuda_without_the_stub_is_not_linkable(tmp_path):
    """The image that motivated this: libcuda.so.1 present, libcuda.so absent.
    Every other check passes and the link still cannot."""
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    with _linux_with_no_ambient_cuda(), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(tmp_path),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is False


def test_the_stub_beside_the_runtime_is_linkable(tmp_path):
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    _stub(tmp_path)
    with _linux_with_no_ambient_cuda(), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(tmp_path),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_a_stub_supplied_through_library_path_counts(tmp_path):
    """LIBRARY_PATH is what the linker consults beyond its defaults, so a caller
    who supplied a stub there must not be told FlashInfer is unavailable.
    Unsloth's own Kaggle GRPO payload does exactly this."""
    supplied = tmp_path / "shim"
    supplied.mkdir()
    _stub(supplied)
    empty = tmp_path / "cuda"
    empty.mkdir()
    with _linux_with_no_ambient_cuda(), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
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
    with _linux_with_no_ambient_cuda(), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": str(root), "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_an_empty_library_path_entry_is_not_a_directory(tmp_path):
    """`LIBRARY_PATH=""` splits to [""], and os.path.join("", "libcuda.so") is
    RELATIVE, so it exists whenever THIS process runs in a directory that has
    one. GCC does treat an empty component as the current directory, but the
    one that matters is not this one: FlashInfer links via `ninja -C
    <build_dir>` with `cwd=<build_dir>`, so `c++` resolves a relative -L
    against its JIT cache directory. Honouring empties would answer a question
    about the wrong directory."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _stub(cwd)
    with _linux_with_no_ambient_cuda(), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        here = os.getcwd()
        os.chdir(cwd)
        try:
            assert vllm_utils._can_link_libcuda() is False
        finally:
            os.chdir(here)


# ------------------------------------------- false negatives are the bad kind
#
# False when the link would have succeeded silently disables FlashInfer on a
# machine where it works; True when the link then fails merely restores the
# pre-check behaviour. The two tests below cover the two ways the first,
# dangerous mistake was reachable.


def test_a_stub_in_a_default_linker_directory_counts(tmp_path):
    """`c++ ... -lcuda` searches ld's built-in SEARCH_DIRs with no -L at all,
    and the NVIDIA driver installer puts the unversioned libcuda.so symlink in
    one of them (/usr/lib/x86_64-linux-gnu on Debian/Ubuntu). A toolkit with no
    stubs directory still links there, so False would disable a working
    FlashInfer."""
    default = tmp_path / "usr-lib"
    default.mkdir()
    _stub(default)
    empty = tmp_path / "nothing"
    empty.mkdir()
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(vllm_utils, "_cuda_roots_from_nvcc", lambda: ()), \
         mock.patch.object(vllm_utils, "_linker_default_dirs",
                           lambda: (str(default),)), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_the_cuda_root_inferred_from_nvcc_counts(tmp_path):
    """With neither CUDA_HOME nor CUDA_PATH set, FlashInfer's get_cuda_path()
    falls back to dirname(dirname(which nvcc)). A toolkit at /opt/cuda or in a
    conda prefix links against its own lib64/stubs, so checking only
    /usr/local/cuda would call that machine unlinkable."""
    root = tmp_path / "opt-cuda"
    stubs = root / "lib64" / "stubs"
    stubs.mkdir(parents = True)
    _stub(stubs)
    empty = tmp_path / "nothing"
    empty.mkdir()
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(vllm_utils, "_cuda_roots_from_nvcc",
                           lambda: (str(root),)), \
         mock.patch.object(vllm_utils, "_linker_default_dirs", lambda: ()), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(empty),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_nvcc_root_is_derived_the_way_flashinfer_derives_it(tmp_path):
    """dirname(dirname(nvcc)), both as found on PATH and symlink-resolved."""
    real = tmp_path / "opt" / "cuda"
    (real / "bin").mkdir(parents = True)
    nvcc = real / "bin" / "nvcc"
    nvcc.write_bytes(b"")
    shim_bin = tmp_path / "usr" / "bin"
    shim_bin.mkdir(parents = True)
    shim = shim_bin / "nvcc"
    shim.symlink_to(nvcc)
    with mock.patch.object(vllm_utils.shutil, "which", lambda name: str(shim)):
        roots = vllm_utils._cuda_roots_from_nvcc()
    assert str(tmp_path / "usr") in roots
    assert str(real) in roots


def test_no_nvcc_means_no_inferred_root():
    with mock.patch.object(vllm_utils.shutil, "which", lambda name: None):
        assert vllm_utils._cuda_roots_from_nvcc() == ()


def test_linker_defaults_never_come_back_empty():
    """A parse failure must widen to the fallback list: an empty tuple would
    silently reintroduce the false negative."""
    vllm_utils._linker_default_dirs.cache_clear()
    try:
        with mock.patch.object(vllm_utils, "re") as fake_re:
            fake_re.findall.return_value = []
            assert vllm_utils._linker_default_dirs() == vllm_utils._FALLBACK_LINKER_DIRS
    finally:
        vllm_utils._linker_default_dirs.cache_clear()


def test_multiarch_dirs_cover_the_debian_ubuntu_driver_location():
    """`/usr/lib/x86_64-linux-gnu` is where the NVIDIA driver puts the
    unversioned `libcuda.so`, and it is in ld's built-in SEARCH_DIRs, so
    `c++ ... -lcuda` resolves there with no -L at all."""
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(platform, "machine", lambda: "x86_64"), \
         mock.patch.object(sysconfig, "get_config_var",
                           lambda name: "x86_64-linux-gnu" if name == "MULTIARCH" else None):
        dirs = vllm_utils._multiarch_linker_dirs()
    assert "/usr/lib/x86_64-linux-gnu" in dirs
    assert "/lib/x86_64-linux-gnu" in dirs


def test_multiarch_dirs_cover_aarch64_without_a_configured_triplet():
    """manylinux and conda interpreters have no MULTIARCH; the triplet is then
    derived from the machine so Jetson / GH200 are still covered."""
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(platform, "machine", lambda: "aarch64"), \
         mock.patch.object(sysconfig, "get_config_var", lambda name: None):
        dirs = vllm_utils._multiarch_linker_dirs()
    assert "/usr/lib/aarch64-linux-gnu" in dirs


def test_multiarch_dirs_are_empty_off_linux():
    """Windows and macOS never reach the `-lcuda` branch; do not invent paths."""
    for fake in ("win32", "darwin"):
        with mock.patch.object(vllm_utils.sys, "platform", fake):
            assert vllm_utils._multiarch_linker_dirs() == ()


def test_the_fallback_includes_the_multiarch_dir_when_ld_cannot_be_consulted():
    """The regression. `ld --verbose` is unavailable whenever the image has no
    binutils, or whenever `ld` is really lld -- lld carries no built-in linker
    script and so emits no SEARCH_DIR (llvm/llvm-project#101661). The fallback
    is then the entire default-directory search, and it used to list only
    /usr/lib, /lib, /usr/lib64, /lib64, /usr/local/lib, /usr/local/lib64 --
    none of which is where Debian/Ubuntu keep libcuda.so. On such a host
    `_can_link_libcuda()` answered False while `c++ -shared -lcuda` links."""
    vllm_utils._linker_default_dirs.cache_clear()
    try:
        with mock.patch.object(subprocess, "run",
                               side_effect = FileNotFoundError("no ld")):
            dirs = vllm_utils._linker_default_dirs()
    finally:
        vllm_utils._linker_default_dirs.cache_clear()
    assert dirs == vllm_utils._FALLBACK_LINKER_DIRS
    if not sys.platform.startswith("linux"):
        pytest.skip("multiarch layout is Linux-only")
    triplet = sysconfig.get_config_var("MULTIARCH") or (platform.machine() + "-linux-gnu")
    assert "/usr/lib/" + triplet in dirs
    # The pre-existing entries must survive: RHEL/Fedora and the .run installer
    # use /usr/lib64, which has no triplet.
    for kept in ("/usr/lib", "/lib", "/usr/lib64", "/lib64",
                 "/usr/local/lib", "/usr/local/lib64"):
        assert kept in dirs


def test_a_runtime_only_multiarch_dir_is_still_not_linkable(tmp_path):
    """The Kaggle constraint, restated against the widened fallback: adding
    /usr/lib/<triplet> must not make a runtime-only image look linkable.
    `libcuda.so.1` present and `libcuda.so` absent stays False."""
    multiarch = tmp_path / "usr-lib-x86_64-linux-gnu"
    multiarch.mkdir()
    (multiarch / "libcuda.so.1").write_bytes(b"")
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(vllm_utils, "_cuda_roots_from_nvcc", lambda: ()), \
         mock.patch.object(vllm_utils, "_linker_default_dirs",
                           lambda: (str(multiarch),)), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", ()), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is False


# ------------------------------------------------------------ platform scope


def test_windows_looks_for_cuda_lib_not_a_posix_stub(tmp_path):
    """Windows links `cuda.lib` from the toolkit and searches LIB. Probing for
    libcuda.so there would find nothing and disable FlashInfer on a platform
    where it works."""
    libdir = tmp_path / "lib" / "x64"
    libdir.mkdir(parents = True)
    (libdir / "cuda.lib").write_bytes(b"")
    with mock.patch.object(vllm_utils.sys, "platform", "win32"), \
         mock.patch.dict(os.environ, {"CUDA_PATH": str(tmp_path), "CUDA_HOME": "",
                                      "LIB": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_windows_without_cuda_lib_is_not_linkable(tmp_path):
    root = tmp_path / "cuda"
    (root / "lib" / "x64").mkdir(parents = True)
    with mock.patch.object(vllm_utils.sys, "platform", "win32"), \
         mock.patch.dict(os.environ, {"CUDA_PATH": str(root), "CUDA_HOME": "",
                                      "LIB": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is False


def test_windows_with_nothing_to_go_on_does_not_claim_failure():
    """No CUDA_PATH and no LIB is no evidence either way. False would disable
    FlashInfer on the strength of a missing env var."""
    with mock.patch.object(vllm_utils.sys, "platform", "win32"), \
         mock.patch.dict(os.environ, {"CUDA_PATH": "", "CUDA_HOME": "",
                                      "LIB": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_macos_never_claims_the_link_will_fail():
    """CUDA is not in play on Darwin, so a negative would be an invention."""
    with mock.patch.object(vllm_utils.sys, "platform", "darwin"):
        assert vllm_utils._can_link_libcuda() is True


def test_wsl_is_linux_and_is_checked(tmp_path):
    """WSL reports sys.platform == "linux" and uses the POSIX toolchain, so it
    must take the Linux branch, not the Windows one."""
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    with _linux_with_no_ambient_cuda(), \
         mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(tmp_path),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is False
