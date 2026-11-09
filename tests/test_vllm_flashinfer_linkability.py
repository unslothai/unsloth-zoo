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

"""FlashInfer must LINK, not just compile: -lcuda needs the driver stub libcuda.so, not runtime libcuda.so.1, so nvcc
and ninja present still fail late on Kaggle (RuntimeError: Ninja build failed); VLLM_USE_FLASHINFER_SAMPLER=0 does
not avoid it."""

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
    """Pins sys.platform to linux and neutralises the nvcc-inferred CUDA root and linker default dirs; on macOS or
    Windows _can_link_libcuda returns early, so unpinned tests answer a different question."""
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(vllm_utils, "_cuda_roots_from_nvcc", lambda: ()), \
         mock.patch.object(vllm_utils, "_linker_default_dirs", lambda: ()):
        yield


def test_a_runtime_libcuda_without_the_stub_is_not_linkable(tmp_path):
    """A runtime libcuda.so.1 without the libcuda.so stub must count as not linkable; every other check passes on the
    image that motivated this and the link still fails."""
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
    """A libcuda.so stub supplied via LIBRARY_PATH must count, since the linker consults it beyond its defaults; Unsloth's own Kaggle GRPO payload does exactly this."""
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
    """An empty LIBRARY_PATH entry must not count: os.path.join gives a relative libcuda.so that tests this process cwd, but FlashInfer links from its JIT build dir."""
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
# False when a link would have succeeded disables FlashInfer where it works, while True only restores the pre-check
# behaviour; two tests below cover false-negative paths.


def test_a_stub_in_a_default_linker_directory_counts(tmp_path):
    """A libcuda.so in ld's built-in SEARCH_DIRs counts even with no -L: the NVIDIA installer puts it in
    /usr/lib/x86_64-linux-gnu on Debian/Ubuntu, so False would disable FlashInfer."""
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
    """With CUDA_HOME and CUDA_PATH unset, get_cuda_path() infers the root from nvcc, so a toolkit at /opt/cuda or in a
    conda prefix must count, not just /usr/local/cuda."""
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
    """A linker-defaults parse failure must widen to the fallback list; an empty tuple would silently reintroduce the false negative."""
    vllm_utils._linker_default_dirs.cache_clear()
    try:
        with mock.patch.object(vllm_utils, "re") as fake_re:
            fake_re.findall.return_value = []
            assert vllm_utils._linker_default_dirs() == vllm_utils._FALLBACK_LINKER_DIRS
    finally:
        vllm_utils._linker_default_dirs.cache_clear()


def test_multiarch_dirs_cover_the_debian_ubuntu_driver_location():
    """/usr/lib/x86_64-linux-gnu holds the NVIDIA driver's unversioned libcuda.so and is in ld's built-in SEARCH_DIRs, so -lcuda resolves there with no -L."""
    with mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(platform, "machine", lambda: "x86_64"), \
         mock.patch.object(sysconfig, "get_config_var",
                           lambda name: "x86_64-linux-gnu" if name == "MULTIARCH" else None):
        dirs = vllm_utils._multiarch_linker_dirs()
    assert "/usr/lib/x86_64-linux-gnu" in dirs
    assert "/lib/x86_64-linux-gnu" in dirs


def test_multiarch_dirs_cover_aarch64_without_a_configured_triplet():
    """manylinux and conda interpreters lack MULTIARCH, so the triplet is derived from the machine, keeping Jetson and GH200 covered."""
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
    """Without ld --verbose (no binutils, or lld with no SEARCH_DIR, llvm/llvm-project#101661) the fallback must include
    the multiarch dir, else _can_link_libcuda() wrongly answers False."""
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
    # Pre-existing fallback entries must survive: RHEL/Fedora and the .run installer use /usr/lib64, which has no multiarch triplet.
    for kept in ("/usr/lib", "/lib", "/usr/lib64", "/lib64",
                 "/usr/local/lib", "/usr/local/lib64"):
        assert kept in dirs


def test_a_runtime_only_multiarch_dir_is_still_not_linkable(tmp_path):
    """Adding /usr/lib/<triplet> to the fallback must not make a runtime-only image look linkable: libcuda.so.1 without libcuda.so stays False."""
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
    """Windows links cuda.lib from the toolkit and searches LIB; probing for libcuda.so there would find nothing and
    wrongly disable FlashInfer on a working platform."""
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
    """On Windows with no CUDA_PATH and no LIB there is no evidence either way, so the check returns True rather than disabling FlashInfer on a missing env var."""
    with mock.patch.object(vllm_utils.sys, "platform", "win32"), \
         mock.patch.dict(os.environ, {"CUDA_PATH": "", "CUDA_HOME": "",
                                      "LIB": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is True


def test_macos_never_claims_the_link_will_fail():
    """CUDA is not in play on Darwin, so a negative would be an invention."""
    with mock.patch.object(vllm_utils.sys, "platform", "darwin"):
        assert vllm_utils._can_link_libcuda() is True


def test_wsl_is_linux_and_is_checked(tmp_path):
    """WSL reports sys.platform as linux and uses the POSIX toolchain, so it must take the Linux branch, not the Windows one."""
    (tmp_path / "libcuda.so.1").write_bytes(b"")
    with _linux_with_no_ambient_cuda(), \
         mock.patch.object(vllm_utils.sys, "platform", "linux"), \
         mock.patch.object(vllm_utils, "_FLASHINFER_LINK_DIRS", (str(tmp_path),)), \
         mock.patch.dict(os.environ, {"CUDA_HOME": "", "CUDA_PATH": "",
                                      "LIBRARY_PATH": ""}, clear = False):
        assert vllm_utils._can_link_libcuda() is False
