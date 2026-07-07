# Tests for the prebuilt-first llama.cpp install path: release resolution,
# asset selection, archive extraction/placement, fallback-to-compile contract,
# and reuse of an existing prebuilt install without deletion.

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_under_test_prebuilt", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


llama_cpp = _load_llama_cpp_module()

IS_POSIX = os.name == "posix"

FAKE_QUANTIZE = "#!/bin/sh\necho 'usage: llama-quantize [--help] model-f32.gguf'\nexit 0\n"
FAKE_QUANTIZE_BROKEN = "#!/bin/sh\necho 'segfault'\nexit 1\n"


def _add_file(tar, name, data, mode = 0o644):
    raw = data.encode() if isinstance(data, str) else data
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    info.mode = mode
    tar.addfile(info, io.BytesIO(raw))


def _make_binary_archive(path, tag, quantize_script = FAKE_QUANTIZE):
    with tarfile.open(path, "w:gz") as tar:
        _add_file(tar, f"llama-{tag}/llama-quantize", quantize_script, mode = 0o755)
        _add_file(tar, f"llama-{tag}/llama-cli", FAKE_QUANTIZE, mode = 0o755)
        _add_file(tar, f"llama-{tag}/libggml-base.so", "not really elf")
        _add_file(tar, f"llama-{tag}/LICENSE", "MIT-ish")


def _make_source_archive(path, tag):
    with tarfile.open(path, "w:gz") as tar:
        _add_file(tar, f"llama.cpp-{tag}/convert_hf_to_gguf.py", "# converter entrypoint\n")
        _add_file(tar, f"llama.cpp-{tag}/conversion/__init__.py", "TEXT_MODEL_MAP = {}\n")
        _add_file(tar, f"llama.cpp-{tag}/gguf-py/gguf/tensor_mapping.py", "mappings_cfg = {}\n")


def _fake_release(tag, asset_names):
    return tag, {name: f"https://example.invalid/{name}" for name in asset_names}


def _patch_platform(monkeypatch, system, machine):
    monkeypatch.setattr(llama_cpp.platform, "system", lambda: system)
    monkeypatch.setattr(llama_cpp.platform, "machine", lambda: machine)


# --- asset selection ---------------------------------------------------------

@pytest.mark.parametrize("system,machine,expected", [
    ("Linux",   "x86_64",  "llama-b9000-bin-ubuntu-x64.tar.gz"),
    ("Linux",   "AMD64",   "llama-b9000-bin-ubuntu-x64.tar.gz"),
    ("Linux",   "aarch64", "llama-b9000-bin-ubuntu-arm64.tar.gz"),
    ("Darwin",  "arm64",   "llama-b9000-bin-macos-arm64.tar.gz"),
    ("Darwin",  "x86_64",  "llama-b9000-bin-macos-x64.tar.gz"),
    ("Windows", "AMD64",   "llama-b9000-bin-win-cpu-x64.zip"),
    ("Windows", "ARM64",   "llama-b9000-bin-win-cpu-arm64.zip"),
])
def test_select_asset_matrix(monkeypatch, system, machine, expected):
    _patch_platform(monkeypatch, system, machine)
    tag, assets = _fake_release("b9000", [expected, "llama-b9000-ui.tar.gz"])
    name, url = llama_cpp._select_prebuilt_asset(tag, assets)
    assert name == expected
    assert url.endswith(expected)


@pytest.mark.parametrize("system,machine", [
    ("Linux", "i686"),
    ("FreeBSD", "x86_64"),
])
def test_select_asset_unsupported_platform(monkeypatch, system, machine):
    _patch_platform(monkeypatch, system, machine)
    tag, assets = _fake_release("b9000", ["llama-b9000-bin-ubuntu-x64.tar.gz"])
    assert llama_cpp._select_prebuilt_asset(tag, assets) is None


def test_select_asset_missing_from_release(monkeypatch):
    _patch_platform(monkeypatch, "Linux", "x86_64")
    tag, assets = _fake_release("b9000", ["llama-b9000-bin-macos-arm64.tar.gz"])
    assert llama_cpp._select_prebuilt_asset(tag, assets) is None


# --- release resolution ------------------------------------------------------

def test_resolve_release_env_pin_hits_tag_endpoint(monkeypatch):
    seen = {}
    class FakeResponse:
        def json(self):
            return {"tag_name": "b7777", "assets": [
                {"name": "a.tar.gz", "browser_download_url": "https://example.invalid/a.tar.gz"},
            ]}
    def fake_get(url, **kwargs):
        seen["url"] = url
        return FakeResponse()
    monkeypatch.setenv("UNSLOTH_LLAMA_TAG", "b7777")
    monkeypatch.setattr(llama_cpp, "_requests_get_with_retries", fake_get)
    tag, assets = llama_cpp._resolve_llama_cpp_release()
    assert tag == "b7777"
    assert seen["url"].endswith("/releases/tags/b7777")
    assert assets == {"a.tar.gz": "https://example.invalid/a.tar.gz"}


def test_resolve_release_failure_returns_none(monkeypatch):
    def fake_get(url, **kwargs):
        raise llama_cpp.requests.exceptions.ConnectionError("offline")
    monkeypatch.delenv("UNSLOTH_LLAMA_TAG", raising = False)
    monkeypatch.setattr(llama_cpp, "_requests_get_with_retries", fake_get)
    assert llama_cpp._resolve_llama_cpp_release() is None


# --- gates -------------------------------------------------------------------

def test_force_compile_skips_prebuilt(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_LLAMA_FORCE_COMPILE", "1")
    def boom(*a, **k):
        raise AssertionError("network must not be touched")
    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", boom)
    assert llama_cpp._maybe_install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp")) is None


def test_gpu_without_detectable_target_falls_back(monkeypatch, tmp_path):
    # gpu_support=True with no detectable GPU target does not substitute a CPU
    # prebuilt: it returns None so the caller compiles a GPU build. (Here no
    # prebuilt release resolves either, so None is returned regardless.)
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: None)
    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", lambda releases_api = None: None)
    assert llama_cpp._maybe_install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = True) is None


# --- GPU asset selection (unslothai/llama.cpp fork bundles) -------------------

FORK_MANIFEST = {"artifacts": [
    {"asset_name": "app-b9585-linux-x64-cuda12-older.tar.gz",    "install_kind": "linux-cuda",
     "runtime_line": "cuda12", "coverage_class": "older",    "supported_sms": ["70","75","80","86","89"],
     "min_sm": 70, "max_sm": 89, "rank": 10},
    {"asset_name": "app-b9585-linux-x64-cuda12-newer.tar.gz",    "install_kind": "linux-cuda",
     "runtime_line": "cuda12", "coverage_class": "newer",    "supported_sms": ["86","89","90","100","103","120"],
     "min_sm": 86, "max_sm": 120, "rank": 20},
    {"asset_name": "app-b9585-linux-x64-cuda12-portable.tar.gz", "install_kind": "linux-cuda",
     "runtime_line": "cuda12", "coverage_class": "portable", "supported_sms": ["70","75","80","86","89","90","100","103","120"],
     "min_sm": 70, "max_sm": 120, "rank": 30},
    {"asset_name": "app-b9585-linux-x64-cuda13-newer.tar.gz",    "install_kind": "linux-cuda",
     "runtime_line": "cuda13", "coverage_class": "newer",    "supported_sms": ["86","89","90","100","103","120"],
     "min_sm": 86, "max_sm": 120, "rank": 50},
    {"asset_name": "app-b9585-linux-x64-rocm-gfx110X.tar.gz",    "install_kind": "linux-rocm"},
    {"asset_name": "app-b9585-linux-x64-rocm-gfx120X.tar.gz",    "install_kind": "linux-rocm"},
    {"asset_name": "app-b9585-linux-x64-cpu.tar.gz",            "install_kind": "linux-cpu"},
    {"asset_name": "app-b9585-linux-arm64-cpu.tar.gz",         "install_kind": "linux-arm64"},
    {"asset_name": "app-b9585-windows-x64-cpu.zip",            "install_kind": "windows-cpu"},
]}
FORK_ASSETS = {a["asset_name"]: f"https://example.invalid/{a['asset_name']}" for a in FORK_MANIFEST["artifacts"]}
FORK_ASSETS["llama-b9585-bin-macos-arm64.tar.gz"] = "https://example.invalid/llama-b9585-bin-macos-arm64.tar.gz"
FORK_ASSETS["llama-b9585-bin-macos-x64.tar.gz"] = "https://example.invalid/llama-b9585-bin-macos-x64.tar.gz"


def test_select_gpu_assets_narrowest_coverage_first(monkeypatch):
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: ("cuda", 86, "cuda12"))
    names = [n for n, _ in llama_cpp._select_gpu_assets("b9585", FORK_ASSETS, FORK_MANIFEST)]
    # sm86 fits cuda12-older (range 19) before cuda12-newer (range 34),
    # then the cuda12 portable, then the other runtime line.
    assert names[:2] == [
        "app-b9585-linux-x64-cuda12-older.tar.gz",
        "app-b9585-linux-x64-cuda12-portable.tar.gz",
    ]
    assert "app-b9585-linux-x64-cuda13-newer.tar.gz" in names


def test_select_gpu_assets_prefers_torch_runtime_line(monkeypatch):
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: ("cuda", 100, "cuda13"))
    names = [n for n, _ in llama_cpp._select_gpu_assets("b9585", FORK_ASSETS, FORK_MANIFEST)]
    assert names[0] == "app-b9585-linux-x64-cuda13-newer.tar.gz"


def test_select_gpu_assets_rocm_family(monkeypatch):
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: ("rocm", "gfx1100"))
    names = [n for n, _ in llama_cpp._select_gpu_assets("b9585", FORK_ASSETS, FORK_MANIFEST)]
    assert names == ["app-b9585-linux-x64-rocm-gfx110X.tar.gz"]


def test_select_gpu_assets_macos_metal_bundle(monkeypatch):
    # macOS needs no torch GPU detection; the fork Metal bundle is selected.
    _patch_platform(monkeypatch, "Darwin", "arm64")
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: None)
    names = [n for n, _ in llama_cpp._select_gpu_assets("b9585", FORK_ASSETS, FORK_MANIFEST)]
    assert names == ["llama-b9585-bin-macos-arm64.tar.gz"]


def test_rocm_gfx_family_mapping():
    assert llama_cpp._rocm_gfx_family("gfx1100") == "gfx110X"
    assert llama_cpp._rocm_gfx_family("gfx1030") == "gfx103X"
    assert llama_cpp._rocm_gfx_family("gfx1201") == "gfx120X"
    assert llama_cpp._rocm_gfx_family("gfx1151") == "gfx1151"
    assert llama_cpp._rocm_gfx_family("gfx906") is None


@pytest.mark.skipif(not IS_POSIX, reason = "shell-script fake binaries")
def test_gpu_full_install_happy_path(monkeypatch, tmp_path):
    tag = "b9585"
    asset = f"app-{tag}-linux-x64-cuda12-newer.tar.gz"
    binary = tmp_path / "fixtures" / asset
    source = tmp_path / "fixtures" / "source.tar.gz"
    os.makedirs(binary.parent, exist_ok = True)
    _make_binary_archive(str(binary), tag)
    _make_source_archive(str(source), tag)
    sha = llama_cpp._sha256_file(str(binary))

    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: ("cuda", 100, "cuda12"))
    monkeypatch.setattr(
        llama_cpp, "_resolve_llama_cpp_release",
        lambda releases_api = None: (tag, {asset: f"https://example.invalid/{asset}"}),
    )
    monkeypatch.setattr(
        llama_cpp, "_fetch_release_json_asset",
        lambda assets, name: FORK_MANIFEST if "manifest" in name else {"artifacts": {asset: {"sha256": sha}}},
    )
    def fake_download(url, dest_path):
        fixture = binary if "app-" in os.path.basename(dest_path) else source
        with open(fixture, "rb") as fr, open(dest_path, "wb") as fw:
            fw.write(fr.read())
    monkeypatch.setattr(llama_cpp, "_download_archive", fake_download)
    monkeypatch.setattr(llama_cpp, "try_execute", lambda *a, **k: "")
    monkeypatch.setattr(llama_cpp, "check_pip", lambda: "pip")

    folder = str(tmp_path / "llama.cpp")
    result = llama_cpp._install_llama_cpp_prebuilt(folder, gpu_support = True)
    assert result is not None
    marker = json.load(open(os.path.join(folder, llama_cpp.UNSLOTH_PREBUILT_INFO_FILENAME)))
    assert marker["repo"] == "unslothai/llama.cpp"
    assert marker["asset"] == asset


@pytest.mark.skipif(not IS_POSIX, reason = "shell-script fake binaries")
def test_gpu_sha256_mismatch_falls_back(monkeypatch, tmp_path):
    tag = "b9585"
    asset = f"app-{tag}-linux-x64-cuda12-newer.tar.gz"
    binary = tmp_path / "fixtures" / asset
    os.makedirs(binary.parent, exist_ok = True)
    _make_binary_archive(str(binary), tag)

    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: ("cuda", 100, "cuda12"))
    monkeypatch.setattr(
        llama_cpp, "_resolve_llama_cpp_release",
        lambda releases_api = None: (tag, {asset: f"https://example.invalid/{asset}"}),
    )
    monkeypatch.setattr(
        llama_cpp, "_fetch_release_json_asset",
        lambda assets, name: FORK_MANIFEST if "manifest" in name else {"artifacts": {asset: {"sha256": "0" * 64}}},
    )
    def fake_download(url, dest_path):
        with open(binary, "rb") as fr, open(dest_path, "wb") as fw:
            fw.write(fr.read())
    monkeypatch.setattr(llama_cpp, "_download_archive", fake_download)

    folder = str(tmp_path / "llama.cpp")
    assert llama_cpp._install_llama_cpp_prebuilt(folder, gpu_support = True) is None
    assert not os.path.exists(folder)


# --- extraction and placement ------------------------------------------------

def test_extract_rejects_path_traversal(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        _add_file(tar, "../escape.txt", "pwned")
    with pytest.raises(RuntimeError, match = "escapes extraction dir"):
        llama_cpp._extract_archive(str(evil), str(tmp_path / "out"))


def _add_link(tar, name, linkname, link_type = tarfile.SYMTYPE):
    info = tarfile.TarInfo(name)
    info.type = link_type
    info.linkname = linkname
    tar.addfile(info)


def test_extract_rejects_symlink_escape(tmp_path):
    evil = tmp_path / "evil-symlink.tar.gz"
    outside = tmp_path / "outside"
    outside.mkdir()
    with tarfile.open(evil, "w:gz") as tar:
        _add_link(tar, "link", str(outside))
        _add_file(tar, "link/pwned.txt", "pwned")
    with pytest.raises(RuntimeError, match = "escapes extraction dir"):
        llama_cpp._extract_archive(str(evil), str(tmp_path / "out"))
    assert not (outside / "pwned.txt").exists()


def test_extract_rejects_hardlink_escape(tmp_path):
    evil = tmp_path / "evil-hardlink.tar.gz"
    with tarfile.open(evil, "w:gz") as tar:
        _add_link(tar, "link", "../../../../etc/passwd", link_type = tarfile.LNKTYPE)
    with pytest.raises(RuntimeError, match = "escapes extraction dir"):
        llama_cpp._extract_archive(str(evil), str(tmp_path / "out"))


def test_extract_rejects_zip_symlink(tmp_path):
    import zipfile
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = 0o120777 << 16  # S_IFLNK
        archive.writestr(info, str(tmp_path / "outside"))
    with pytest.raises(RuntimeError, match = "symlink"):
        llama_cpp._extract_archive(str(evil), str(tmp_path / "out"))


@pytest.mark.skipif(not IS_POSIX or os.geteuid() == 0, reason = "read-only dir perms (non-root POSIX)")
def test_extract_allows_readonly_dir_before_contents(tmp_path):
    # A read-only dir entry preceding its files must not trip extraction:
    # extractall defers directory perms until contents are written.
    archive = tmp_path / "ro.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        d = tarfile.TarInfo("ro")
        d.type = tarfile.DIRTYPE
        d.mode = 0o555
        tar.addfile(d)
        _add_file(tar, "ro/file.txt", "ok")
    out = tmp_path / "out"
    llama_cpp._extract_archive(str(archive), str(out))
    content = (out / "ro" / "file.txt").read_text()
    os.chmod(out / "ro", 0o755)  # restore write bit so tmp_path cleanup can unlink
    assert content == "ok"


@pytest.mark.skipif(not IS_POSIX, reason = "symlink semantics")
def test_extract_allows_in_tree_symlink(tmp_path):
    # Real release tarballs ship relative .so symlinks that stay in-tree.
    archive = tmp_path / "libs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_file(tar, "libllama.so.0", "not really elf")
        _add_link(tar, "libllama.so", "libllama.so.0")
    out = tmp_path / "out"
    llama_cpp._extract_archive(str(archive), str(out))
    assert (out / "libllama.so").is_symlink()
    assert (out / "libllama.so").resolve() == (out / "libllama.so.0").resolve()


@pytest.mark.skipif(not IS_POSIX, reason = "shell-script fake binaries")
def test_extract_and_place_linux(tmp_path):
    archive = tmp_path / "llama-b9000-bin-ubuntu-x64.tar.gz"
    _make_binary_archive(str(archive), "b9000")
    extract_dir = tmp_path / "extracted"
    os.makedirs(extract_dir)
    llama_cpp._extract_archive(str(archive), str(extract_dir))
    root = llama_cpp._single_extracted_root(str(extract_dir))
    assert os.path.basename(root) == "llama-b9000"
    install = tmp_path / "install"
    llama_cpp._place_prebuilt_binaries(root, str(install))
    quantizer = install / "llama-quantize"
    assert quantizer.is_file() and os.access(quantizer, os.X_OK)
    assert (install / "libggml-base.so").is_file()


def _make_binary_zip(path, tag):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"llama-{tag}/llama-quantize.exe", b"MZ fake exe")
        z.writestr(f"llama-{tag}/llama-cli.exe", b"MZ fake exe")
        z.writestr(f"llama-{tag}/ggml.dll", b"fake dll")
        z.writestr(f"llama-{tag}/llama.dll", b"fake dll")
        z.writestr(f"llama-{tag}/LICENSE", b"MIT-ish")


def test_extract_and_place_windows_zip(monkeypatch, tmp_path):
    # The fork/ggml Windows bundle is a .zip; _place_prebuilt_binaries must land
    # the executables + DLLs under build/bin/Release (where check_llama_cpp looks
    # on Windows). Exercise the Windows placement on any host by patching
    # IS_WINDOWS -- nothing is executed, only copied, so no POSIX skip is needed.
    archive = tmp_path / "llama-b9000-bin-win-cpu-x64.zip"
    _make_binary_zip(str(archive), "b9000")
    extract_dir = tmp_path / "extracted"
    os.makedirs(extract_dir)
    llama_cpp._extract_archive(str(archive), str(extract_dir))
    root = llama_cpp._single_extracted_root(str(extract_dir))
    assert os.path.basename(root) == "llama-b9000"
    monkeypatch.setattr(llama_cpp, "IS_WINDOWS", True)
    install = tmp_path / "install"
    llama_cpp._place_prebuilt_binaries(root, str(install))
    release = install / "build" / "bin" / "Release"
    assert (release / "llama-quantize.exe").is_file()
    assert (release / "llama-cli.exe").is_file()
    assert (release / "ggml.dll").is_file()
    # On Windows nothing is placed at the flat install root.
    assert not (install / "llama-quantize.exe").exists()


# --- full install orchestration ----------------------------------------------

def _wire_fake_downloads(monkeypatch, tmp_path, tag, quantize_script = FAKE_QUANTIZE):
    """Point release resolution and downloads at local fixture archives."""
    binary = tmp_path / "fixtures" / f"llama-{tag}-bin-ubuntu-x64.tar.gz"
    source = tmp_path / "fixtures" / "source.tar.gz"
    os.makedirs(binary.parent, exist_ok = True)
    _make_binary_archive(str(binary), tag, quantize_script)
    _make_source_archive(str(source), tag)

    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    # The fork (published) release is unreachable here, so this exercises the
    # ggml-org CPU fallback (attempt 3). The fork CPU path has its own tests.
    monkeypatch.setattr(
        llama_cpp, "_resolve_llama_cpp_release",
        lambda releases_api = None: None if releases_api == llama_cpp.LLAMA_CPP_PUBLISHED_RELEASES_API
        else _fake_release(tag, [f"llama-{tag}-bin-ubuntu-x64.tar.gz"]),
    )
    def fake_download(url, dest_path):
        fixture = binary if "-bin-" in os.path.basename(dest_path) else source
        with open(fixture, "rb") as fr, open(dest_path, "wb") as fw:
            fw.write(fr.read())
    monkeypatch.setattr(llama_cpp, "_download_archive", fake_download)
    monkeypatch.setattr(llama_cpp, "try_execute", lambda *a, **k: "")
    monkeypatch.setattr(llama_cpp, "check_pip", lambda: "pip")


@pytest.mark.skipif(not IS_POSIX, reason = "shell-script fake binaries")
def test_full_prebuilt_install_happy_path(monkeypatch, tmp_path):
    _wire_fake_downloads(monkeypatch, tmp_path, "b9000")
    folder = str(tmp_path / "llama.cpp")
    result = llama_cpp._install_llama_cpp_prebuilt(folder)
    assert result is not None
    quantizer, converter = result
    assert quantizer == os.path.join(folder, "llama-quantize")
    assert converter == os.path.join(folder, "convert_hf_to_gguf.py")
    assert os.path.isfile(os.path.join(folder, "gguf-py", "gguf", "tensor_mapping.py"))
    assert os.path.isfile(os.path.join(folder, "conversion", "__init__.py"))
    marker = json.load(open(os.path.join(folder, llama_cpp.UNSLOTH_PREBUILT_INFO_FILENAME)))
    assert marker["tag"] == "b9000"
    assert marker["asset"] == "llama-b9000-bin-ubuntu-x64.tar.gz"
    # Staging directories are cleaned up
    leftovers = [e for e in os.listdir(tmp_path) if e.startswith(".llama_cpp_prebuilt_")]
    assert leftovers == []


@pytest.mark.skipif(not IS_POSIX, reason = "shell-script fake binaries")
def test_validation_failure_falls_back(monkeypatch, tmp_path):
    _wire_fake_downloads(monkeypatch, tmp_path, "b9000", quantize_script = FAKE_QUANTIZE_BROKEN)
    folder = str(tmp_path / "llama.cpp")
    assert llama_cpp._install_llama_cpp_prebuilt(folder) is None
    # A failed install never materializes the target folder
    assert not os.path.exists(folder)


def test_corrupt_archive_falls_back(monkeypatch, tmp_path):
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    monkeypatch.setattr(
        llama_cpp, "_resolve_llama_cpp_release",
        lambda releases_api = None: None if releases_api == llama_cpp.LLAMA_CPP_PUBLISHED_RELEASES_API
        else _fake_release("b9000", ["llama-b9000-bin-ubuntu-x64.tar.gz"]),
    )
    def fake_download(url, dest_path):
        with open(dest_path, "wb") as f:
            f.write(b"this is not a tarball")
    monkeypatch.setattr(llama_cpp, "_download_archive", fake_download)
    folder = str(tmp_path / "llama.cpp")
    assert llama_cpp._install_llama_cpp_prebuilt(folder) is None
    assert not os.path.exists(folder)


@pytest.mark.skipif(not IS_POSIX, reason = "shell-script fake binaries")
def test_install_llama_cpp_reuses_prebuilt_without_delete(monkeypatch, tmp_path):
    # First call installs via the (stubbed) prebuilt path; the second must hit
    # the existing-install reuse and NOT rmtree the marker-bearing folder.
    folder = str(tmp_path / "llama.cpp")

    def fake_prebuilt(llama_cpp_folder, **kwargs):
        os.makedirs(llama_cpp_folder, exist_ok = True)
        quantizer = os.path.join(llama_cpp_folder, "llama-quantize")
        with open(quantizer, "w") as f:
            f.write(FAKE_QUANTIZE)
        os.chmod(quantizer, 0o755)
        converter = os.path.join(llama_cpp_folder, "convert_hf_to_gguf.py")
        with open(converter, "w") as f:
            f.write("# converter\n")
        with open(os.path.join(llama_cpp_folder, llama_cpp.UNSLOTH_PREBUILT_INFO_FILENAME), "w") as f:
            json.dump({"tag": "b9000"}, f)
        with open(os.path.join(llama_cpp_folder, "sentinel.txt"), "w") as f:
            f.write("must survive")
        return quantizer, converter

    monkeypatch.setattr(llama_cpp, "_maybe_install_llama_cpp_prebuilt", fake_prebuilt)
    def no_rmtree(path, *a, **k):
        raise AssertionError(f"rmtree must not run on {path}")
    monkeypatch.setattr(llama_cpp.shutil, "rmtree", no_rmtree)

    q1, c1 = llama_cpp.install_llama_cpp(llama_cpp_folder = folder)
    q2, c2 = llama_cpp.install_llama_cpp(llama_cpp_folder = folder)
    assert (q1, c1) == (q2, c2)
    assert os.path.isfile(os.path.join(folder, "sentinel.txt"))


# --- fork CPU asset selection (final prebuilt fallback) -----------------------

@pytest.mark.parametrize("system,machine,expected", [
    ("Linux",   "x86_64",  "app-b9585-linux-x64-cpu.tar.gz"),
    ("Linux",   "AMD64",   "app-b9585-linux-x64-cpu.tar.gz"),
    ("Linux",   "aarch64", "app-b9585-linux-arm64-cpu.tar.gz"),
    ("Darwin",  "arm64",   "llama-b9585-bin-macos-arm64.tar.gz"),
    ("Darwin",  "x86_64",  "llama-b9585-bin-macos-x64.tar.gz"),
    ("Windows", "AMD64",   "app-b9585-windows-x64-cpu.zip"),
])
def test_select_cpu_assets_matrix(monkeypatch, system, machine, expected):
    _patch_platform(monkeypatch, system, machine)
    selected = llama_cpp._select_cpu_assets("b9585", FORK_ASSETS, FORK_MANIFEST)
    names = [n for n, _ in selected]
    assert names == [expected]
    assert selected[0][1].endswith(expected)


@pytest.mark.parametrize("system,machine", [
    ("Linux",   "i686"),    # unsupported arch
    ("FreeBSD", "x86_64"),  # unsupported OS
])
def test_select_cpu_assets_unsupported(monkeypatch, system, machine):
    _patch_platform(monkeypatch, system, machine)
    assert llama_cpp._select_cpu_assets("b9585", FORK_ASSETS, FORK_MANIFEST) == []


def test_select_cpu_assets_macos_missing_bundle(monkeypatch):
    # Darwin asks for the Metal bundle by convention; absent -> no fork CPU asset.
    _patch_platform(monkeypatch, "Darwin", "arm64")
    assets = {k: v for k, v in FORK_ASSETS.items() if k != "llama-b9585-bin-macos-arm64.tar.gz"}
    assert llama_cpp._select_cpu_assets("b9585", assets, FORK_MANIFEST) == []


def test_select_cpu_assets_linux_missing_from_release(monkeypatch):
    # Manifest lists linux-cpu but the asset isn't in the release map -> skipped.
    _patch_platform(monkeypatch, "Linux", "x86_64")
    assets = {k: v for k, v in FORK_ASSETS.items() if k != "app-b9585-linux-x64-cpu.tar.gz"}
    assert llama_cpp._select_cpu_assets("b9585", assets, FORK_MANIFEST) == []


# --- converter hydration source resolution (fork "mix" tag fix) ---------------

def _capture_hydrate_url(monkeypatch):
    """Record the source URL _hydrate_converter_sources resolves, then short-circuit
    the download so only URL selection is under test."""
    seen = {}
    def rec(url, dest_path):
        seen["url"] = url
        raise RuntimeError("stop after URL resolution")
    monkeypatch.setattr(llama_cpp, "_download_archive", rec)
    return seen


def test_hydrate_prefers_fork_source_asset(monkeypatch, tmp_path):
    seen = _capture_hydrate_url(monkeypatch)
    src_assets = {"llama.cpp-source-b9739-mix-2d6bd50.tar.gz": "https://fork.invalid/forksrc.tar.gz"}
    with pytest.raises(RuntimeError):
        llama_cpp._hydrate_converter_sources(
            "b9739-mix-2d6bd50", str(tmp_path / "install"), source_assets = src_assets,
        )
    assert seen["url"] == "https://fork.invalid/forksrc.tar.gz"


def test_hydrate_strips_mix_tag_when_no_fork_source(monkeypatch, tmp_path):
    # Fork mix tag, but no fork source asset -> strip the -mix-... suffix and pull
    # the matching upstream tag from ggml-org (the verbatim mix tag 404s there).
    seen = _capture_hydrate_url(monkeypatch)
    with pytest.raises(RuntimeError):
        llama_cpp._hydrate_converter_sources(
            "b9739-mix-2d6bd50", str(tmp_path / "install"), source_assets = None,
        )
    assert seen["url"] == llama_cpp.LLAMA_CPP_SOURCE_TARBALL.format(tag = "b9739")


def test_hydrate_plain_ggml_tag_unchanged(monkeypatch, tmp_path):
    # A plain ggml-org tag carries no -mix- suffix, so resolution is a no-op.
    seen = _capture_hydrate_url(monkeypatch)
    with pytest.raises(RuntimeError):
        llama_cpp._hydrate_converter_sources("b9000", str(tmp_path / "install"))
    assert seen["url"] == llama_cpp.LLAMA_CPP_SOURCE_TARBALL.format(tag = "b9000")


# --- attempt ORDER across the fork-GPU -> fork-CPU -> ggml-org chain ----------

def _wire_attempt_recorder(monkeypatch, *, fork_release, ggml_release, manifest = FORK_MANIFEST):
    """Record (repo, asset_name) for every staged attempt then raise, so the whole
    attempt chain is walked deterministically without any real download."""
    calls = []
    def fake_resolve(releases_api = None):
        if releases_api == llama_cpp.LLAMA_CPP_PUBLISHED_RELEASES_API:
            return fork_release
        return ggml_release
    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", fake_resolve)
    monkeypatch.setattr(
        llama_cpp, "_fetch_release_json_asset",
        lambda assets, name: manifest if "manifest" in name else {"artifacts": {}},
    )
    def fake_stage(folder, tag, asset_name, asset_url, expected_sha256 = None, repo = None, source_assets = None):
        calls.append((repo, asset_name))
        raise RuntimeError("forced fall-through")
    monkeypatch.setattr(llama_cpp, "_stage_prebuilt_install", fake_stage)
    monkeypatch.setattr(llama_cpp, "try_execute", lambda *a, **k: "")
    monkeypatch.setattr(llama_cpp, "check_pip", lambda: "pip")
    return calls


def test_attempt_order_cpu_only_linux(monkeypatch, tmp_path):
    # gpu_support=False on Linux: fork CPU bundle first, ggml-org CPU as tertiary.
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    ggml = _fake_release("b9000", ["llama-b9000-bin-ubuntu-x64.tar.gz"])
    calls = _wire_attempt_recorder(monkeypatch, fork_release = ("b9585", FORK_ASSETS), ggml_release = ggml)
    assert llama_cpp._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = False) is None
    assert calls == [
        ("unslothai/llama.cpp", "app-b9585-linux-x64-cpu.tar.gz"),
        ("ggml-org/llama.cpp",  "llama-b9000-bin-ubuntu-x64.tar.gz"),
    ]


def test_attempt_order_gpu_cuda_host(monkeypatch, tmp_path):
    # gpu_support=True with a CUDA target: ONLY fork GPU bundles are attempted --
    # no CPU prebuilt and no ggml-org -- so if every GPU prebuilt fails the caller
    # compiles a GPU build rather than silently landing on a CPU-only prebuilt.
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: ("cuda", 100, "cuda13"))
    ggml = _fake_release("b9000", ["llama-b9000-bin-ubuntu-x64.tar.gz"])
    calls = _wire_attempt_recorder(monkeypatch, fork_release = ("b9585", FORK_ASSETS), ggml_release = ggml)
    assert llama_cpp._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = True) is None
    assert calls, "expected at least one fork GPU attempt"
    assert all(r == "unslothai/llama.cpp" and "cuda" in a for r, a in calls)
    assert not any(a == "app-b9585-linux-x64-cpu.tar.gz" for _, a in calls)
    assert not any(r == "ggml-org/llama.cpp" for r, _ in calls)


def test_attempt_order_gpu_no_target_compiles(monkeypatch, tmp_path):
    # gpu_support=True but NO GPU target: no GPU bundle matches and CPU prebuilts
    # are never substituted for a GPU request, so nothing is attempted and the
    # caller compiles a GPU build (instead of the old silent CPU fallback).
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: None)
    ggml = _fake_release("b9000", ["llama-b9000-bin-ubuntu-x64.tar.gz"])
    calls = _wire_attempt_recorder(monkeypatch, fork_release = ("b9585", FORK_ASSETS), ggml_release = ggml)
    assert llama_cpp._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = True) is None
    assert calls == []


def test_attempt_order_darwin_fork_only_no_ggml(monkeypatch, tmp_path):
    # macOS: only the fork Metal bundle (GPU+CPU selectors dedup to one), and
    # ggml-org is NEVER consulted (its recent macOS build needs macOS 26+).
    _patch_platform(monkeypatch, "Darwin", "arm64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: None)
    def fork_only(releases_api = None):
        if releases_api == llama_cpp.LLAMA_CPP_PUBLISHED_RELEASES_API:
            return ("b9585", FORK_ASSETS)
        raise AssertionError("ggml-org must not be consulted on macOS")
    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", fork_only)
    monkeypatch.setattr(
        llama_cpp, "_fetch_release_json_asset",
        lambda assets, name: FORK_MANIFEST if "manifest" in name else {"artifacts": {}},
    )
    calls = []
    def fake_stage(folder, tag, asset_name, asset_url, expected_sha256 = None, repo = None, source_assets = None):
        calls.append((repo, asset_name))
        raise RuntimeError("forced")
    monkeypatch.setattr(llama_cpp, "_stage_prebuilt_install", fake_stage)
    monkeypatch.setattr(llama_cpp, "try_execute", lambda *a, **k: "")
    monkeypatch.setattr(llama_cpp, "check_pip", lambda: "pip")
    assert llama_cpp._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = True) is None
    assert calls == [("unslothai/llama.cpp", "llama-b9585-bin-macos-arm64.tar.gz")]


def test_attempt_order_fork_unreachable_uses_ggml(monkeypatch, tmp_path):
    # Fork release resolution fails -> ggml-org CPU is still tried (resilience).
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    ggml = _fake_release("b9000", ["llama-b9000-bin-ubuntu-x64.tar.gz"])
    calls = _wire_attempt_recorder(monkeypatch, fork_release = None, ggml_release = ggml)
    assert llama_cpp._install_llama_cpp_prebuilt(str(tmp_path / "llama.cpp"), gpu_support = False) is None
    assert calls == [("ggml-org/llama.cpp", "llama-b9000-bin-ubuntu-x64.tar.gz")]


@pytest.mark.skipif(not IS_POSIX, reason = "shell-script fake binaries")
def test_cpu_fork_full_install_happy_path(monkeypatch, tmp_path):
    # End-to-end (real _stage_prebuilt_install): gpu_support=False installs the
    # fork app-*-cpu bundle and hydrates the converter from the fork's OWN source
    # asset for a "mix" tag -- never the 404-ing ggml-org codeload URL.
    tag = "b9585-mix-abc1234"
    asset = f"app-{tag}-linux-x64-cpu.tar.gz"
    fork_source = f"llama.cpp-source-{tag}.tar.gz"
    binary = tmp_path / "fixtures" / asset
    source = tmp_path / "fixtures" / "source.tar.gz"
    os.makedirs(binary.parent, exist_ok = True)
    _make_binary_archive(str(binary), tag)
    _make_source_archive(str(source), tag)

    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    manifest = {"artifacts": [{"asset_name": asset, "install_kind": "linux-cpu"}]}
    fork_assets = {
        asset       : f"https://example.invalid/{asset}",
        fork_source : f"https://example.invalid/{fork_source}",
    }
    def fake_resolve(releases_api = None):
        # Only the fork resolves; ggml is unreachable so the fork CPU bundle wins.
        return (tag, fork_assets) if releases_api == llama_cpp.LLAMA_CPP_PUBLISHED_RELEASES_API else None
    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", fake_resolve)
    monkeypatch.setattr(
        llama_cpp, "_fetch_release_json_asset",
        lambda assets, name: manifest if "manifest" in name else {"artifacts": {}},
    )
    downloaded = []
    def fake_download(url, dest_path):
        downloaded.append(url)
        fixture = binary if os.path.basename(dest_path) == asset else source
        with open(fixture, "rb") as fr, open(dest_path, "wb") as fw:
            fw.write(fr.read())
    monkeypatch.setattr(llama_cpp, "_download_archive", fake_download)
    monkeypatch.setattr(llama_cpp, "try_execute", lambda *a, **k: "")
    monkeypatch.setattr(llama_cpp, "check_pip", lambda: "pip")

    folder = str(tmp_path / "llama.cpp")
    result = llama_cpp._install_llama_cpp_prebuilt(folder, gpu_support = False)
    assert result is not None
    quantizer, converter = result
    assert quantizer == os.path.join(folder, "llama-quantize")
    assert converter == os.path.join(folder, "convert_hf_to_gguf.py")
    marker = json.load(open(os.path.join(folder, llama_cpp.UNSLOTH_PREBUILT_INFO_FILENAME)))
    assert marker["repo"] == "unslothai/llama.cpp"
    assert marker["asset"] == asset
    # Converter hydrated from the fork's own source asset, not ggml-org codeload.
    from urllib.parse import urlparse
    assert f"https://example.invalid/{fork_source}" in downloaded
    assert all(urlparse(u).netloc != "codeload.github.com" for u in downloaded)


def test_force_compile_skips_prebuilt_gpu(monkeypatch, tmp_path):
    # UNSLOTH_LLAMA_FORCE_COMPILE=1 must bypass the prebuilt path for gpu_support=True
    # too (no release resolution, no GPU probing).
    monkeypatch.setenv("UNSLOTH_LLAMA_FORCE_COMPILE", "1")
    def boom(*a, **k):
        raise AssertionError("network must not be touched")
    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", boom)
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", boom)
    assert llama_cpp._maybe_install_llama_cpp_prebuilt(
        str(tmp_path / "llama.cpp"), gpu_support = True,
    ) is None
