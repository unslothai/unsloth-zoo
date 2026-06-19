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
    monkeypatch.delenv("UNSLOTH_LLAMA_FORCE_COMPILE", raising = False)
    _patch_platform(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(llama_cpp, "_detect_gpu_target", lambda: None)
    def boom(*a, **k):
        raise AssertionError("network must not be touched")
    monkeypatch.setattr(llama_cpp, "_resolve_llama_cpp_release", boom)
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
]}
FORK_ASSETS = {a["asset_name"]: f"https://example.invalid/{a['asset_name']}" for a in FORK_MANIFEST["artifacts"]}
FORK_ASSETS["llama-b9585-bin-macos-arm64.tar.gz"] = "https://example.invalid/llama-b9585-bin-macos-arm64.tar.gz"


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
    monkeypatch.setattr(
        llama_cpp, "_resolve_llama_cpp_release",
        lambda: _fake_release(tag, [f"llama-{tag}-bin-ubuntu-x64.tar.gz"]),
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
        lambda: _fake_release("b9000", ["llama-b9000-bin-ubuntu-x64.tar.gz"]),
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
