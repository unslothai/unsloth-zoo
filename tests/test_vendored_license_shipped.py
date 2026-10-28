"""Vendored MIT license and notice must ship in the wheel: unsloth_zoo/_vendored/_licenses.py mirrors the plain files as
importable strings, byte-identical, with no package-data entry."""
import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDORED = os.path.join(_REPO_ROOT, "unsloth_zoo", "_vendored")


def _load_licenses_module():
    # Load by file path to avoid importing unsloth_zoo (it pulls in the patching machinery); setuptools packages.find
    # ships every .py under _vendored, so the module reaches the wheel.
    path = os.path.join(_VENDORED, "_licenses.py")
    spec = importlib.util.spec_from_file_location("unsloth_zoo_vendored_licenses", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(*parts):
    with open(os.path.join(_VENDORED, *parts), encoding="utf-8") as f:
        return f.read()


def test_embedded_license_is_mit_and_present():
    lic = _load_licenses_module()
    assert "MIT License" in lic.FLA_LICENSE
    assert "Songlin Yang, Yu Zhang, Zhiyuan Li" in lic.FLA_LICENSE
    # The redistribution clause itself must be present, not just a copyright line.
    assert "shall be included in all" in lic.FLA_LICENSE


def test_embedded_text_matches_canonical_files():
    lic = _load_licenses_module()
    assert lic.FLA_LICENSE.strip() == _read("fla", "LICENSE").strip()
    assert lic.NOTICE.strip() == _read("NOTICE").strip()
    assert lic.FLA_MANIFEST.strip() == _read("fla", "MANIFEST").strip()


def test_embedded_vllm_license_is_apache_and_matches():
    lic = _load_licenses_module()
    assert "Apache License" in lic.VLLM_LICENSE and "Version 2.0" in lic.VLLM_LICENSE
    assert lic.VLLM_LICENSE.strip() == _read("vllm", "LICENSE").strip()
    assert lic.VLLM_MANIFEST.strip() == _read("vllm", "MANIFEST").strip()
    assert "vllm-project/vllm/pull/48792" in lic.NOTICE
