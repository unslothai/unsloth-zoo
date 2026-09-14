"""The vendored third-party MIT license and notice must ship inside the wheel.

A built wheel installs only importable .py modules (setuptools packages.find with
include-package-data = false), so the plain LICENSE / NOTICE / MANIFEST files under
unsloth_zoo/_vendored are not packaged. unsloth_zoo/_vendored/_licenses.py mirrors
that text as importable module strings so the MIT attribution required to
redistribute the vendored fla sources travels with the code, with no pyproject
package-data entry. These CPU-only tests guard that the module exists, carries the
MIT license, and stays byte-identical to the canonical source-tree files.
"""
import importlib.util
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENDORED = os.path.join(_REPO_ROOT, "unsloth_zoo", "_vendored")


def _load_licenses_module():
    # Load by file path so the test does not import the full unsloth_zoo package
    # (which pulls in the patching machinery). setuptools packages.find ships every
    # .py under the _vendored package, so a valid module here travels in the wheel.
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
