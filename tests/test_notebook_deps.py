# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The auto-installer reaches transformers' real call sites; pip is replaced by a
fake that writes the module into a temp dir on sys.path."""

import importlib
import sys
import textwrap

import pytest

pytest.importorskip("transformers")
nd = importlib.import_module("unsloth_zoo.temporary_patches.notebook_deps")

FAKE = "unsloth_fake_optional_dep"


@pytest.fixture
def fake_pip(tmp_path, monkeypatch):
    for var in ("UNSLOTH_AUTO_INSTALL", "UNSLOTH_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    (tmp_path / "site").mkdir()
    monkeypatch.setitem(nd._ALLOW_LIST, FAKE, FAKE)
    monkeypatch.setitem(nd._BY_IMPORT_NAME, FAKE, FAKE)
    monkeypatch.setattr(nd, "_attempted", set())
    calls = []

    def _pip_install(pkg):
        calls.append(pkg)
        (tmp_path / "site" / f"{pkg}.py").write_text("VALUE = 1\n")
        importlib.invalidate_caches()
        return True

    monkeypatch.setattr(nd, "_pip_install", _pip_install)
    yield calls
    sys.modules.pop(FAKE, None)


def _fake_backend(monkeypatch):
    from transformers.utils import import_utils as iu
    from functools import lru_cache

    @lru_cache
    def is_fake_available():
        return importlib.util.find_spec(FAKE) is not None

    monkeypatch.setitem(iu.BACKENDS_MAPPING, FAKE, (is_fake_available, "{0} needs the fake dep"))


def test_requires_backends_reaches_model_file_binding(fake_pip, monkeypatch):
    _fake_backend(monkeypatch)
    # Same binding TimmWrapperModel uses: `from ...utils import requires_backends`.
    from transformers.utils import requires_backends
    import transformers.models.timm_wrapper.modeling_timm_wrapper as tw
    assert getattr(tw.requires_backends, "_unsloth_patched", False)
    requires_backends(object(), [FAKE])
    assert fake_pip == [FAKE]


@pytest.mark.parametrize("env", [("UNSLOTH_AUTO_INSTALL", "0"), ("HF_HUB_OFFLINE", "true"), ("UNSLOTH_OFFLINE", "1")])
def test_opt_out_and_offline_keep_original_error(fake_pip, monkeypatch, env):
    _fake_backend(monkeypatch)
    monkeypatch.setenv(*env)
    from transformers.utils import requires_backends
    with pytest.raises(ImportError, match = "needs the fake dep"):
        requires_backends(object(), [FAKE])
    assert nd._install(FAKE) is False
    assert fake_pip == []


def test_check_imports_installs_allow_listed(fake_pip, tmp_path):
    from transformers import dynamic_module_utils as dmu
    f = tmp_path / "modeling_x.py"
    f.write_text(f"import {FAKE}\n")
    dmu.check_imports(str(f))
    assert fake_pip == [FAKE]


def test_check_imports_refuses_mixed_unlisted(fake_pip, tmp_path):
    from transformers import dynamic_module_utils as dmu
    f = tmp_path / "modeling_y.py"
    f.write_text(f"import {FAKE}\nimport unsloth_not_allow_listed_dep\n")
    with pytest.raises(ImportError, match = "unsloth_not_allow_listed_dep"):
        dmu.check_imports(str(f))
    assert fake_pip == []


def test_get_class_in_module_installs_sibling_import(fake_pip, tmp_path, monkeypatch):
    from transformers import dynamic_module_utils as dmu
    # A submodule import check_imports never scanned (DeepSeek-OCR deepencoder.py -> easydict).
    (tmp_path / "modeling_z.py").write_text(f"from {FAKE} import VALUE\nclass ZModel:\n    value = VALUE\n")
    monkeypatch.setattr(dmu, "HF_MODULES_CACHE", str(tmp_path))
    monkeypatch.delitem(sys.modules, "modeling_z", raising = False)
    assert dmu.get_class_in_module("ZModel", "modeling_z.py").value == 1
    assert fake_pip == [FAKE]


def test_guarded_module_import_is_rebound(fake_pip, tmp_path, monkeypatch):
    # modeling_timm_wrapper shape: the guarded `import timm` ran while timm was missing.
    _fake_backend(monkeypatch)
    src = tmp_path / "modeling_fake.py"
    src.write_text(textwrap.dedent(f"""
        from transformers.utils import requires_backends
        def is_{FAKE}_available():
            return False
        if is_{FAKE}_available():
            import {FAKE}
        class FakeModel:
            def __init__(self):
                requires_backends(self, ["{FAKE}"])
                self.value = {FAKE}.VALUE
    """))
    name = "transformers.models.unsloth_fake.modeling_fake"
    spec = importlib.util.spec_from_file_location(name, src)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.FakeModel().value == 1
    assert fake_pip == [FAKE]
