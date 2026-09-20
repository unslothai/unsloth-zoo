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

"""The GGUF converter child's `gguf` resolution (unsloth#3581): child environment,
requirement scan, candidate ranking and failure diagnosis. No network, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from unittest import mock

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location("llama_cpp_child_env_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_llama_cpp_module()


def _make_gguf_py(root, *, package=True, symbols=(), version=None):
    """`symbols` are module-level names defined in gguf/__init__.py."""
    root = Path(root)
    gguf_py = root / "gguf-py"
    if package:
        pkg = gguf_py / "gguf"
        pkg.mkdir(parents=True, exist_ok=True)
        body = "".join(f"{name} = object()\n" for name in symbols)
        (pkg / "__init__.py").write_text(body or "# empty gguf package\n")
    else:
        gguf_py.mkdir(parents=True, exist_ok=True)
    if version is not None:
        (gguf_py / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    return gguf_py


# --- _converter_child_env ---

def test_child_env_is_a_copy_of_the_parent_environment(mod, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CHILD_ENV_CANARY", "kept")
    env = mod._converter_child_env()
    assert env["UNSLOTH_CHILD_ENV_CANARY"] == "kept"
    assert "PATH" in env
    assert env is not os.environ


def test_child_env_without_a_pin_changes_nothing(mod, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/already/here")
    monkeypatch.delenv("NO_LOCAL_GGUF", raising=False)
    env = mod._converter_child_env(None)
    assert env["PYTHONPATH"] == "/already/here"
    assert "NO_LOCAL_GGUF" not in env


def test_child_env_prepends_the_pin_and_keeps_existing_entries(mod, monkeypatch, tmp_path):
    existing = os.pathsep.join(["/first", "/second"])
    monkeypatch.setenv("PYTHONPATH", existing)
    gguf_py = _make_gguf_py(tmp_path, symbols=("Metadata",))
    env = mod._converter_child_env(str(gguf_py))
    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(gguf_py)
    assert entries[1:] == ["/first", "/second"]
    # Without this the entrypoint's own sys.path.insert(1, ...) still wins.
    assert env["NO_LOCAL_GGUF"] == "1"


def test_child_env_uses_the_platform_path_separator(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/only")
    gguf_py = _make_gguf_py(tmp_path)
    env = mod._converter_child_env(str(gguf_py))
    assert env["PYTHONPATH"] == os.pathsep.join([str(gguf_py), "/only"])
    assert ";" not in env["PYTHONPATH"] or os.pathsep == ";"


def test_child_env_is_idempotent(mod, monkeypatch, tmp_path):
    gguf_py = _make_gguf_py(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    once = mod._converter_child_env(str(gguf_py))
    twice = mod._converter_child_env(str(gguf_py), base_env=once)
    assert once["PYTHONPATH"] == twice["PYTHONPATH"]
    assert twice["PYTHONPATH"].count(str(gguf_py)) == 1


def test_child_env_does_not_duplicate_an_entry_already_present(mod, monkeypatch, tmp_path):
    gguf_py = _make_gguf_py(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/x", str(gguf_py)]))
    env = mod._converter_child_env(str(gguf_py))
    assert env["PYTHONPATH"].split(os.pathsep) == [str(gguf_py), "/x"]


# --- _importable_gguf_py ---

def test_importable_gguf_py_accepts_a_real_package(mod, tmp_path):
    gguf_py = _make_gguf_py(tmp_path)
    assert mod._importable_gguf_py(str(tmp_path)) == str(gguf_py)


def test_importable_gguf_py_rejects_a_dir_without_the_package(mod, tmp_path):
    _make_gguf_py(tmp_path, package=False)
    assert mod._importable_gguf_py(str(tmp_path)) is None


def test_importable_gguf_py_handles_missing_and_empty_input(mod, tmp_path):
    assert mod._importable_gguf_py(str(tmp_path / "nope")) is None
    assert mod._importable_gguf_py("") is None
    assert mod._importable_gguf_py(None) is None


# --- requirement scanning ---

def test_module_level_imports_are_certain(mod):
    source = b"from gguf.vocab import MistralTokenizerType\nimport gguf.utility\n"
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.vocab.MistralTokenizerType" in certain
    assert "gguf.utility" in certain
    assert not advisory


def test_class_body_attribute_chains_are_certain(mod):
    source = b"import gguf\nclass Gemma4Model:\n    model_arch = gguf.MODEL_ARCH.GEMMA4\n"
    certain, _ = mod._gguf_requirements_from_source(source)
    assert "gguf.MODEL_ARCH.GEMMA4" in certain


def test_function_body_attribute_chains_are_advisory(mod):
    source = b"import gguf\ndef later():\n    return gguf.MODEL_ARCH.MAYBE\n"
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.MODEL_ARCH.MAYBE" in advisory
    assert "gguf.MODEL_ARCH.MAYBE" not in certain


def test_try_guarded_requirements_are_advisory(mod):
    source = b"try:\n    from gguf.new import Thing\nexcept ImportError:\n    Thing = None\n"
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.new.Thing" in advisory
    assert not certain


def test_type_checking_imports_are_advisory(mod):
    """A TYPE_CHECKING branch never executes, so its imports are advisory."""
    source = (
        b"from typing import TYPE_CHECKING\n"
        b"if TYPE_CHECKING:\n"
        b"    from gguf.future import TypeOnly\n"
    )
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.future.TypeOnly" in advisory
    assert not certain

    for test in (b"typing.TYPE_CHECKING", b"t.TYPE_CHECKING", b"False"):
        certain, advisory = mod._gguf_requirements_from_source(
            b"if " + test + b":\n    from gguf.future import TypeOnly\n"
        )
        assert "gguf.future.TypeOnly" in advisory, test
        assert not certain, test

    # The ELSE branch really runs, so it stays certain.
    source = (
        b"if TYPE_CHECKING:\n"
        b"    from gguf.future import TypeOnly\n"
        b"else:\n"
        b"    from gguf.vocab import Real\n"
    )
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.vocab.Real" in certain
    assert "gguf.future.TypeOnly" in advisory

    # An ordinary guard runs AT MOST ONE branch, so neither is certain.
    source = b"import os\nif os.environ.get('X'):\n    from gguf.vocab import Real\n"
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.vocab.Real" not in certain
    assert "gguf.vocab.Real" in advisory

    source = (
        b"import sys\n"
        b"if sys.platform == 'win32':\n"
        b"    from gguf.vocab import WindowsOnly\n"
        b"else:\n"
        b"    from gguf.vocab import PosixOnly\n"
    )
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert not certain, certain
    assert {"gguf.vocab.WindowsOnly", "gguf.vocab.PosixOnly"} <= set(advisory)

    # The TEST is evaluated either way, so a gguf symbol inside it stays certain.
    source = b"import gguf\nif gguf.HAS_FEATURE:\n    X = 1\n"
    certain, _ = mod._gguf_requirements_from_source(source)
    assert "gguf.HAS_FEATURE" in certain, certain


def test_unrelated_names_are_ignored(mod):
    source = b"import numpy\nother = notgguf.MODEL_ARCH.X\nfrom ggufextra import Y\n"
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert not certain and not advisory


def test_unparseable_source_contributes_nothing(mod):
    certain, advisory = mod._gguf_requirements_from_source(b"def broken(:\n")
    assert not certain and not advisory


def test_conversion_package_is_scanned_only_when_imported(mod, tmp_path):
    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "base.py").write_text("from gguf.vocab import MistralTokenizerType\n")

    package_entry = tmp_path / "package_entry.py"
    package_entry.write_text("from conversion import get_model_class\n")
    certain, _ = mod._converter_gguf_requirements(str(package_entry))
    assert "gguf.vocab.MistralTokenizerType" in certain

    # A monolith entrypoint does not import conversion/, so it places no requirement on it.
    monolith_entry = tmp_path / "monolith_entry.py"
    monolith_entry.write_text("import gguf\n")
    certain, _ = mod._converter_gguf_requirements(str(monolith_entry))
    assert "gguf.vocab.MistralTokenizerType" not in certain


# --- architecture scoping of the requirement scan ---

def _make_conversion_tree(tmp_path):
    """An entrypoint plus a conversion/ package shaped like llama.cpp's own."""
    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "__init__.py").write_text(textwrap.dedent("""
        from .base import ModelBase
        TEXT_MODEL_MAP: dict[str, str] = {
            "Gemma3ForCausalLM": "gemma",
            "AfmoeForCausalLM": "afmoe",
        }
        MMPROJ_MODEL_MAP: dict[str, str] = {
            "Gemma3ForConditionalGeneration": "gemma",
        }
    """))
    (conversion / "base.py").write_text("import gguf\nBASE = gguf.Metadata\n")
    (conversion / "gemma.py").write_text(
        "import gguf\n\nclass G:\n    model_arch = gguf.MODEL_ARCH.GEMMA3\n"
    )
    (conversion / "afmoe.py").write_text(
        "import gguf\n\nclass A:\n    model_arch = gguf.MODEL_ARCH.AFMOE\n"
    )
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("from conversion import get_model_class\nimport gguf\n")
    return entry


def test_only_this_architectures_conversion_module_is_a_certain_requirement(mod, tmp_path):
    entry = _make_conversion_tree(tmp_path)

    certain, advisory = mod._converter_gguf_requirements(
        str(entry), "Gemma3ForCausalLM",
    )
    assert "gguf.MODEL_ARCH.GEMMA3" in certain
    assert "gguf.Metadata" in certain, "conversion/base.py is imported eagerly"
    # Another architecture's module is advisory only.
    assert "gguf.MODEL_ARCH.AFMOE" not in certain
    assert "gguf.MODEL_ARCH.AFMOE" in advisory


def test_an_mmproj_architecture_resolves_through_the_mmproj_map(mod, tmp_path):
    entry = _make_conversion_tree(tmp_path)
    certain, _ = mod._converter_gguf_requirements(
        str(entry), "Gemma3ForConditionalGeneration",
    )
    assert "gguf.MODEL_ARCH.GEMMA3" in certain


def test_an_unknown_architecture_scopes_to_the_eager_modules_only(mod, tmp_path):
    """An unknown architecture scopes to the eager modules only."""
    entry = _make_conversion_tree(tmp_path)
    for architecture in (None, "SomethingNobodySupports"):
        certain, advisory = mod._converter_gguf_requirements(str(entry), architecture)
        assert "gguf.Metadata" in certain
        assert "gguf.MODEL_ARCH.GEMMA3" not in certain
        assert "gguf.MODEL_ARCH.AFMOE" not in certain
        assert {"gguf.MODEL_ARCH.GEMMA3", "gguf.MODEL_ARCH.AFMOE"} <= set(advisory)


def test_an_unrelated_missing_architecture_does_not_switch_converters(
    mod, tmp_path, monkeypatch,
):
    """unsloth#3581: a gguf-py behind on an architecture nobody is converting must
    leave the requested converter and the environment untouched."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    entry = _make_conversion_tree(tmp_path)
    # MODEL_ARCH knows GEMMA3 but not AFMOE, and AFMOE is not in this export's path.
    _make_gguf_py(
        tmp_path, symbols=("Metadata",), version="0.19.0",
    )
    (tmp_path / "gguf-py" / "gguf" / "__init__.py").write_text(textwrap.dedent("""
        Metadata = object()
        class MODEL_ARCH:
            GEMMA3 = "gemma3"
    """))
    # A sibling entrypoint exists, so a wrongly blocked candidate 0 could downgrade.
    (tmp_path / "convert_hf_to_gguf.py").write_text("import gguf\n")

    chosen, pin, _report = mod._resolve_converter_and_gguf(
        str(entry), sys.executable, "Gemma3ForCausalLM",
    )
    assert chosen == str(entry), "switched converters over an unrelated architecture"
    assert pin is None


def test_a_missing_architecture_on_this_path_still_switches(
    mod, tmp_path, monkeypatch, capsys,
):
    """When the gguf-py cannot satisfy the architecture being converted, falling
    back is correct."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(mod, "_installed_gguf_tree", lambda *args, **kwargs: None)
    entry = _make_conversion_tree(tmp_path)
    (tmp_path / "gguf-py").mkdir(exist_ok=True)
    pkg = tmp_path / "gguf-py" / "gguf"
    pkg.mkdir(parents=True, exist_ok=True)
    pkg.joinpath("__init__.py").write_text(textwrap.dedent("""
        Metadata = object()
        class MODEL_ARCH:
            AFMOE = "afmoe"
    """))
    sibling = tmp_path / "convert_hf_to_gguf.py"
    sibling.write_text("import gguf\nOLD = gguf.Metadata\n")

    chosen, _pin, _report = mod._resolve_converter_and_gguf(
        str(entry), sys.executable, "Gemma3ForCausalLM",
    )
    assert chosen == str(sibling)
    # The announcement names the symbol that stopped this export's own module.
    announced = capsys.readouterr().out
    assert "gguf.MODEL_ARCH.GEMMA3" in announced
    assert "gguf.MODEL_ARCH.AFMOE" not in announced


def test_the_diagnosis_names_the_symbol_that_actually_stopped_the_import(
    mod, tmp_path, monkeypatch,
):
    """Scoped requirements name the real blocker, not three irrelevant symbols."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    entry = _make_conversion_tree(tmp_path)
    pkg = tmp_path / "gguf-py" / "gguf"
    pkg.mkdir(parents=True, exist_ok=True)
    pkg.joinpath("__init__.py").write_text(textwrap.dedent("""
        Metadata = object()
        class MODEL_ARCH:
            AFMOE = "afmoe"
    """))
    (tmp_path / "gguf-py" / "pyproject.toml").write_text('[project]\nversion = "0.17.1"\n')

    note = mod._gguf_skew_diagnosis(
        str(entry), sys.executable, None, "Gemma3ForCausalLM",
    )
    assert "gguf.MODEL_ARCH.GEMMA3" in note
    assert "gguf.MODEL_ARCH.AFMOE" not in note


# --- the child probe ---

def test_probe_reports_the_tree_the_entrypoint_would_self_locate(mod, tmp_path):
    """The probe resolves gguf through the entrypoint's own sibling gguf-py."""
    gguf_py = _make_gguf_py(tmp_path, symbols=("Present",), version="0.17.1")
    entry = tmp_path / "convert_hf_to_gguf.py"
    entry.write_text("import gguf\n")

    env = dict(os.environ)
    env.pop("NO_LOCAL_GGUF", None)
    env.pop("PYTHONPATH", None)
    report = mod._probe_child_gguf(
        sys.executable, env, ("gguf.Present", "gguf.Absent"), str(entry),
    )
    assert report is not None
    assert Path(report["location"]).parent == gguf_py / "gguf"
    assert report["missing"] == ["gguf.Absent"]
    assert report["version"] == "0.17.1"


def test_probe_ignores_json_it_did_not_write(mod, tmp_path):
    """Foreign JSON on the child's stdout must not be read as the probe report."""
    gguf_py = _make_gguf_py(tmp_path, symbols=("Present",))
    entry = tmp_path / "convert_hf_to_gguf.py"
    (gguf_py / "gguf" / "__init__.py").write_text(
        'Present = object()\nprint("{}")\nprint(\'{"status": "ok"}\')\n'
    )
    entry.write_text("import gguf\n")

    env = dict(os.environ)
    env.pop("NO_LOCAL_GGUF", None)
    env.pop("PYTHONPATH", None)
    report = mod._probe_child_gguf(
        sys.executable, env, ("gguf.Present", "gguf.Absent"), str(entry),
    )
    assert report is not None
    assert report.get("unsloth_gguf_probe") == 1
    assert report["missing"] == ["gguf.Absent"]


def test_probe_returns_none_when_only_foreign_json_is_printed(mod):
    """No report of ours means None, and the resolver leaves the request alone."""
    completed = types.SimpleNamespace(stdout = '{}\n{"missing": []}\n', returncode = 0)
    with mock.patch.object(mod.subprocess, "run", return_value = completed):
        assert mod._probe_child_gguf(sys.executable, dict(os.environ), ("gguf.X",)) is None


def test_probe_reports_an_unimportable_gguf(mod, tmp_path):
    entry = tmp_path / "convert_hf_to_gguf.py"
    entry.write_text("import gguf\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "empty")
    env["PYTHONNOUSERSITE"] = "1"
    report = mod._probe_child_gguf(sys.executable, env, ("gguf.Thing",), str(entry))
    assert report is not None
    # Absent or present, what matters is that the probe always returns a report.
    assert "missing" in report


def test_probe_returns_none_when_it_cannot_run(mod):
    assert mod._probe_child_gguf("/nonexistent/python", dict(os.environ), ("gguf.X",)) is None


# --- candidate ranking ---

def test_candidates_start_with_the_requested_pair_untouched(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\n")
    candidates = mod._gguf_candidate_converters(str(entry))
    assert candidates[0] == (str(entry), None, "as configured")


def test_candidates_offer_the_sibling_then_the_installed_gguf_py(mod, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle_gguf_py = _make_gguf_py(bundle, symbols=("Metadata",))
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    sibling_gguf_py = _make_gguf_py(scripts, symbols=("Metadata",))
    entry = scripts / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\n")

    pins = [pin for _, pin, _ in mod._gguf_candidate_converters(str(entry))]
    assert pins[0] is None
    assert str(sibling_gguf_py) in pins
    assert str(bundle_gguf_py) in pins
    assert pins.index(str(sibling_gguf_py)) < pins.index(str(bundle_gguf_py))


def test_candidates_offer_the_checkouts_own_converter_last(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    _make_gguf_py(tmp_path, symbols=("Metadata",))
    (tmp_path / "convert_hf_to_gguf.py").write_text("import gguf\n")
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\n")

    candidates = mod._gguf_candidate_converters(str(entry))
    assert candidates[-1][0] == str(tmp_path / "convert_hf_to_gguf.py")
    # Never offer the requested converter back as its own fallback.
    assert [c for c, _, _ in candidates[1:]].count(str(entry)) == len(candidates) - 2


def test_resolver_leaves_a_consistent_install_alone(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    _make_gguf_py(tmp_path, symbols=("Metadata", "MODEL_ARCH"))
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\nX = gguf.Metadata\n")

    chosen, pin, report = mod._resolve_converter_and_gguf(str(entry), sys.executable)
    assert chosen == str(entry)
    assert pin is None
    assert "NO_LOCAL_GGUF" not in mod._converter_child_env(pin)
    assert report is not None and not report["missing"]


def test_resolver_falls_back_to_the_co_versioned_converter(mod, tmp_path, monkeypatch, capsys):
    """A newer downloaded entrypoint, an older sibling gguf-py, and the checkout's
    own co-versioned converter beside it."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(mod, "_installed_gguf_tree", lambda *args, **kwargs: None)
    _make_gguf_py(tmp_path, symbols=("Metadata",), version="0.17.1")
    (tmp_path / "convert_hf_to_gguf.py").write_text("import gguf\nX = gguf.Metadata\n")
    newer = tmp_path / "unsloth_convert_hf_to_gguf.py"
    newer.write_text("import gguf\nX = gguf.SafetensorsLocal\n")

    chosen, _pin, _ = mod._resolve_converter_and_gguf(str(newer), sys.executable)
    assert chosen == str(tmp_path / "convert_hf_to_gguf.py")
    message = capsys.readouterr().out
    assert "Falling back" in message
    assert "gguf.SafetensorsLocal" in message


def test_resolver_pins_a_satisfying_gguf_py_from_elsewhere(mod, tmp_path, monkeypatch):
    """With no sibling tree, the installed bundle's satisfying tree is pinned."""
    bundle = tmp_path / "bundle"
    bundle_gguf_py = _make_gguf_py(bundle, symbols=("Metadata", "SafetensorsLocal"))
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(bundle))

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entry = scripts / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\nX = gguf.SafetensorsLocal\n")

    stale = _make_gguf_py(tmp_path / "stale", symbols=("Metadata",))
    monkeypatch.setenv("PYTHONPATH", str(stale))

    chosen, pin, report = mod._resolve_converter_and_gguf(str(entry), sys.executable)
    assert chosen == str(entry)
    assert pin == str(bundle_gguf_py)
    env = mod._converter_child_env(pin)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(bundle_gguf_py)
    assert env["NO_LOCAL_GGUF"] == "1"
    assert not report["missing"]


def test_resolver_does_not_move_for_an_advisory_miss_only(mod, tmp_path, monkeypatch):
    """An advisory-only miss must not switch a working install onto an older converter."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    _make_gguf_py(tmp_path, symbols=("Metadata",), version="0.17.1")
    (tmp_path / "convert_hf_to_gguf.py").write_text("import gguf\nX = gguf.Metadata\n")
    newer = tmp_path / "unsloth_convert_hf_to_gguf.py"
    newer.write_text("import gguf\ndef maybe():\n    return gguf.Absent\n")

    chosen, _pin, _ = mod._resolve_converter_and_gguf(str(newer), sys.executable)
    assert chosen == str(newer)


def test_the_best_candidate_wins_not_the_first_that_clears_the_certain_bar(
    mod, tmp_path, monkeypatch,
):
    """A candidate can clear every certain name and still fail on an advisory one
    that runs in practice, so the best candidate must win, not the first."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "__init__.py").write_text(
        'from .base import ModelBase\nTEXT_MODEL_MAP = {"L": "llama"}\n'
        'MMPROJ_MODEL_MAP = {}\n'
    )
    # Certain (module scope) and advisory (function scope) in the same module.
    (conversion / "base.py").write_text(
        "import gguf\n"
        "CERTAIN = gguf.Metadata\n"
        "def run(data):\n"
        "    return gguf.RunsInPractice(data)\n"
    )
    (conversion / "llama.py").write_text("import gguf\n")
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text(
        "from conversion import get_model_class\nimport gguf\nE = gguf.OnlyNew\n"
    )

    # Missing a certain name, so candidate 0 is out.
    _make_gguf_py(tmp_path, symbols=("Metadata",))
    # Clears every certain name but not the advisory one.
    plausible = tmp_path / "plausible"
    _make_gguf_py(plausible, symbols=("Metadata", "OnlyNew"))
    # Clears everything.
    coversioned = tmp_path / "coversioned"
    _make_gguf_py(coversioned, symbols=("Metadata", "OnlyNew", "RunsInPractice"))

    monkeypatch.setattr(
        mod, "_gguf_candidate_converters",
        lambda _requested: [
            (str(entry), None, "as configured"),
            (str(entry), str(plausible / "gguf-py"), "a plausible gguf-py"),
            (str(entry), str(coversioned / "gguf-py"), "the co-versioned gguf-py"),
        ],
    )
    chosen, pin, _report = mod._resolve_converter_and_gguf(str(entry), sys.executable, "L")
    assert chosen == str(entry)
    assert pin == str(coversioned / "gguf-py"), "took a plausible tree over the matching one"


def test_resolver_keeps_the_request_when_nothing_satisfies_it(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    _make_gguf_py(tmp_path, symbols=("Metadata",))
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\nX = gguf.NeverExists\n")

    chosen, pin, report = mod._resolve_converter_and_gguf(str(entry), sys.executable)
    assert chosen == str(entry)
    assert pin is None
    assert "NO_LOCAL_GGUF" not in mod._converter_child_env(pin)
    assert "gguf.NeverExists" in report["missing"]


# --- failure diagnosis ---

@pytest.mark.parametrize("text", [
    "ImportError: cannot import name 'MistralTokenizerType' from 'gguf.vocab' (/x/gguf/vocab.py)",
    "AttributeError: module 'gguf.utility' has no attribute 'SafetensorsLocal'",
    "ModuleNotFoundError: No module named 'gguf'",
])
def test_skew_signatures_are_recognised(mod, text):
    """These name a gguf module outright, so no requirement list is needed."""
    assert mod._looks_like_gguf_skew(text) is True


@pytest.mark.parametrize("text", [
    "",
    None,
    "RuntimeError: not enough disk space",
    "AttributeError: type object 'Foo' has no attribute 'bar'",
])
def test_unrelated_failures_are_not_diagnosed_as_skew(mod, text):
    assert mod._looks_like_gguf_skew(text) is False


# A bare attribute error names the class but not the package, so it is only ours
# when the class is one the converter asked gguf for.
_ARCH_ATTRIBUTE_ERROR = (
    "AttributeError: type object 'MODEL_ARCH' has no attribute 'GEMMA4'\n"
    "  gguf.MODEL_ARCH.GEMMA4"
)


def test_an_attribute_error_on_a_required_gguf_class_is_skew(mod):
    assert mod._looks_like_gguf_skew(
        _ARCH_ATTRIBUTE_ERROR, ("gguf.MODEL_ARCH.GEMMA4",)
    ) is True


def test_an_attribute_error_on_a_model_side_class_is_not_skew(mod):
    """"gguf" appears in every failure's output, so a model-side AttributeError must
    not be diagnosed as skew."""
    text = (
        "INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only\n"
        "AttributeError: type object 'Gemma3Config' has no attribute 'rope_local'"
    )
    assert mod._looks_like_gguf_skew(text, ("gguf.MODEL_ARCH.GEMMA4",)) is False


def test_an_attribute_error_without_requirements_is_not_skew(mod):
    assert mod._looks_like_gguf_skew(_ARCH_ATTRIBUTE_ERROR) is False


def test_diagnosis_names_the_gguf_the_symbols_and_the_remedy(mod, tmp_path):
    gguf_py = _make_gguf_py(tmp_path, symbols=("Metadata",), version="0.17.1")
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\nX = gguf.SafetensorsLocal\n")

    note = mod._gguf_skew_diagnosis(str(entry), sys.executable)
    assert "gguf version skew" in note
    assert str(entry) in note
    assert str(gguf_py) in note
    assert "0.17.1" in note
    assert "gguf.SafetensorsLocal" in note
    assert "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR" in note


def test_diagnosis_never_raises(mod):
    assert mod._gguf_skew_diagnosis("/nonexistent/convert.py", "/nonexistent/python") == ""


def test_report_summary_handles_every_shape(mod):
    assert mod._gguf_report_summary(None, ()) == "reason unknown"
    assert "not importable" in mod._gguf_report_summary({"error": "ImportError: x"}, ())
    assert mod._gguf_report_summary({"missing": ["gguf.A"]}, ["gguf.B"]) == "missing gguf.B"
    # Only the blocking subset is named.
    assert mod._gguf_report_summary({"missing": ["gguf.A"]}, ()) == "reason unknown"
    assert mod._gguf_report_summary({"missing": []}, ()) == "reason unknown"


# --- wiring ---

def test_every_converter_launch_passes_an_explicit_env(mod):
    """A launch site without env= silently reintroduces the bug."""
    import inspect
    source = inspect.getsource(mod.convert_to_gguf)
    assert "_resolve_converter_and_gguf(" in source
    runs = [line for line in source.splitlines() if "subprocess.run(command" in line]
    assert runs, "no converter launch found"
    assert source.count("env=_converter_child_env(_gguf_py_pin)") == len(runs)
    # Built at launch time, never frozen at preflight time.
    assert "_converter_env =" not in source


def test_failure_path_attaches_the_diagnosis(mod):
    import inspect
    source = inspect.getsource(mod.convert_to_gguf)
    assert "_looks_like_gguf_skew(" in source
    assert "_gguf_skew_diagnosis(" in source
    # The requirement list keeps a model-side AttributeError out of the skew advice.
    assert "_looks_like_gguf_skew(captured)" not in source


# --- end to end, skipped unless the real toolchain is staged ---

_TOOLCHAIN = Path(os.environ.get("UNSLOTH_TEST_LLAMACPP_DIR") or os.devnull)
_MODEL = Path(os.environ.get("UNSLOTH_TEST_GGUF_MODEL") or os.devnull)


def _staged(root, name):
    # An unreadable parent raises EACCES, which at module scope would fail
    # collection of the whole file rather than skip this one test.
    try:
        return (root / name).is_file()
    except OSError:
        return False


@pytest.mark.skipif(
    not _staged(_TOOLCHAIN, "convert_hf_to_gguf.py") or not _staged(_MODEL, "config.json"),
    reason="set UNSLOTH_TEST_LLAMACPP_DIR and UNSLOTH_TEST_GGUF_MODEL to run this",
)
def test_end_to_end_conversion_survives_a_stale_ambient_gguf(mod, tmp_path, monkeypatch):
    """With the bundle's tree pinned, a stale gguf on PYTHONPATH does not decide
    the child's import."""
    import shutil

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(_TOOLCHAIN / "convert_hf_to_gguf.py", scripts / "unsloth_convert_hf_to_gguf.py")
    shutil.copytree(_TOOLCHAIN / "conversion", scripts / "conversion")

    stale = tmp_path / "stale"
    shutil.copytree(_TOOLCHAIN / "gguf-py", stale / "gguf-py")
    vocab = stale / "gguf-py" / "gguf" / "vocab.py"
    vocab.write_text(vocab.read_text().replace("MistralTokenizerType", "_Removed"))

    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(_TOOLCHAIN))
    monkeypatch.setenv("PYTHONPATH", str(stale / "gguf-py"))
    monkeypatch.setenv("UNSLOTH_AUTO_INSTALL", "0")

    (tmp_path / "out").mkdir()
    files, _ = mod.convert_to_gguf(
        model_name=str(tmp_path / "out" / "model"),
        input_folder=str(_MODEL),
        model_dtype="bf16",
        quantization_type="bf16",
        converter_location=str(scripts / "unsloth_convert_hf_to_gguf.py"),
    )
    assert files and Path(files[0]).is_file()
    assert Path(files[0]).stat().st_size > 1024 * 1024
    # The stale tree is still first on the parent's PYTHONPATH, so success
    # can only have come from the pin.
    assert os.environ["PYTHONPATH"] == str(stale / "gguf-py")


def test_the_preflight_returns_a_pin_not_a_frozen_environment(mod, tmp_path, monkeypatch):
    """The preflight returns a directory or None, never a frozen environment."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    _make_gguf_py(tmp_path, symbols=("Metadata",))
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\nX = gguf.Metadata\n")
    _chosen, pin, _report = mod._resolve_converter_and_gguf(str(entry), sys.executable)
    assert pin is None or isinstance(pin, str)


# --- The installed gguf wheel as a candidate ---

def test_the_installed_gguf_wheel_is_used_when_the_sibling_tree_is_too_old(
    mod, tmp_path, monkeypatch, capsys,
):
    """An installed wheel newer than the sibling tree must be pinned, keeping the
    newer converter rather than dropping back to an older one."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    wheel = _make_gguf_py(tmp_path / "site-packages", symbols=("Metadata", "SafetensorsLocal"))
    monkeypatch.setattr(mod, "_installed_gguf_tree", lambda *a, **kw: str(wheel))

    _make_gguf_py(tmp_path, symbols=("Metadata",), version="0.17.1")
    (tmp_path / "convert_hf_to_gguf.py").write_text("import gguf\nX = gguf.Metadata\n")
    newer = tmp_path / "unsloth_convert_hf_to_gguf.py"
    newer.write_text("import gguf\nX = gguf.SafetensorsLocal\n")

    chosen, pin, _report = mod._resolve_converter_and_gguf(str(newer), sys.executable)

    assert chosen == str(newer), "the newer converter must be kept"
    assert pin == str(wheel), pin
    # Keeping the requested converter is not a fallback, so it is not announced.
    assert "Falling back" not in capsys.readouterr().out


def test_the_installed_wheel_is_not_probed_when_the_request_already_works(
    mod, tmp_path, monkeypatch,
):
    """The extra probe is a subprocess, and a working install must not pay for it."""
    called = {"hit": False}
    def _trap(*args, **kwargs):
        called["hit"] = True
        return None
    monkeypatch.setattr(mod, "_installed_gguf_tree", _trap)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))

    # Its own sibling gguf-py, so this does not assert on whatever `gguf` the
    # developer's venv happens to ship.
    _make_gguf_py(tmp_path, symbols=("GGUFWriter",))

    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("import gguf\nX = gguf.GGUFWriter\n")
    chosen, _pin, _report = mod._resolve_converter_and_gguf(str(converter), sys.executable)

    assert chosen == str(converter)
    assert called["hit"] is False


def test_installed_gguf_tree_reads_the_child_not_the_parent(mod, monkeypatch, tmp_path):
    """The tree is what the CHILD resolves with the sibling tree suppressed, and only
    when a real package sits inside it."""
    captured = {}
    tree = tmp_path / "site-packages"
    (tree / "gguf").mkdir(parents=True)
    (tree / "gguf" / "__init__.py").write_text("# gguf\n")

    def _fake_probe(python_exe, env, requirements, converter_location=None, timeout=120):
        captured["env"] = env
        captured["converter"] = converter_location
        return {"location": str(tree / "gguf" / "__init__.py")}
    monkeypatch.setattr(mod, "_probe_child_gguf", _fake_probe)

    assert mod._installed_gguf_tree(sys.executable) == str(tree)
    assert captured["env"]["NO_LOCAL_GGUF"] == "1"
    assert captured["converter"] is None

    monkeypatch.setattr(mod, "_probe_child_gguf", lambda *a, **kw: None)
    assert mod._installed_gguf_tree(sys.executable) is None
    monkeypatch.setattr(mod, "_probe_child_gguf",
                        lambda *a, **kw: {"location": str(tmp_path / "nothing" / "gguf" / "__init__.py")})
    assert mod._installed_gguf_tree(sys.executable) is None


def test_signature_expressions_are_certain(mod):
    """A default, a decorator and an annotation are evaluated while the module is
    IMPORTED, so they are certain."""
    source = (
        b"import gguf\n"
        b"@gguf.register\n"
        b"def convert(kind = gguf.NEW_KIND, *, mode = gguf.MODE.FAST) -> gguf.Result:\n"
        b"    return gguf.MODEL_ARCH.MAYBE\n"
    )
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.register" in certain
    assert "gguf.NEW_KIND" in certain
    assert "gguf.MODE.FAST" in certain
    if sys.version_info < (3, 14):
        # PEP 649 defers this one from 3.14 on.
        assert "gguf.Result" in certain
    assert "gguf.MODEL_ARCH.MAYBE" in advisory
    assert "gguf.MODEL_ARCH.MAYBE" not in certain


def test_a_lambda_default_is_certain_and_its_body_is_not(mod):
    source = b"import gguf\nf = lambda kind = gguf.KIND: gguf.LATER.value\n"
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.KIND" in certain
    assert "gguf.LATER.value" in advisory


def test_postponed_annotations_are_not_evaluated(mod):
    """PEP 563 annotations cost nothing at import time. A default still does."""
    source = (
        b"from __future__ import annotations\n"
        b"import gguf\n"
        b"def convert(kind = gguf.NEW_KIND) -> gguf.Result:\n"
        b"    return None\n"
    )
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.NEW_KIND" in certain
    assert "gguf.Result" not in certain


# --- PEP 649: from 3.14 an annotation is deferred with no future import needed ---

def test_an_annotation_is_deferred_on_314_without_the_future_import(mod, monkeypatch):
    """On 3.14 the return annotation is never evaluated at import."""
    source = (
        b"import gguf\n"
        b"def convert(kind = gguf.NEW_KIND) -> gguf.Result:\n"
        b"    return None\n"
    )
    monkeypatch.setattr(mod.sys, "version_info", (3, 14, 0, "final", 0))
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.NEW_KIND" in certain, "a default is still evaluated eagerly on 3.14"
    assert "gguf.Result" not in certain


def test_the_same_annotation_is_eager_on_313(mod, monkeypatch):
    """Below 3.14 the annotation really does run at import."""
    source = (
        b"import gguf\n"
        b"def convert(kind = gguf.NEW_KIND) -> gguf.Result:\n"
        b"    return None\n"
    )
    monkeypatch.setattr(mod.sys, "version_info", (3, 13, 0, "final", 0))
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.Result" in certain


def test_an_argument_annotation_is_deferred_on_314(mod, monkeypatch):
    source = (
        b"import gguf\n"
        b"def convert(kind: gguf.Kind, *rest: gguf.Rest, **kw: gguf.Kw) -> None:\n"
        b"    return None\n"
    )
    monkeypatch.setattr(mod.sys, "version_info", (3, 14, 0, "final", 0))
    certain, advisory = mod._gguf_requirements_from_source(source)
    for symbol in ("gguf.Kind", "gguf.Rest", "gguf.Kw"):
        assert symbol not in certain


def test_a_decorator_is_still_certain_on_314(mod, monkeypatch):
    """PEP 649 defers annotations and nothing else."""
    source = (
        b"import gguf\n"
        b"@gguf.register\n"
        b"def convert(kind: gguf.Kind) -> gguf.Result:\n"
        b"    return None\n"
    )
    monkeypatch.setattr(mod.sys, "version_info", (3, 14, 0, "final", 0))
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.register" in certain


def test_a_signature_under_a_try_is_still_advisory(mod):
    """The try covers what is inside it, signature included."""
    source = (
        b"import gguf\n"
        b"try:\n"
        b"    def convert(kind = gguf.NEW_KIND):\n"
        b"        return None\n"
        b"except AttributeError:\n"
        b"    convert = None\n"
    )
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.NEW_KIND" in advisory
    assert "gguf.NEW_KIND" not in certain


def _make_older_sibling_converter(tmp_path, *, maps_the_architecture, mmproj_only = False):
    """A checkout's own package-based converter, co-versioned with the sibling gguf-py."""
    conversion = tmp_path / "conversion"
    conversion.mkdir(exist_ok = True)
    mapped = '"Gemma3ForCausalLM": "gemma",' if maps_the_architecture else ""
    projector = mapped if mmproj_only else ""
    if mmproj_only:
        mapped = ""
    (conversion / "__init__.py").write_text(textwrap.dedent(f"""
        from .base import ModelBase
        TEXT_MODEL_MAP: dict[str, str] = {{
            "AfmoeForCausalLM": "afmoe",
            {mapped}
        }}
        MMPROJ_MODEL_MAP: dict[str, str] = {{
            {projector}
        }}
    """))
    (conversion / "base.py").write_text("import gguf\nBASE = gguf.Metadata\n")
    (conversion / "gemma.py").write_text("import gguf\nG = gguf.Metadata\n")
    (conversion / "afmoe.py").write_text("import gguf\nA = gguf.Metadata\n")
    sibling = tmp_path / "convert_hf_to_gguf.py"
    sibling.write_text("from conversion import get_model_class\nimport gguf\n")
    return sibling


def test_a_fallback_that_cannot_convert_this_architecture_is_not_taken(
    mod, tmp_path, monkeypatch, capsys,
):
    """A candidate that ranks as missing nothing certain but cannot convert this
    architecture at all must not be substituted in."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(mod, "_installed_gguf_tree", lambda *args, **kwargs: None)
    _make_gguf_py(tmp_path, symbols = ("Metadata",), version = "0.17.1")
    _make_older_sibling_converter(tmp_path, maps_the_architecture = False)
    newer = tmp_path / "unsloth_convert_hf_to_gguf.py"
    newer.write_text("import gguf\nX = gguf.SafetensorsLocal\n")

    chosen, _pin, _report = mod._resolve_converter_and_gguf(
        str(newer), sys.executable, "Gemma3ForCausalLM",
    )
    assert chosen == str(newer), "switched to a converter that does not map this architecture"
    assert "Falling back" not in capsys.readouterr().out


def test_a_fallback_that_does_map_this_architecture_is_still_taken(
    mod, tmp_path, monkeypatch, capsys,
):
    """The control: the same shape with the architecture mapped is still chosen."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    monkeypatch.setattr(mod, "_installed_gguf_tree", lambda *args, **kwargs: None)
    _make_gguf_py(tmp_path, symbols = ("Metadata",), version = "0.17.1")
    sibling = _make_older_sibling_converter(tmp_path, maps_the_architecture = True)
    newer = tmp_path / "unsloth_convert_hf_to_gguf.py"
    newer.write_text("import gguf\nX = gguf.SafetensorsLocal\n")

    chosen, _pin, _report = mod._resolve_converter_and_gguf(
        str(newer), sys.executable, "Gemma3ForCausalLM",
    )
    assert chosen == str(sibling)
    assert "Falling back" in capsys.readouterr().out


def test_the_architecture_check_only_refuses_on_evidence(mod, tmp_path):
    """Only a readable map without this architecture in it counts as a refusal."""
    monolith = tmp_path / "unsloth_convert_hf_to_gguf.py"
    monolith.write_text("import gguf\nX = gguf.Metadata\n")
    packaged = _make_older_sibling_converter(tmp_path, maps_the_architecture = False)
    assert mod._converter_maps_architecture(str(monolith), "Gemma3ForCausalLM") is True
    assert mod._converter_maps_architecture(str(monolith), None) is True
    assert mod._converter_maps_architecture(str(packaged), "AfmoeForCausalLM") is True
    assert mod._converter_maps_architecture(str(packaged), "Gemma3ForCausalLM") is False
    (tmp_path / "conversion" / "__init__.py").write_text("TEXT_MODEL_MAP = broken(\n")
    assert mod._converter_maps_architecture(str(packaged), "Gemma3ForCausalLM") is True


def test_a_fallback_that_maps_this_architecture_in_the_wrong_half_is_refused(mod, tmp_path):
    """The two maps are not interchangeable: presence in EITHER used to approve a
    fallback. Asserted on the predicate because every sibling candidate reads the
    SAME conversion package, so two diverging maps cannot be built."""
    _make_older_sibling_converter(
        tmp_path, maps_the_architecture = True, mmproj_only = True
    )
    projector_only = tmp_path / "convert_hf_to_gguf.py"
    assert mod._converter_maps_architecture(
        str(projector_only), "Gemma3ForCausalLM", {"TEXT_MODEL_MAP"}
    ) is False
    assert mod._converter_maps_architecture(
        str(projector_only), "Gemma3ForCausalLM", {"MMPROJ_MODEL_MAP"}
    ) is True
    # Both halves required, only one served: refused.
    assert mod._converter_maps_architecture(
        str(projector_only), "Gemma3ForCausalLM",
        {"TEXT_MODEL_MAP", "MMPROJ_MODEL_MAP"},
    ) is False


def test_the_required_halves_are_read_off_the_requested_converter(mod, tmp_path):
    """The halves that architecture really has, not a guess."""
    _make_older_sibling_converter(tmp_path, maps_the_architecture = True)
    packaged = tmp_path / "convert_hf_to_gguf.py"
    assert mod._converter_architecture_maps(str(packaged), "Gemma3ForCausalLM") == {
        "TEXT_MODEL_MAP"
    }
    assert mod._converter_architecture_maps(str(packaged), "NobodyForCausalLM") == set()
    # A monolith has no maps to read, which is a failure to look and not an answer.
    monolith = tmp_path / "unsloth_convert_hf_to_gguf.py"
    monolith.write_text("import gguf\nX = gguf.Metadata\n")
    assert mod._converter_architecture_maps(str(monolith), "Gemma3ForCausalLM") == set()
    # With nothing required, presence in either map still stands.
    assert mod._converter_maps_architecture(str(packaged), "Gemma3ForCausalLM") is True


# --- the conversion halves this call really runs ---

def _make_dual_mapped_tree(tmp_path):
    """An architecture named by BOTH maps, whose projector module needs a symbol
    the text module does not."""
    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "__init__.py").write_text(textwrap.dedent("""
        from .base import ModelBase
        TEXT_MODEL_MAP: dict[str, str] = {"DualForConditionalGeneration": "dualtext"}
        MMPROJ_MODEL_MAP: dict[str, str] = {"DualForConditionalGeneration": "dualproj"}
    """))
    (conversion / "base.py").write_text("import gguf\nBASE = gguf.Metadata\n")
    (conversion / "dualtext.py").write_text(
        "import gguf\n\nclass T:\n    model_arch = gguf.MODEL_ARCH.DUAL\n"
    )
    (conversion / "dualproj.py").write_text(
        "import gguf\n\nclass P:\n    projector = gguf.OnlyTheProjectorNeedsThis\n"
    )
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("from conversion import get_model_class\nimport gguf\n")
    return entry


def test_a_text_only_export_does_not_require_the_projector_module(mod, tmp_path):
    """A text-only conversion never imports the projector module, so a symbol only
    that module names is not certain."""
    entry = _make_dual_mapped_tree(tmp_path)
    arch = "DualForConditionalGeneration"

    text_certain, text_advisory = mod._converter_gguf_requirements(str(entry), arch, False)
    assert "gguf.MODEL_ARCH.DUAL" in text_certain
    assert "gguf.OnlyTheProjectorNeedsThis" not in text_certain
    assert "gguf.OnlyTheProjectorNeedsThis" in text_advisory

    # A VLM export runs both halves, so there it IS certain.
    vlm_certain, _ = mod._converter_gguf_requirements(str(entry), arch, True)
    assert "gguf.OnlyTheProjectorNeedsThis" in vlm_certain
    assert "gguf.MODEL_ARCH.DUAL" in vlm_certain

    # Not knowing keeps today's answer: both count.
    unknown_certain, _ = mod._converter_gguf_requirements(str(entry), arch)
    assert unknown_certain == vlm_certain


def test_a_text_only_fallback_is_not_rejected_for_lacking_a_projector_map(mod, tmp_path):
    """For a text-only call, the text map alone is the half a fallback must serve."""
    entry = _make_dual_mapped_tree(tmp_path)
    arch = "DualForConditionalGeneration"

    assert mod._converter_architecture_maps(str(entry), arch, False) == {"TEXT_MODEL_MAP"}
    assert mod._converter_architecture_maps(str(entry), arch, True) == {
        "TEXT_MODEL_MAP", "MMPROJ_MODEL_MAP",
    }

    fallback_dir = tmp_path / "fallback"
    (fallback_dir / "conversion").mkdir(parents=True)
    (fallback_dir / "conversion" / "__init__.py").write_text(
        'TEXT_MODEL_MAP = {"DualForConditionalGeneration": "dualtext"}\n'
        'MMPROJ_MODEL_MAP = {}\n'
    )
    fallback = fallback_dir / "unsloth_convert_hf_to_gguf.py"
    fallback.write_text("from conversion import get_model_class\nimport gguf\n")

    assert mod._converter_maps_architecture(str(fallback), arch, {"TEXT_MODEL_MAP"})
    assert not mod._converter_maps_architecture(
        str(fallback), arch, {"TEXT_MODEL_MAP", "MMPROJ_MODEL_MAP"},
    )


def test_the_conversion_is_resolved_after_the_vlm_downgrade(mod):
    """The downgrade to text-only must happen BEFORE the resolver runs."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(mod.convert_to_gguf))
    downgrade_line = resolve_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "_resolve_converter_and_gguf":
            resolve_line = node.lineno
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "not supported for MMPROJ conversion" in node.value:
            downgrade_line = node.lineno
    assert downgrade_line is not None and resolve_line is not None
    assert downgrade_line < resolve_line, "is_vlm is downgraded after the resolver ran"


# --- A module-level `finally` runs on every path, so it is not guarded ---

def test_a_module_level_finally_is_certain(mod):
    """A `finally` runs on every path, so at module level its gguf names are certain."""
    certain, advisory = mod._gguf_requirements_from_source(
        b"import gguf\ntry:\n    pass\nfinally:\n    x = gguf.InFinally\n"
    )
    assert "gguf.InFinally" in certain
    assert "gguf.InFinally" not in advisory


@pytest.mark.parametrize("label, source, name", [
    ("try body",  b"import gguf\ntry:\n    x = gguf.InTry\nexcept Exception:\n    pass\n", "gguf.InTry"),
    ("handler",   b"import gguf\ntry:\n    pass\nexcept Exception:\n    x = gguf.InHandler\n", "gguf.InHandler"),
    ("else",      b"import gguf\ntry:\n    pass\nexcept Exception:\n    pass\nelse:\n    x = gguf.InElse\n", "gguf.InElse"),
])
def test_the_other_try_parts_stay_advisory(mod, label, source, name):
    """The finally change must not have widened: none of these three is certain."""
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert name in advisory, label
    assert name not in certain, label


def test_a_finally_inside_a_function_body_is_still_advisory(mod):
    """The function body demotes first: a finally inside it never runs at import."""
    certain, advisory = mod._gguf_requirements_from_source(
        b"import gguf\ndef f():\n    try:\n        pass\n    finally:\n        x = gguf.InFnFinally\n"
    )
    assert "gguf.InFnFinally" in advisory
    assert "gguf.InFnFinally" not in certain
