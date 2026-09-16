"""The GGUF converter child's `gguf` resolution (unsloth#3581).

convert_to_gguf launches convert_hf_to_gguf.py in a child process. Every
llama.cpp converter entrypoint self-locates its own tree with
`sys.path.insert(1, Path(__file__).parent / "gguf-py")`, which sits ahead of
PYTHONPATH and site-packages, and Unsloth downloads that entrypoint from
llama.cpp master while the sibling gguf-py belongs to whatever checkout is on
disk. A newer entrypoint against an older gguf-py fails with
`module 'gguf.utility' has no attribute 'SafetensorsLocal'`; an entrypoint with
no sibling tree falls through to the unpinned site-packages gguf and fails with
`cannot import name 'MistralTokenizerType' from 'gguf.vocab'`.

These tests cover the environment the child is given, the requirement scan, the
candidate ranking and the failure diagnosis. Loads llama_cpp.py in isolation
(spec_from_file_location), matching tests/test_convert_hf_to_gguf_patcher.py.
No network, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

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
    """A gguf-py tree. `symbols` are module-level names defined in gguf/__init__.py."""
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


# ---------------------------------------------------------------------------
# _converter_child_env
# ---------------------------------------------------------------------------

def test_child_env_is_a_copy_of_the_parent_environment(mod, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CHILD_ENV_CANARY", "kept")
    env = mod._converter_child_env()
    assert env["UNSLOTH_CHILD_ENV_CANARY"] == "kept"
    # An env built from scratch would strip PATH and break the child outright.
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


# ---------------------------------------------------------------------------
# _importable_gguf_py
# ---------------------------------------------------------------------------

def test_importable_gguf_py_accepts_a_real_package(mod, tmp_path):
    gguf_py = _make_gguf_py(tmp_path)
    assert mod._importable_gguf_py(str(tmp_path)) == str(gguf_py)


def test_importable_gguf_py_rejects_a_dir_without_the_package(mod, tmp_path):
    # A gguf-py directory can exist and hold no importable gguf at all.
    _make_gguf_py(tmp_path, package=False)
    assert mod._importable_gguf_py(str(tmp_path)) is None


def test_importable_gguf_py_handles_missing_and_empty_input(mod, tmp_path):
    assert mod._importable_gguf_py(str(tmp_path / "nope")) is None
    assert mod._importable_gguf_py("") is None
    assert mod._importable_gguf_py(None) is None


# ---------------------------------------------------------------------------
# requirement scanning
# ---------------------------------------------------------------------------

def test_module_level_imports_are_certain(mod):
    source = b"from gguf.vocab import MistralTokenizerType\nimport gguf.utility\n"
    certain, advisory = mod._gguf_requirements_from_source(source)
    assert "gguf.vocab.MistralTokenizerType" in certain
    assert "gguf.utility" in certain
    assert not advisory


def test_class_body_attribute_chains_are_certain(mod):
    # conversion/gemma.py does exactly this, and it brings the converter down on
    # import, so it must count as a hard requirement.
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

    # A monolith entrypoint does not import conversion/, so a package left beside
    # it by a newer install places no requirement on it.
    monolith_entry = tmp_path / "monolith_entry.py"
    monolith_entry.write_text("import gguf\n")
    certain, _ = mod._converter_gguf_requirements(str(monolith_entry))
    assert "gguf.vocab.MistralTokenizerType" not in certain


# ---------------------------------------------------------------------------
# architecture scoping of the requirement scan
#
# conversion/__init__.py imports base.py eagerly and then, through
# get_model_class, the ONE module the architecture maps to. load_all_models
# imports the rest inside a per-module `try/except Exception` that only warns,
# and the entrypoint does not call it. So another architecture's module cannot
# break this conversion, and counting its names as certain would abandon a
# working converter for an older one.
# ---------------------------------------------------------------------------

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
    # The module this export needs, and an unrelated one that names a symbol the
    # installed gguf-py does not have.
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
    # The load-bearing assertion: another architecture's module is advisory only.
    assert "gguf.MODEL_ARCH.AFMOE" not in certain
    assert "gguf.MODEL_ARCH.AFMOE" in advisory


def test_an_mmproj_architecture_resolves_through_the_mmproj_map(mod, tmp_path):
    entry = _make_conversion_tree(tmp_path)
    certain, _ = mod._converter_gguf_requirements(
        str(entry), "Gemma3ForConditionalGeneration",
    )
    assert "gguf.MODEL_ARCH.GEMMA3" in certain


def test_an_unknown_architecture_scopes_to_the_eager_modules_only(mod, tmp_path):
    """The conservative direction: fewer certain names means fewer reasons to
    switch converters. An MLX-style config with no `architectures` lands here."""
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
    """unsloth#3581 regression, the other direction. A gguf-py one architecture
    behind on some model nobody is converting must leave the requested converter
    and the environment exactly as they are, or every export silently downgrades
    to an older converter whenever upstream adds an architecture."""
    # Neutralised so only the tree under test is in play; otherwise a complete
    # bundle on the host rescues the run and the test passes for a second reason.
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
    # A sibling entrypoint exists, so a wrongly blocked candidate 0 has somewhere
    # to fall back to; that is exactly the downgrade this test forbids.
    (tmp_path / "convert_hf_to_gguf.py").write_text("import gguf\n")

    chosen, pin, _report = mod._resolve_converter_and_gguf(
        str(entry), sys.executable, "Gemma3ForCausalLM",
    )
    assert chosen == str(entry), "switched converters over an unrelated architecture"
    # Candidate 0 is the requested pair with the environment untouched, so a
    # clean verdict there means the launch is byte for byte what it is today.
    assert pin is None


def test_a_missing_architecture_on_this_path_still_switches(
    mod, tmp_path, monkeypatch, capsys,
):
    """The complement: when the architecture being converted is the one the
    installed gguf-py cannot satisfy, falling back is correct and required."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    # The reported host's installed `gguf` wheel does not satisfy the converter
    # either, which is why the fallback is what has to happen. This runner's does,
    # so it is pinned out; the case where it satisfies has its own test below.
    monkeypatch.setattr(mod, "_installed_gguf_tree", lambda *args, **kwargs: None)
    entry = _make_conversion_tree(tmp_path)
    (tmp_path / "gguf-py").mkdir(exist_ok=True)
    pkg = tmp_path / "gguf-py" / "gguf"
    pkg.mkdir(parents=True, exist_ok=True)
    # No GEMMA3 this time, so the module this export imports cannot load.
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
    # The announcement names the symbol that stopped this export's own module,
    # not some unrelated architecture's.
    announced = capsys.readouterr().out
    assert "gguf.MODEL_ARCH.GEMMA3" in announced
    assert "gguf.MODEL_ARCH.AFMOE" not in announced


def test_the_diagnosis_names_the_symbol_that_actually_stopped_the_import(
    mod, tmp_path, monkeypatch,
):
    """The branch printed missing[:3] of a 147 name architecture list, so it named
    three irrelevant symbols. Scoped requirements name the real blocker."""
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


# ---------------------------------------------------------------------------
# the child probe
# ---------------------------------------------------------------------------

def test_probe_reports_the_tree_the_entrypoint_would_self_locate(mod, tmp_path):
    """The probe must resolve gguf the way the real run does, i.e. through the
    entrypoint's own sibling gguf-py, not through `python -c`'s empty sys.path[0]."""
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


def test_probe_reports_an_unimportable_gguf(mod, tmp_path):
    entry = tmp_path / "convert_hf_to_gguf.py"
    entry.write_text("import gguf\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "empty")
    # Isolate the interpreter from any installed gguf.
    env["PYTHONNOUSERSITE"] = "1"
    report = mod._probe_child_gguf(sys.executable, env, ("gguf.Thing",), str(entry))
    assert report is not None
    # Either gguf is absent (error set) or present from site-packages; both are
    # valid reports, what matters is that the probe always returns one.
    assert "missing" in report


def test_probe_returns_none_when_it_cannot_run(mod):
    assert mod._probe_child_gguf("/nonexistent/python", dict(os.environ), ("gguf.X",)) is None


# ---------------------------------------------------------------------------
# candidate ranking
# ---------------------------------------------------------------------------

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
    # No pin: the entrypoint self-locates its own tree exactly as before.
    assert pin is None
    assert "NO_LOCAL_GGUF" not in mod._converter_child_env(pin)
    assert report is not None and not report["missing"]


def test_resolver_falls_back_to_the_co_versioned_converter(mod, tmp_path, monkeypatch, capsys):
    """The reported case: a newer downloaded entrypoint, an older sibling gguf-py,
    and the checkout's own matching converter sitting right next to it."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    # Same as above: on the reported host nothing installed satisfies the newer
    # entrypoint, so the co-versioned converter beside it is the only way out.
    monkeypatch.setattr(mod, "_installed_gguf_tree", lambda *args, **kwargs: None)
    _make_gguf_py(tmp_path, symbols=("Metadata",), version="0.17.1")
    (tmp_path / "convert_hf_to_gguf.py").write_text("import gguf\nX = gguf.Metadata\n")
    newer = tmp_path / "unsloth_convert_hf_to_gguf.py"
    newer.write_text("import gguf\nX = gguf.SafetensorsLocal\n")

    chosen, _pin, _ = mod._resolve_converter_and_gguf(str(newer), sys.executable)
    assert chosen == str(tmp_path / "convert_hf_to_gguf.py")
    message = capsys.readouterr().out
    # The one switch that changes what produces the GGUF is announced, not silent.
    assert "Falling back" in message
    assert "gguf.SafetensorsLocal" in message


def test_resolver_pins_a_satisfying_gguf_py_from_elsewhere(mod, tmp_path, monkeypatch):
    """No sibling tree, so the entrypoint would fall through to whatever the
    ambient environment resolves. The installed bundle's tree satisfies it."""
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
    """A symbol referenced only inside a function may never be reached, so it must
    not be allowed to switch a working install onto an older converter."""
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
    """A candidate can satisfy every name that CERTAINLY runs and still fail on
    one that runs in practice: a `gguf.X` inside a function body of
    conversion/base.py is advisory by construction yet executes on every
    conversion (`gguf.LazyChunkedTensor` is exactly this). Taking the first
    candidate that cleared the certain bar picked such a tree over a genuinely
    co-versioned one, and the export then died anyway.
    """
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

    # The requested pair: missing a certain name, so candidate 0 is out.
    _make_gguf_py(tmp_path, symbols=("Metadata",))
    # A plausible tree: clears every certain name but not the advisory one.
    plausible = tmp_path / "plausible"
    _make_gguf_py(plausible, symbols=("Metadata", "OnlyNew"))
    # A co-versioned tree: clears everything.
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


# ---------------------------------------------------------------------------
# failure diagnosis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "ImportError: cannot import name 'MistralTokenizerType' from 'gguf.vocab' (/x/gguf/vocab.py)",
    "AttributeError: module 'gguf.utility' has no attribute 'SafetensorsLocal'",
    "ModuleNotFoundError: No module named 'gguf'",
])
def test_skew_signatures_are_recognised(mod, text):
    """These three name a gguf module outright, so no requirement list is needed."""
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
    """The converter is named convert_hf_to_gguf.py and logs `INFO:gguf.gguf_writer:`,
    so "gguf" appears in every failure's output. Without this the user is told to
    delete their llama.cpp folder over a bug in their model."""
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
    # Only the blocking subset is named: an advisory miss cannot be the reason,
    # so reporting it would point at the wrong symbol.
    assert mod._gguf_report_summary({"missing": ["gguf.A"]}, ()) == "reason unknown"
    assert mod._gguf_report_summary({"missing": []}, ()) == "reason unknown"


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_every_converter_launch_passes_an_explicit_env(mod):
    """Regression guard: a launch site without env= silently reintroduces the bug."""
    import inspect
    source = inspect.getsource(mod.convert_to_gguf)
    assert "_resolve_converter_and_gguf(" in source
    runs = [line for line in source.splitlines() if "subprocess.run(command" in line]
    assert runs, "no converter launch found"
    assert source.count("env=_converter_child_env(_gguf_py_pin)") == len(runs)
    # The env must be built at launch time, never frozen at preflight time, or a
    # later edit to os.environ (save.py's token boundary) would be dropped.
    assert "_converter_env =" not in source


def test_failure_path_attaches_the_diagnosis(mod):
    import inspect
    source = inspect.getsource(mod.convert_to_gguf)
    assert "_looks_like_gguf_skew(" in source
    assert "_gguf_skew_diagnosis(" in source
    # The requirement list is what keeps a model-side AttributeError from
    # collecting the skew advice, so the call must pass one.
    assert "_looks_like_gguf_skew(captured)" not in source


# ---------------------------------------------------------------------------
# end to end, skipped unless the real toolchain is staged
# ---------------------------------------------------------------------------

_TOOLCHAIN = Path("/mnt/disks/unslothai/daniel2/workspace_2/temp/r2b_llamacpp")
_MODEL = Path("/mnt/disks/unslothai/daniel2/workspace_2/temp/r2b_models/gemma-3-270m-it")


@pytest.mark.skipif(
    not (_TOOLCHAIN / "convert_hf_to_gguf.py").is_file() or not (_MODEL / "config.json").is_file(),
    reason="real llama.cpp converter toolchain or test model not staged",
)
def test_end_to_end_conversion_survives_a_stale_ambient_gguf(mod, tmp_path, monkeypatch):
    """A stale gguf on PYTHONPATH used to decide the child's import. With the
    bundle's tree pinned, the conversion completes."""
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
    # And the stale tree is still first on the parent's PYTHONPATH, so success
    # can only have come from the pin.
    assert os.environ["PYTHONPATH"] == str(stale / "gguf-py")


def test_the_preflight_returns_a_pin_not_a_frozen_environment(mod, tmp_path, monkeypatch):
    """Guard the contract the launch sites rely on: a directory (or None), so the
    environment is assembled from the live os.environ at launch."""
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))
    _make_gguf_py(tmp_path, symbols=("Metadata",))
    entry = tmp_path / "unsloth_convert_hf_to_gguf.py"
    entry.write_text("import gguf\nX = gguf.Metadata\n")
    _chosen, pin, _report = mod._resolve_converter_and_gguf(str(entry), sys.executable)
    assert pin is None or isinstance(pin, str)


# ---------------------------------------------------------------------------
# The installed gguf wheel as a candidate (verification pass)
# ---------------------------------------------------------------------------

def test_the_installed_gguf_wheel_is_used_when_the_sibling_tree_is_too_old(
    mod, tmp_path, monkeypatch, capsys,
):
    """Someone who upgraded the `gguf` wheel past their llama.cpp checkout has a
    satisfying gguf already installed, but the entrypoint's own
    `sys.path.insert(1, <sibling gguf-py>)` puts the old tree first, so it never
    gets used. The resolver must pin it and keep the newer converter rather than
    dropping back to an older one."""
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
    """The extra probe is a subprocess. An install that already works must not pay
    for it."""
    called = {"hit": False}
    def _trap(*args, **kwargs):
        called["hit"] = True
        return None
    monkeypatch.setattr(mod, "_installed_gguf_tree", _trap)
    monkeypatch.setattr(mod, "LLAMA_CPP_DEFAULT_DIR", str(tmp_path / "absent"))

    # The converter gets its own sibling gguf-py carrying the symbol it needs, which
    # is the tree every llama.cpp entrypoint puts on sys.path ahead of anything else.
    # Without it this test asserts on whatever `gguf` the developer's venv happens to
    # ship: in an environment with no `gguf` wheel the requested pair does NOT work,
    # the resolver legitimately reaches for the installed tree, and the assertion
    # below fails for a reason that has nothing to do with the behaviour under test.
    _make_gguf_py(tmp_path, symbols=("GGUFWriter",))

    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("import gguf\nX = gguf.GGUFWriter\n")
    chosen, _pin, _report = mod._resolve_converter_and_gguf(str(converter), sys.executable)

    assert chosen == str(converter)
    assert called["hit"] is False


def test_installed_gguf_tree_reads_the_child_not_the_parent(mod, monkeypatch, tmp_path):
    """The tree is whatever the CHILD resolves with the sibling tree suppressed, and
    it is only accepted when a real package sits inside it."""
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
