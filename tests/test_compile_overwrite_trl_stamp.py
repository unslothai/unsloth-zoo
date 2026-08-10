"""UNSLOTH_COMPILE_OVERWRITE=0 must still recompile when TRL changes.

The version stamp `create_new_function` writes into every generated file is
[unsloth_zoo, unsloth, transformers, trl], but the escape hatch only ever
compared index 2. TRL matters just as much: the generated RL trainers mirror
the installed TRL config signature and import symbols straight out of
`trl.trainer.<x>_trainer`. A cache built against TRL 0.25 and reused on TRL 1.9
fails to import (`cannot import name 'AutoConfig' from
'trl.trainer.grpo_trainer'`), Unsloth quietly falls back to TRL's own untouched
trainer, and the user is back to

    TypeError: GRPOConfig.__init__() got an unexpected keyword argument
    'max_prompt_length'

with none of Unsloth's patching applied.

The second test is the counterweight: the hatch has to keep working when
nothing moved, or hand-edited caches stop being editable.
"""

import os

import pytest

from unsloth_zoo import compiler

PROBE_NAME = "UnslothCompileStampProbe"
PROBE_SOURCE = "def _unsloth_stamp_probe():\n    return 1\n"


def _emit(tmp_path, monkeypatch, overwrite):
    """Run create_new_function against an isolated cache dir, return the path."""
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    compiler.create_new_function(
        PROBE_NAME,
        PROBE_SOURCE,
        "torch",
        [],
        overwrite=overwrite,
    )
    return tmp_path / f"{PROBE_NAME}.py"


def _stamp_lines(path):
    header = path.read_text()
    header = header[: header.find("__UNSLOTH_VERSIONING__")]
    return [line.strip() for line in header.strip().strip('"').split("\n") if line.strip()]


def _rewrite_trl_stamp(path, version):
    """Doctor only the trl line, leaving transformers and the body alone."""
    text = path.read_text()
    lines = _stamp_lines(path)
    assert len(lines) > 3, lines
    old = lines[3]
    head, sep, tail = text.partition("__UNSLOTH_VERSIONING__")
    assert sep, "no version stamp written"
    return path.write_text(head.replace(f"\n{old}\n", f"\n{version}\n", 1) + sep + tail)


def test_a_changed_trl_stamp_forces_a_recompile(tmp_path, monkeypatch):
    path = _emit(tmp_path, monkeypatch, overwrite=True)
    installed_trl = _stamp_lines(path)[3]
    _rewrite_trl_stamp(path, "0.0.1-stale")
    assert _stamp_lines(path)[3] == "0.0.1-stale"

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch, overwrite=False)

    # Only TRL moved, so before the fix this file was left stale.
    assert _stamp_lines(path)[3] == installed_trl


def test_an_unchanged_stamp_still_leaves_a_hand_edited_cache_alone(tmp_path, monkeypatch):
    """The whole point of the hatch: iterate on a generated file by editing it."""
    path = _emit(tmp_path, monkeypatch, overwrite=True)
    marker = "# hand edited, do not clobber\n"
    path.write_text(path.read_text() + marker)

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch, overwrite=False)

    assert marker in path.read_text()


def test_the_warning_names_the_library_that_moved(tmp_path, monkeypatch, caplog):
    """Silently recompiling would leave the user guessing why their edit went."""
    path = _emit(tmp_path, monkeypatch, overwrite=True)
    stale_trl = _stamp_lines(path)[3]
    _rewrite_trl_stamp(path, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    # warning_once dedupes process wide, so clear whatever it remembers.
    getattr(compiler.logger.warning_once, "cache_clear", lambda: None)()
    with caplog.at_level("WARNING", logger="unsloth_zoo.log"):
        _emit(tmp_path, monkeypatch, overwrite=False)

    assert f"trl 0.0.1-stale -> {stale_trl}" in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
