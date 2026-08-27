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

The counterweights are that the hatch has to keep working when nothing moved,
or hand-edited caches stop being editable, and that a TRL bump must not reach
caches TRL had no hand in: the combined model modules and the peft/torch
forward patches are generated from transformers, peft and torch source.
"""

import pytest

from unsloth_zoo import compiler

PROBE_NAME = "UnslothCompileStampProbe"
PROBE_SOURCE = "def _unsloth_stamp_probe():\n    return 1\n"

# What unsloth/models/rl.py passes for a generated RL trainer, and what the
# zoo's own peft / combined-module call sites pass.
TRL_LOCATION = "trl.trainer.sft_trainer"
NON_TRL_LOCATION = "transformers.models.qwen2.modeling_qwen2"

_OMIT = object()


def _emit(tmp_path, monkeypatch, overwrite=_OMIT, model_location=TRL_LOCATION, source=PROBE_SOURCE):
    """Run create_new_function against an isolated cache dir, return the path.

    Leaving `overwrite` out exercises the default-argument call sites, which
    read the cache down a different path than the explicit `overwrite=False`
    ones and so have to be covered separately.
    """
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_LOCATION", str(tmp_path))
    monkeypatch.setattr(compiler, "UNSLOTH_COMPILE_USE_TEMP", False)
    kwargs = {} if overwrite is _OMIT else {"overwrite": overwrite}
    compiler.create_new_function(
        PROBE_NAME,
        source,
        model_location,
        [],
        **kwargs,
    )
    return tmp_path / f"{PROBE_NAME}.py"


def _stamp_lines(path):
    header = path.read_text()
    header = header[: header.find("__UNSLOTH_VERSIONING__")]
    return [line.strip() for line in header.strip().strip('"').split("\n") if line.strip()]


def _rewrite_stamp(path, index, version):
    """Doctor one stamp line, leaving the other entries and the body alone."""
    text = path.read_text()
    lines = _stamp_lines(path)
    assert len(lines) > index, lines
    old = lines[index]
    head, sep, tail = text.partition("__UNSLOTH_VERSIONING__")
    assert sep, "no version stamp written"
    return path.write_text(head.replace(f"\n{old}\n", f"\n{version}\n", 1) + sep + tail)


def _rewrite_trl_stamp(path, version):
    return _rewrite_stamp(path, 3, version)


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


def test_a_default_overwrite_caller_is_invalidated_too(tmp_path, monkeypatch):
    """The zoo's own peft and combined-module caches omit the argument.

    Those callers never populated `file_source`, so the stamp comparison was
    unreachable for them and the hatch kept the cache whatever had moved.
    """
    path = _emit(tmp_path, monkeypatch, overwrite=True)
    installed_trl = _stamp_lines(path)[3]
    _rewrite_trl_stamp(path, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch)

    assert _stamp_lines(path)[3] == installed_trl


def test_the_transformers_half_reaches_default_callers_too(tmp_path, monkeypatch):
    """Same gap, older half of the check: a transformers bump was ignored."""
    path = _emit(tmp_path, monkeypatch, overwrite=True)
    installed_tf = _stamp_lines(path)[2]
    _rewrite_stamp(path, 2, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch)

    assert _stamp_lines(path)[2] == installed_tf


def test_an_unchanged_stamp_leaves_a_default_callers_cache_alone(tmp_path, monkeypatch):
    """Reading the cache must not turn the hatch into an unconditional rebuild."""
    path = _emit(tmp_path, monkeypatch, overwrite=True)
    marker = "# hand edited, do not clobber\n"
    path.write_text(path.read_text() + marker)

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch)
    _emit(tmp_path, monkeypatch)

    assert marker in path.read_text()


def test_a_stamp_that_predates_the_trl_entry_is_replaced(tmp_path, monkeypatch):
    """A short stamp reads as trl "0", which differs from any installed trl."""
    path = _emit(tmp_path, monkeypatch, overwrite=True)
    installed_trl = _stamp_lines(path)[3]
    if installed_trl == "0":
        pytest.skip("trl is not installed, so a short stamp is not stale")
    text = path.read_text()
    head, sep, tail = text.partition("__UNSLOTH_VERSIONING__")
    path.write_text(head.replace(f"\n{installed_trl}\n", "\n", 1) + sep + tail)
    assert len(_stamp_lines(path)) == 3

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch)

    assert _stamp_lines(path)[3] == installed_trl


def test_a_first_run_with_no_cache_still_writes_the_file(tmp_path, monkeypatch):
    """Nothing to compare against must not mean nothing gets written."""
    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    path = _emit(tmp_path, monkeypatch)

    assert path.is_file()
    assert len(_stamp_lines(path)) == 4


def test_a_trl_bump_leaves_a_non_trl_cache_alone(tmp_path, monkeypatch):
    """A combined model module cannot go stale because TRL moved.

    It is generated from transformers source and never imports trl, so
    regenerating it here would destroy the hand edit for nothing.
    """
    path = _emit(tmp_path, monkeypatch, overwrite=True, model_location=NON_TRL_LOCATION)
    marker = "# hand edited, do not clobber\n"
    path.write_text(path.read_text() + marker)
    _rewrite_trl_stamp(path, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch, model_location=NON_TRL_LOCATION)

    assert marker in path.read_text()
    assert _stamp_lines(path)[3] == "0.0.1-stale"


def test_a_transformers_bump_still_reaches_a_non_trl_cache(tmp_path, monkeypatch):
    """The narrowing is for the TRL half only; transformers stays unconditional."""
    path = _emit(tmp_path, monkeypatch, overwrite=True, model_location=NON_TRL_LOCATION)
    installed_tf = _stamp_lines(path)[2]
    _rewrite_stamp(path, 2, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch, model_location=NON_TRL_LOCATION)

    assert _stamp_lines(path)[2] == installed_tf


def test_a_trl_body_is_invalidated_even_from_a_non_trl_location(tmp_path, monkeypatch):
    """Backstop for a model_location that ever resolves outside trl.

    Missing an invalidation is the expensive direction: the stale trainer fails
    to import and Unsloth silently falls back to TRL's own untouched trainer.
    Every generated trainer body carries the marker even if the path does not.
    """
    trl_body = 'def _unsloth_stamp_probe():\n    _tag_names = ["trl", "sft"]\n    return 1\n'
    path = _emit(
        tmp_path, monkeypatch, overwrite=True,
        model_location="somewhere.else.weird_trainer", source=trl_body,
    )
    installed_trl = _stamp_lines(path)[3]
    _rewrite_trl_stamp(path, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(
        tmp_path, monkeypatch,
        model_location="somewhere.else.weird_trainer", source=trl_body,
    )

    assert _stamp_lines(path)[3] == installed_trl


def test_a_relocated_experimental_trainer_is_still_invalidated(tmp_path, monkeypatch):
    """TRL moves trainers to trl.experimental, and rl.py follows them there.

    On trl 0.25.1 BCO already resolves to trl.experimental.bco.bco_trainer, so
    the dependency signal keys on the top-level package rather than a
    `trl.trainer.` prefix, which would skip exactly the trainers that moved.
    """
    location = "trl.experimental.bco.bco_trainer"
    path = _emit(tmp_path, monkeypatch, overwrite=True, model_location=location)
    installed_trl = _stamp_lines(path)[3]
    _rewrite_trl_stamp(path, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch, overwrite=False, model_location=location)

    assert _stamp_lines(path)[3] == installed_trl


def test_the_trl_marker_is_matched_as_a_whole_word(tmp_path, monkeypatch):
    """`ctrl` in a body must not make a non-TRL cache look TRL-dependent."""
    ctrl_body = 'def _unsloth_stamp_probe():\n    ctrl = 1  # ctrl, not trl\n    return ctrl\n'
    path = _emit(
        tmp_path, monkeypatch, overwrite=True,
        model_location=NON_TRL_LOCATION, source=ctrl_body,
    )
    # The body comment above deliberately contains a real whole-word `trl`;
    # strip it so only `ctrl` remains, which must not trip the backstop.
    path.write_text(path.read_text().replace("# ctrl, not trl", "# ctrl only"))
    _rewrite_trl_stamp(path, "0.0.1-stale")

    monkeypatch.setenv("UNSLOTH_COMPILE_OVERWRITE", "0")
    _emit(tmp_path, monkeypatch, model_location=NON_TRL_LOCATION, source=ctrl_body)

    assert _stamp_lines(path)[3] == "0.0.1-stale"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
