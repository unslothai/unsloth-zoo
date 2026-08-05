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

"""An OOM-killed GGUF conversion can usually just be retried onto disk.

The converter holds tensors in host RAM. A large model on a free Colab or
Kaggle VM is taken by the kernel OOM-killer, and subprocess reports only

    Command '[...]' died with <Signals.SIGKILL: 9>

`Gemma3N_(4B)-Audio` dies exactly here, measured: it trains, infers and merges
cleanly (15.7GB written, 155GB disk free, freed before the conversion) and the
converter is then killed, on the high-RAM T4 as well as the plain one.

llama.cpp already has the answer. `--use-temp-file` is documented as "helpful
when running out of memory, process killed". The catch is that it refuses to
run alongside splitting:

    Error: Cannot use temp file when splitting

and `--split-max-size` is always passed, so the flag has to arrive with the
split options removed or the retry swaps one hard failure for another.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth_zoo.llama_cpp import (  # noqa: E402
    _converter_was_oom_killed,
    _gguf_output_paths,
    _remove_gguf_outputs,
    _retry_with_temp_file,
)


# ---- recognising the kill --------------------------------------------------

def test_sigkill_from_the_message():
    assert _converter_was_oom_killed(
        RuntimeError("Command '[...]' died with <Signals.SIGKILL: 9>."))


@pytest.mark.parametrize("code", [-9, 137])
def test_sigkill_from_the_returncode(code):
    class _Called(Exception):
        returncode = code

    assert _converter_was_oom_killed(_Called())


def test_an_ordinary_failure_is_not_a_kill():
    """A converter that fails for its own reasons must not be run twice."""
    assert not _converter_was_oom_killed(
        RuntimeError("NotImplementedError: Unknown tensor audio_tower.x"))


def test_a_nonzero_exit_is_not_a_kill():
    class _Called(Exception):
        returncode = 1

    assert not _converter_was_oom_killed(_Called())


# ---- building the retry ----------------------------------------------------

BASE = ["/usr/bin/python3", "/root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py",
        "--outfile", "model.F16.gguf", "--outtype", "f16",
        "--split-max-size", "50G", "/tmp/model_dir"]


def test_the_split_option_is_removed():
    """This is the whole point. llama.cpp exits 1 on the combination, so a
    retry that kept --split-max-size would fail differently and look like a
    new bug."""
    out = _retry_with_temp_file(BASE)
    assert "--split-max-size" not in out
    assert "50G" not in out


def test_split_max_tensors_is_removed_too():
    cmd = BASE[:-3] + ["--split-max-tensors", "128", "/tmp/model_dir"]
    out = _retry_with_temp_file(cmd)
    assert "--split-max-tensors" not in out and "128" not in out


def test_a_valueless_split_flag_does_not_eat_the_next_option():
    """Defensive: skipping "the token after" would drop an unrelated flag if a
    split option ever arrives without a value."""
    cmd = ["/usr/bin/python3", "conv.py", "--split-max-size", "--outtype", "f16",
           "/tmp/model_dir"]
    out = _retry_with_temp_file(cmd)
    assert "--outtype" in out and "f16" in out
    assert "--split-max-size" not in out


def test_the_flag_is_added():
    assert "--use-temp-file" in _retry_with_temp_file(BASE)


def test_the_model_path_stays_last():
    """It is positional; anything after it is parsed as another positional."""
    assert _retry_with_temp_file(BASE)[-1] == "/tmp/model_dir"


def test_everything_else_survives():
    out = _retry_with_temp_file(BASE)
    for token in ("--outfile", "model.F16.gguf", "--outtype", "f16"):
        assert token in out


def test_it_refuses_to_retry_twice():
    """Without this the loop would re-issue the same command forever on a
    machine that is simply too small."""
    once = _retry_with_temp_file(BASE)
    assert _retry_with_temp_file(once) is None


def test_the_result_is_a_new_list():
    out = _retry_with_temp_file(BASE)
    assert out is not BASE
    assert "--split-max-size" in BASE, "the caller's command must not be mutated"


# ---- how the run loop uses it ----------------------------------------------

def _loop_src():
    src = (ROOT / "unsloth_zoo" / "llama_cpp.py").read_text(encoding="utf-8")
    i = src.index("attempted_temp_file = False")
    return src[i:src.index("if not required:", i)]


def test_the_retry_is_gated_on_a_kill():
    assert "_converter_was_oom_killed(e)" in _loop_src()


def test_the_retry_happens_at_most_once():
    body = _loop_src()
    assert "not attempted_temp_file" in body
    assert "attempted_temp_file = True" in body


def test_it_says_what_it_is_doing():
    """A silent retry that then succeeds hides a real capacity problem, and one
    that then fails looks like a single inexplicable failure."""
    body = _loop_src()
    assert "--use-temp-file" in body and "host RAM" in body


def test_the_dependency_repair_retry_is_independent():
    """Two retries share one loop; one must not consume the other's chance."""
    body = _loop_src()
    assert body.index("attempted_repair") < body.index("attempted_temp_file = True")


# ---- clearing what the killed run left behind ------------------------------

def _touch(directory, name, data = b"GGUF"):
    path = directory / name
    path.write_bytes(data)
    return path


def test_the_shards_of_a_killed_split_run_are_removed(tmp_path):
    """GGUFWriter.open_output_file opens every shard with "wb" before a single
    tensor byte, so a SIGKILL always leaves them. The retry then drops
    --split-max-size and writes the unsharded name instead, and callers upload
    every save_directory/*.gguf, so the stubs would ship beside the real file."""
    out = str(tmp_path / "model.BF16.gguf")
    shards = [_touch(tmp_path, f"model.BF16-{i:05d}-of-00003.gguf") for i in (1, 2, 3)]
    _remove_gguf_outputs(out)
    assert not any(s.exists() for s in shards)


def test_a_partial_output_file_is_removed(tmp_path):
    out = _touch(tmp_path, "model.BF16-mmproj.gguf")
    _remove_gguf_outputs(str(out))
    assert not out.exists()


def test_unrelated_ggufs_are_left_alone(tmp_path):
    """Only this run's own name and its shards, never a sibling export."""
    keep = [_touch(tmp_path, "model.BF16.gguf"),
            _touch(tmp_path, "other.BF16-00001-of-00002.gguf"),
            _touch(tmp_path, "model.Q4_K_M.gguf")]
    _remove_gguf_outputs(str(tmp_path / "model.BF16-mmproj.gguf"))
    assert all(k.exists() for k in keep)


def test_a_neighbour_whose_name_ends_with_ours_is_left_alone(tmp_path):
    """The shard pattern is matched against the whole filename.

    These paths are os.remove'd, so an unanchored match would take somebody
    else's export: "old-model.BF16-00001-of-00002" ends with the shard name
    that cleaning "model.BF16" looks for.
    """
    keep = [_touch(tmp_path, "old-model.BF16-00001-of-00002.gguf"),
            _touch(tmp_path, "my-model.BF16-00002-of-00002.gguf")]
    mine = _touch(tmp_path, "model.BF16-00001-of-00002.gguf")
    _remove_gguf_outputs(str(tmp_path / "model.BF16.gguf"))
    assert all(k.exists() for k in keep)
    assert not mine.exists()


def test_removing_a_missing_output_is_not_an_error(tmp_path):
    _remove_gguf_outputs(str(tmp_path / "nothing" / "model.gguf"))


def test_the_paths_include_the_file_and_its_shards(tmp_path):
    _touch(tmp_path, "model.BF16-00001-of-00002.gguf")
    _touch(tmp_path, "model.BF16-00002-of-00002.gguf")
    got = [Path(p).name for p in _gguf_output_paths(str(tmp_path / "model.BF16.gguf"))]
    assert got == ["model.BF16.gguf",
                   "model.BF16-00001-of-00002.gguf",
                   "model.BF16-00002-of-00002.gguf"]


def test_the_retry_clears_the_old_output_first():
    body = _loop_src()
    assert body.index("_remove_gguf_outputs(output_file)") < body.index("command = retry")


def test_a_failed_projector_is_removed_and_stops_claiming_vlm():
    """The converter truncates its --outfile at header time, so "a failed
    optional run wrote no file" was never true: the partial projector, or a good
    one from an earlier export, sits there and gets uploaded as valid."""
    src = (ROOT / "unsloth_zoo" / "llama_cpp.py").read_text(encoding="utf-8")
    i = src.index("if not required:")
    body = src[i:src.index("if optional_failed:", i)]
    assert "_remove_gguf_outputs(output_file)" in body
    assert "is_vlm = False" in body


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
