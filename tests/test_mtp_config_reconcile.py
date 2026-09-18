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

"""An exported config must not declare an MTP head the weights do not contain.

Qwen3.5 ships its multi-token prediction head as top-level `mtp.*` tensors and
declares it with `mtp_num_hidden_layers`. transformers has no MTP module for
the architecture and lists `^mtp.*` in `_keys_to_ignore_on_load_unexpected`, so
the head is dropped the moment the model loads and no re-save can put it back.
An export that keeps the declaration therefore promises weights that are not in
the file, and every consumer that trusts the config goes looking for them
(unsloth#7681).
"""

import json
import os
import stat

import numpy as np
import pytest
# numpy rather than torch: nothing here needs a tensor library, only real
# safetensors bytes on disk, and this way the file runs on a CPU only runner
# that ships no torch wheel.
from safetensors.numpy import save_file

from unsloth_zoo.saving_utils import (
    MTP_CONFIG_KEY,
    _checkpoint_tensor_names,
    is_mtp_tensor_name,
    mtp_head_is_present,
    reconcile_mtp_config,
)


def checkpoint_mtp_tensor_names(folder):
    """The MTP names in a saved checkpoint, or None when the weights cannot be
    read. `reconcile_mtp_config` reads the full name list once and filters it
    itself, so this composition lives here rather than in the module, where it
    would be an export nothing calls."""
    names = _checkpoint_tensor_names(folder)
    if names is None:
        return None
    return sorted(name for name in names if is_mtp_tensor_name(name))

# The 15 names Qwen/Qwen3.5-0.8B actually ships, trimmed to the distinct shapes.
QWEN35_MTP_NAMES = (
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
)
BODY_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
)


def _write_checkpoint(folder, names, sharded = False):
    """A minimal but real safetensors checkpoint, index and all."""
    tensors = {name: np.zeros((2, 2), dtype = np.float32) for name in names}
    if not sharded:
        save_file(tensors, str(folder / "model.safetensors"))
        return
    items = list(tensors.items())
    half = max(1, len(items) // 2)
    shards = {
        "model-00001-of-00002.safetensors": dict(items[:half]),
        "model-00002-of-00002.safetensors": dict(items[half:]),
    }
    weight_map = {}
    for shard_name, shard in shards.items():
        save_file(shard, str(folder / shard_name))
        weight_map.update({key: shard_name for key in shard})
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding = "utf-8",
    )


def _write_config(folder, nested = True, declared = 1):
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
    if nested:
        config["text_config"] = {"num_hidden_layers": 24}
        if declared is not None:
            config["text_config"][MTP_CONFIG_KEY] = declared
    elif declared is not None:
        config[MTP_CONFIG_KEY] = declared
    (folder / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    return config


def _saved(folder):
    return json.loads((folder / "config.json").read_text(encoding = "utf-8"))


# ---- the tensor-name predicate -------------------------------------------


@pytest.mark.parametrize("name", [
    "mtp.fc.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "model.mtp.norm.weight",
    "language_model.mtp.fc.weight",
    "model.language_model.mtp.fc.weight",
    # The older Qwen2-VL prefix ordering. Unsloth's own remap rewrites it to
    # the line above before anything is written, but a name read back off a
    # checkpoint someone else produced has not been through that remap, so the
    # prefix group is order free rather than a fixed `model.language_model.`.
    "language_model.model.mtp.fc.weight",
])
def test_mtp_tensor_names_are_recognised(name):
    assert is_mtp_tensor_name(name)


@pytest.mark.parametrize("name", [
    # A body tensor, the visual tower, and two near misses that must not match:
    # a block whose name merely starts with the same letters, and a body layer
    # that happens to contain the substring.
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "mtptower.fc.weight",
    "model.language_model.layers.0.mtp_like.weight",
])
def test_non_mtp_tensor_names_are_not_recognised(name):
    assert not is_mtp_tensor_name(name)


# ---- reading the checkpoint ----------------------------------------------


def test_checkpoint_mtp_names_from_a_single_file(tmp_path):
    _write_checkpoint(tmp_path, BODY_NAMES + QWEN35_MTP_NAMES)
    assert checkpoint_mtp_tensor_names(tmp_path) == sorted(QWEN35_MTP_NAMES)


def test_checkpoint_mtp_names_from_a_shard_index(tmp_path):
    _write_checkpoint(tmp_path, BODY_NAMES + QWEN35_MTP_NAMES, sharded = True)
    assert checkpoint_mtp_tensor_names(tmp_path) == sorted(QWEN35_MTP_NAMES)


def test_checkpoint_with_no_mtp_reports_an_empty_list(tmp_path):
    _write_checkpoint(tmp_path, BODY_NAMES)
    assert checkpoint_mtp_tensor_names(tmp_path) == []


def test_unreadable_checkpoint_is_unknown_not_empty(tmp_path):
    """The distinction the repair turns on: no weights to read is not a missing
    MTP head, and must never license editing the config."""
    assert checkpoint_mtp_tensor_names(tmp_path) is None
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file")
    assert checkpoint_mtp_tensor_names(tmp_path) is None


def test_corrupt_shard_index_is_unknown(tmp_path):
    _write_checkpoint(tmp_path, BODY_NAMES, sharded = True)
    (tmp_path / "model.safetensors.index.json").write_text("{", encoding = "utf-8")
    assert checkpoint_mtp_tensor_names(tmp_path) is None


# ---- the repair ----------------------------------------------------------


@pytest.mark.parametrize("nested", [True, False])
def test_declaration_is_stripped_when_the_weights_have_no_mtp(tmp_path, nested):
    """The reproduced defect: a Qwen3.5 export keeps `mtp_num_hidden_layers`
    while the merged weights carry no `mtp.*` tensor at all."""
    _write_config(tmp_path, nested = nested)
    _write_checkpoint(tmp_path, BODY_NAMES)

    assert reconcile_mtp_config(tmp_path) == "stripped"

    saved = _saved(tmp_path)
    holder = saved["text_config"] if nested else saved
    assert MTP_CONFIG_KEY not in holder
    # Nothing else may be disturbed.
    assert saved["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    if nested:
        assert saved["text_config"]["num_hidden_layers"] == 24


def test_declaration_is_kept_when_the_weights_do_have_mtp(tmp_path):
    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES + QWEN35_MTP_NAMES)

    assert reconcile_mtp_config(tmp_path) == "agrees"
    assert _saved(tmp_path)["text_config"][MTP_CONFIG_KEY] == 1


def test_repair_is_idempotent(tmp_path):
    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    assert reconcile_mtp_config(tmp_path) == "stripped"
    assert reconcile_mtp_config(tmp_path) == "not-declared"
    assert MTP_CONFIG_KEY not in _saved(tmp_path)["text_config"]


def test_unreadable_weights_leave_the_config_alone(tmp_path):
    """Fail closed: an export we cannot inspect keeps whatever it declared."""
    _write_config(tmp_path)
    assert reconcile_mtp_config(tmp_path) == "unknown"
    assert _saved(tmp_path)["text_config"][MTP_CONFIG_KEY] == 1


def test_a_config_that_never_declared_mtp_is_untouched(tmp_path):
    before = _write_config(tmp_path, declared = None)
    _write_checkpoint(tmp_path, BODY_NAMES)
    assert reconcile_mtp_config(tmp_path) == "not-declared"
    assert _saved(tmp_path) == before


def test_missing_config_is_reported_not_raised(tmp_path):
    assert reconcile_mtp_config(tmp_path) == "no-config"


def test_repair_never_raises_on_a_broken_config(tmp_path):
    (tmp_path / "config.json").write_text("{not json", encoding = "utf-8")
    _write_checkpoint(tmp_path, BODY_NAMES)
    assert reconcile_mtp_config(tmp_path) == "unknown"


def test_caller_supplied_tensor_names_are_honoured(tmp_path):
    """A push_to_hub export has no local folder; the writer passes the names."""
    _write_config(tmp_path)
    assert reconcile_mtp_config(tmp_path, tensor_names = BODY_NAMES) == "stripped"
    assert MTP_CONFIG_KEY not in _saved(tmp_path)["text_config"]


def test_a_head_stored_as_extra_layers_is_not_stripped(tmp_path):
    """The other MTP spelling: DeepSeek-V3 / GLM style heads live in `layers.N`
    blocks past `num_hidden_layers` rather than under `mtp.`. A checkpoint that
    declares MTP and stores it that way still has a head, so the declaration
    must survive even though no `mtp.*` name is present."""
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {"num_hidden_layers": 2, MTP_CONFIG_KEY: 1},
    }), encoding = "utf-8")
    _write_checkpoint(tmp_path, (
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.1.mlp.up_proj.weight",
        "model.language_model.layers.2.mlp.up_proj.weight",  # the MTP block
    ))
    assert reconcile_mtp_config(tmp_path) == "agrees"
    assert _saved(tmp_path)["text_config"][MTP_CONFIG_KEY] == 1


def test_a_checkpoint_within_its_layer_count_is_still_stripped(tmp_path):
    """The guard above must not swallow the real case."""
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {"num_hidden_layers": 3, MTP_CONFIG_KEY: 1},
    }), encoding = "utf-8")
    _write_checkpoint(tmp_path, (
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.1.mlp.up_proj.weight",
        "model.language_model.layers.2.mlp.up_proj.weight",
    ))
    assert reconcile_mtp_config(tmp_path) == "stripped"
    assert MTP_CONFIG_KEY not in _saved(tmp_path)["text_config"]


def test_the_layer_count_is_found_outside_the_declaring_container(tmp_path):
    """A multimodal config may declare the key at the top level while keeping
    `num_hidden_layers` in `text_config`. Resolving the count only out of the
    declaring container yields None there, which silently disables the
    extra-layers check and strips a declaration the weights do back."""
    (tmp_path / "config.json").write_text(json.dumps({
        MTP_CONFIG_KEY: 1,
        "text_config": {"num_hidden_layers": 2},
    }), encoding = "utf-8")
    _write_checkpoint(tmp_path, (
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.1.mlp.up_proj.weight",
        "model.language_model.layers.2.mlp.up_proj.weight",  # the MTP block
    ))
    assert reconcile_mtp_config(tmp_path) == "agrees"
    assert _saved(tmp_path)[MTP_CONFIG_KEY] == 1


def test_the_old_prefix_ordering_is_read_as_a_head(tmp_path):
    """`language_model.model.mtp.*`, the pre-Qwen3.5 ordering, is a head."""
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {"num_hidden_layers": 2, MTP_CONFIG_KEY: 1},
    }), encoding = "utf-8")
    _write_checkpoint(tmp_path, BODY_NAMES + ("language_model.model.mtp.fc.weight",))
    assert reconcile_mtp_config(tmp_path) == "agrees"
    assert _saved(tmp_path)["text_config"][MTP_CONFIG_KEY] == 1


# ---- the shared rule both writers use ------------------------------------


def test_shared_rule_sees_the_mtp_spelling():
    assert mtp_head_is_present(BODY_NAMES + ("mtp.fc.weight",)) is True


def test_shared_rule_sees_the_extra_layers_spelling():
    names = ("model.layers.0.mlp.up_proj.weight", "model.layers.2.mlp.up_proj.weight")
    assert mtp_head_is_present(names, {"num_hidden_layers": 2}) is True
    assert mtp_head_is_present(names, {"num_hidden_layers": 3}) is False


def test_shared_rule_without_a_layer_count_reports_only_the_mtp_spelling():
    """The extra-layers form cannot be decided without a count, so a caller
    that has no config gets the conservative answer rather than a guess."""
    names = ("model.layers.9.mlp.up_proj.weight",)
    assert mtp_head_is_present(names) is False
    assert mtp_head_is_present(names + ("mtp.fc.weight",)) is True


def test_shared_rule_accepts_a_live_config_object():
    """`unsloth/save.py`'s around-the-write guard passes a PretrainedConfig, not
    a dict, so attribute access has to work as well as `.get`."""
    import types

    config = types.SimpleNamespace(text_config = types.SimpleNamespace(num_hidden_layers = 2))
    names = ("model.language_model.layers.2.mlp.up_proj.weight",)
    assert mtp_head_is_present(names, config) is True
    assert mtp_head_is_present(names, config, config.text_config) is True


def test_shared_rule_tolerates_no_names():
    assert mtp_head_is_present(None) is False
    assert mtp_head_is_present(()) is False


def test_shared_rule_accepts_a_bare_layer_count():
    """A caller that already has the integer must not silently get False."""
    names = ("model.layers.2.mlp.up_proj.weight",)
    assert mtp_head_is_present(names, 2) is True
    assert mtp_head_is_present(names, 3) is False


def test_shared_rule_does_not_consume_a_generator_per_holder():
    """`unsloth/save.py` evaluates this once per config holder, so a generator
    handed in must not be exhausted by the first call."""
    names = ("model.layers.0.mlp.up_proj.weight", "mtp.fc.weight")
    generator = (name for name in names)
    assert mtp_head_is_present(generator) is True


# ---- against the real published config, and an unwritable folder -----------


@pytest.fixture
def real_qwen35_config():
    """`config.json` as Qwen publishes it, so the shape this repair assumes is
    checked against the model it was written for rather than against our own
    fixture. Skipped cleanly when the Hub is unreachable."""
    requests = pytest.importorskip("requests")
    url = "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/main/config.json"
    try:
        response = requests.get(url, timeout = 30)
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"hub unreachable: {exc}")
    if response.status_code in (401, 403, 429, 503):
        pytest.skip(f"hub unavailable: HTTP {response.status_code}")
    if response.status_code != 200:
        pytest.skip(f"config not published at that address: HTTP {response.status_code}")
    try:
        config = response.json()
    except ValueError as exc:
        pytest.skip(f"config is not json: {exc}")
    if MTP_CONFIG_KEY not in json.dumps(config):
        pytest.skip("this release no longer declares the MTP layer count")
    return config


def test_the_real_published_config_is_repaired_and_only_there(tmp_path, real_qwen35_config):
    """Qwen keeps the declaration in `text_config`. A merged export of that model
    carries no `mtp.*` tensor, so the key must go, and nothing else in a config
    this size may move."""
    (tmp_path / "config.json").write_text(
        json.dumps(real_qwen35_config, indent = 2), encoding = "utf-8",
    )
    _write_checkpoint(tmp_path, BODY_NAMES)

    assert reconcile_mtp_config(tmp_path) == "stripped"

    saved = _saved(tmp_path)
    expected = json.loads(json.dumps(real_qwen35_config))
    for container in (expected, expected.get("text_config")):
        if isinstance(container, dict):
            container.pop(MTP_CONFIG_KEY, None)
    assert saved == expected


def test_the_real_published_config_is_kept_when_the_head_is_there(tmp_path, real_qwen35_config):
    (tmp_path / "config.json").write_text(
        json.dumps(real_qwen35_config, indent = 2), encoding = "utf-8",
    )
    _write_checkpoint(tmp_path, BODY_NAMES + QWEN35_MTP_NAMES)

    before = (tmp_path / "config.json").read_bytes()
    assert reconcile_mtp_config(tmp_path) == "agrees"
    assert (tmp_path / "config.json").read_bytes() == before


def test_a_config_that_cannot_be_rewritten_is_reported_not_raised(tmp_path):
    """The repair runs after the weights are on disk, so a folder it cannot write
    to must cost a warning, not the save."""
    if os.name == "nt":
        pytest.skip("read-only file permissions are not enforced the same way on Windows")
    if os.geteuid() == 0:
        pytest.skip("root ignores the read-only bit")

    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    config_path = tmp_path / "config.json"
    config_path.chmod(0o444)
    try:
        assert reconcile_mtp_config(tmp_path) == "unknown"
        # Unwritable means unchanged, not half written.
        assert MTP_CONFIG_KEY in json.dumps(_saved(tmp_path))
    finally:
        config_path.chmod(0o644)


def test_a_dump_that_fails_part_way_leaves_the_original_config(tmp_path, monkeypatch):
    """The repair runs after the weights are on disk and promises it cannot break the export.

    Opening the real file "w" truncates it before `json.dump` has written a byte, so a dump
    that fails part way -- a disk that fills at the end of an export is the realistic one --
    left the checkpoint with an empty or half-written config.json while the handler reported
    "unknown" and the save carried on.
    """
    import unsloth_zoo.saving_utils as saving_utils

    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    config_path = tmp_path / "config.json"
    before = config_path.read_bytes()

    def _fails(obj, handle, **kwargs):
        handle.write('{"partial": ')
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(saving_utils.json, "dump", _fails)
    assert reconcile_mtp_config(tmp_path) == "unknown"
    assert config_path.read_bytes() == before
    assert MTP_CONFIG_KEY in json.dumps(_saved(tmp_path))
    # And nothing is left behind in the checkpoint folder.
    assert [p.name for p in tmp_path.glob("config.json.*")] == []


def test_the_repaired_config_keeps_the_mode_it_had(tmp_path):
    """Staged through a temporary file, which is created 0600. A repaired export must not
    become unreadable to everyone but the user who ran it."""
    if os.name == "nt":
        pytest.skip("POSIX file modes are not what Windows enforces")
    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    config_path = tmp_path / "config.json"
    config_path.chmod(0o644)
    assert reconcile_mtp_config(tmp_path) == "stripped"
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o644


def test_a_read_only_config_is_left_alone_even_as_root(tmp_path, monkeypatch):
    """`os.access(..., W_OK)` answers "may this process write it", and as root that is yes
    for a `0444` file.

    Containers and hosted notebooks run as root by default, which is where an export most
    often runs, so the access check alone let a replacement land on a config the operator had
    explicitly marked read-only -- the exact thing the check exists to refuse. Root is
    simulated here rather than required, so the case is covered on an ordinary CI user too.
    """
    if os.name == "nt":
        pytest.skip("read-only file permissions are not enforced the same way on Windows")

    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    config_path = tmp_path / "config.json"
    before = config_path.read_text(encoding = "utf-8")
    config_path.chmod(0o444)
    # What root sees: the access check says yes whatever the mode says.
    monkeypatch.setattr(os, "access", lambda path, mode: True)
    try:
        assert reconcile_mtp_config(tmp_path) == "unknown"
        assert config_path.read_text(encoding = "utf-8") == before
        assert MTP_CONFIG_KEY in before
        # And nothing staged is left beside it.
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("config.json.")) == []
    finally:
        config_path.chmod(0o644)


def test_a_writable_config_is_still_rewritten(tmp_path, monkeypatch):
    """The other half: the mode check must not refuse an ordinary file, and a mode that
    cannot be read at all is "cannot tell" rather than a refusal."""
    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    assert reconcile_mtp_config(tmp_path) == "stripped"
    assert MTP_CONFIG_KEY not in json.dumps(_saved(tmp_path))

    # And a mode that cannot be read at all is "cannot tell", which falls back to the access
    # check rather than refusing: an unreadable stat must not turn every export into a
    # warning. Asked of the predicate directly, since `reconcile_mtp_config` stats the path
    # for other reasons too.
    from unsloth_zoo.saving_utils import _config_is_writable

    config_path = tmp_path / "config.json"
    assert _config_is_writable(config_path) is True
    real_stat = os.stat
    monkeypatch.setattr(
        os, "stat",
        lambda path, *a, **k: (_ for _ in ()).throw(OSError("stat is not available here"))
        if str(path) == str(config_path)
        else real_stat(path, *a, **k),
    )
    assert _config_is_writable(config_path) is True


# ---------------------------------------------------------------------------
# Nested config shapes. The three nested containers are the ones
# _sync_gguf_nextn_layer_config in unsloth_zoo/mlx/utils.py already reads; a
# declaration missed here is left in the exported config with no weights behind it,
# which is the state that makes a later GGUF conversion fail on missing MTP layers.
# ---------------------------------------------------------------------------

from unsloth_zoo.saving_utils import _mtp_config_containers


@pytest.mark.parametrize("shape, build", [
    ("top_level",                  lambda k: {k: 1}),
    ("text_config",                lambda k: {"text_config": {k: 1}}),
    ("language_config",            lambda k: {"language_config": {k: 1}}),
    ("thinker_config_text_config", lambda k: {"thinker_config": {"text_config": {k: 1}}}),
])
def test_every_supported_nested_shape_is_reconciled(tmp_path, shape, build):
    folder = tmp_path / shape
    folder.mkdir()
    _write_checkpoint(folder, ["model.layers.0.self_attn.q_proj.weight"])
    (folder / "config.json").write_text(json.dumps(build(MTP_CONFIG_KEY)), encoding = "utf-8")

    assert reconcile_mtp_config(str(folder)) == "stripped"
    rewritten = (folder / "config.json").read_text(encoding = "utf-8")
    assert MTP_CONFIG_KEY not in rewritten, f"{shape}: the unbacked declaration survived"


def test_a_declaration_backed_by_weights_is_kept_in_a_nested_shape(tmp_path):
    folder = tmp_path / "backed"
    folder.mkdir()
    _write_checkpoint(folder, [
        "model.layers.0.self_attn.q_proj.weight",
        "model.mtp.0.weight",
    ])
    (folder / "config.json").write_text(
        json.dumps({"language_config": {MTP_CONFIG_KEY: 1}}), encoding = "utf-8")
    assert reconcile_mtp_config(str(folder)) == "agrees"
    kept = json.loads((folder / "config.json").read_text(encoding = "utf-8"))
    assert kept["language_config"][MTP_CONFIG_KEY] == 1


def test_the_same_container_reachable_twice_is_collected_once():
    """Identity, not equality: two shapes can hold equal dicts and both need
    rewriting, while one dict reachable by two paths must be collected once."""
    shared = {MTP_CONFIG_KEY: 1}
    aliased = {MTP_CONFIG_KEY: 1, "text_config": shared, "language_config": shared}
    assert len(_mtp_config_containers(aliased)) == 2
    distinct = {"text_config": {MTP_CONFIG_KEY: 1}, "language_config": {MTP_CONFIG_KEY: 1}}
    assert len(_mtp_config_containers(distinct)) == 2
