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

transformers drops Qwen3.5's `mtp.*` tensors on load, so a re-saved export keeps
`mtp_num_hidden_layers` with no weights behind it (unsloth#7681).
"""

import json
import os
import stat

import numpy as np
import pytest
# numpy rather than torch so this runs on a CPU-only runner with no torch wheel.
from safetensors.numpy import save_file

from unsloth_zoo.saving_utils import (
    MTP_CONFIG_KEY,
    _checkpoint_tensor_names,
    is_mtp_tensor_name,
    mtp_head_is_present,
    reconcile_mtp_config,
)


def checkpoint_mtp_tensor_names(folder):
    """The MTP names in a saved checkpoint, or None when unreadable."""
    names = _checkpoint_tensor_names(folder)
    if names is None:
        return None
    return sorted(name for name in names if is_mtp_tensor_name(name))

# The names Qwen/Qwen3.5-0.8B ships, trimmed to the distinct shapes.
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
    # The older Qwen2-VL prefix ordering, seen in foreign checkpoints.
    "language_model.model.mtp.fc.weight",
])
def test_mtp_tensor_names_are_recognised(name):
    assert is_mtp_tensor_name(name)


@pytest.mark.parametrize("name", [
    # Body, visual tower, then two near misses.
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
    """Unreadable weights are not a missing MTP head."""
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
    """The reproduced defect: declaration kept, no `mtp.*` tensor."""
    _write_config(tmp_path, nested = nested)
    _write_checkpoint(tmp_path, BODY_NAMES)

    assert reconcile_mtp_config(tmp_path) == "stripped"

    saved = _saved(tmp_path)
    holder = saved["text_config"] if nested else saved
    assert MTP_CONFIG_KEY not in holder
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
    """DeepSeek-V3 / GLM heads live in `layers.N` past `num_hidden_layers`."""
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
    """The key may be top level while `num_hidden_layers` sits in `text_config`."""
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
    """No layer count means the conservative answer, not a guess."""
    names = ("model.layers.9.mlp.up_proj.weight",)
    assert mtp_head_is_present(names) is False
    assert mtp_head_is_present(names + ("mtp.fc.weight",)) is True


def test_shared_rule_accepts_a_live_config_object():
    """`unsloth/save.py` passes a PretrainedConfig, so attribute access must work."""
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
    """Called once per config holder, so a generator must not be exhausted."""
    names = ("model.layers.0.mlp.up_proj.weight", "mtp.fc.weight")
    generator = (name for name in names)
    assert mtp_head_is_present(generator) is True


# ---- against the real published config, and an unwritable folder -----------


@pytest.fixture
def real_qwen35_config():
    """`config.json` as Qwen publishes it. Skipped when the Hub is unreachable."""
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
    """Qwen declares it in `text_config`; only that key may move."""
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
    """An unwritable folder costs a warning, not the save."""
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
        assert MTP_CONFIG_KEY in json.dumps(_saved(tmp_path))
    finally:
        config_path.chmod(0o644)


def test_a_dump_that_fails_part_way_leaves_the_original_config(tmp_path, monkeypatch):
    """A dump that fails part way must leave the original config whole."""
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
    assert [p.name for p in tmp_path.glob("config.json.*")] == []


def test_the_repaired_config_keeps_the_mode_it_had(tmp_path):
    """Staging goes through a 0600 temp file; the original mode must survive."""
    if os.name == "nt":
        pytest.skip("POSIX file modes are not what Windows enforces")
    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    config_path = tmp_path / "config.json"
    config_path.chmod(0o644)
    assert reconcile_mtp_config(tmp_path) == "stripped"
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o644


def test_a_read_only_config_is_left_alone_even_as_root(tmp_path, monkeypatch):
    """`os.access(W_OK)` says yes to a 0444 file as root; the mode must still win."""
    if os.name == "nt":
        pytest.skip("read-only file permissions are not enforced the same way on Windows")

    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    config_path = tmp_path / "config.json"
    before = config_path.read_text(encoding = "utf-8")
    config_path.chmod(0o444)
    monkeypatch.setattr(os, "access", lambda path, mode: True)
    try:
        assert reconcile_mtp_config(tmp_path) == "unknown"
        assert config_path.read_text(encoding = "utf-8") == before
        assert MTP_CONFIG_KEY in before
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("config.json.")) == []
    finally:
        config_path.chmod(0o644)


def test_a_writable_config_is_still_rewritten(tmp_path, monkeypatch):
    """An ordinary file is rewritten, and an unreadable mode is "cannot tell"."""
    _write_config(tmp_path)
    _write_checkpoint(tmp_path, BODY_NAMES)
    assert reconcile_mtp_config(tmp_path) == "stripped"
    assert MTP_CONFIG_KEY not in json.dumps(_saved(tmp_path))

    # Asked of the predicate directly; `reconcile_mtp_config` stats for other reasons too.
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


# --- Nested config shapes, matching _sync_gguf_nextn_layer_config in mlx/utils.py. ---

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
    """Identity, not equality: equal-but-distinct dicts both count."""
    shared = {MTP_CONFIG_KEY: 1}
    aliased = {MTP_CONFIG_KEY: 1, "text_config": shared, "language_config": shared}
    assert len(_mtp_config_containers(aliased)) == 2
    distinct = {"text_config": {MTP_CONFIG_KEY: 1}, "language_config": {MTP_CONFIG_KEY: 1}}
    assert len(_mtp_config_containers(distinct)) == 2
