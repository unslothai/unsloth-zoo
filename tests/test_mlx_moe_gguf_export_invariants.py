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

"""Invariants the MoE GGUF staging pass holds whatever it is handed: what it does to a
directory it cannot help, whether it answers twice alike, and whether what it wrote is
the checkpoint the model was saved from and not merely something sanitize() accepts.
Synthetic fixtures on `tests/mlx_simulation/`, so no mlx and no Apple hardware."""

import copy
import hashlib
import json

import pytest


@pytest.fixture(autouse=True, scope="module")
def _install_mlx_torch_shim():
    pytest.importorskip("torch")
    from mlx_simulation import simulate_mlx_on_torch
    simulate_mlx_on_torch()


def _digest(path):
    """Every file in the staging directory, by content."""
    return {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.iterdir())
    }


def _regressions():
    """The neighbouring suite's model and staging fixtures."""
    import test_mlx_save_export_regressions as module
    return module


def _staged(path):
    import unsloth_zoo.mlx.utils as mutils

    return {
        name: tensor
        for file in sorted(path.glob("*.safetensors"))
        for name, tensor in mutils.mx.load(str(file)).items()
    }


def test_a_refused_confirmation_writes_nothing(tmp_path, monkeypatch):
    """The whole-directory proof is the last gate before any file is touched."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model()
    # Unpatched, this fixture is one the pass does rewrite.
    rewritten = reg._stage_moe_directory(tmp_path / "rewritten", model, shards=2)
    assert mutils._prepare_moe_gguf_export_directory(rewritten, model=model) > 0

    refused = reg._stage_moe_directory(tmp_path / "refused", model, shards=2)
    before = _digest(refused)
    monkeypatch.setattr(mutils, "_confirmed_mlx_moe_rewrite", lambda *a, **k: False)
    assert mutils._prepare_moe_gguf_export_directory(refused, model=model) == 0
    assert _digest(refused) == before


@pytest.mark.parametrize("failure", ["raises", "returns_none", "returns_a_list"])
def test_a_broken_sanitizer_is_refused_not_obeyed(tmp_path, failure):
    """A sanitizer that cannot be replayed leaves the directory exactly as staged."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model()
    path = reg._stage_moe_directory(tmp_path, model, shards=2)
    before = _digest(path)

    def broken(weights):
        if failure == "raises":
            raise RuntimeError("this sanitizer is not replayable")
        return None if failure == "returns_none" else list(weights)

    type(model).sanitize = staticmethod(broken)
    assert mutils._prepare_moe_gguf_export_directory(path, model=model) == 0
    assert _digest(path) == before


@pytest.mark.parametrize("model", [None, object()])
def test_a_model_the_pass_cannot_read_is_not_a_crash(tmp_path, model):
    """`save_pretrained_gguf` calls this unconditionally, so it must tolerate anything."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    staged = reg._make_moe_model()
    path = reg._stage_moe_directory(tmp_path, staged, shards=2)
    before = _digest(path)
    assert mutils._prepare_moe_gguf_export_directory(path, model=model) == 0
    assert _digest(path) == before


def test_an_empty_directory_is_not_a_crash(tmp_path):
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    path = tmp_path / "empty"
    path.mkdir()
    assert mutils._prepare_moe_gguf_export_directory(
        path, model=reg._make_moe_model()
    ) == 0
    assert list(path.iterdir()) == []


@pytest.mark.parametrize("shards", [1, 2, 5])
def test_a_model_whose_sanitizer_rewrites_nothing_is_byte_identical(tmp_path, shards):
    """Dense exports pay a scan and nothing else: no shard and no index may move."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model(stacks=False)
    path = reg._stage_moe_directory(tmp_path, model, shards=shards)
    before, index_before = _digest(path), (
        path / "model.safetensors.index.json").read_text()

    assert mutils._prepare_moe_gguf_export_directory(path, model=model) == 0
    assert _digest(path) == before
    assert (path / "model.safetensors.index.json").read_text() == index_before


def test_a_second_pass_over_a_rewritten_directory_changes_nothing(tmp_path):
    """Restoring names twice must not restore them twice over."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model()
    path = reg._stage_moe_directory(tmp_path, model, shards=2)

    assert mutils._prepare_moe_gguf_export_directory(path, model=model) > 0
    once = _digest(path)
    assert mutils._prepare_moe_gguf_export_directory(path, model=model) == 0
    assert _digest(path) == once


def test_the_pass_leaves_the_model_it_was_given_as_it_found_it(tmp_path):
    """Replay runs on the live trained model, thousands of times, during planning."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model()
    path = reg._stage_moe_directory(tmp_path, model, shards=2)
    before_attributes = dict(vars(model))
    before_modules = dict(model.switch_modules)
    before_expected = copy.copy(model.expected)

    assert mutils._prepare_moe_gguf_export_directory(path, model=model) > 0

    assert set(vars(model)) == set(before_attributes)
    assert model.switch_modules == before_modules
    assert set(model.expected) == set(before_expected)
    for name, tensor in before_expected.items():
        assert mutils._mlx_arrays_match(model.expected[name], tensor), name


@pytest.mark.parametrize("shards", [1, 2, 3, 5, 7])
def test_the_result_does_not_depend_on_how_the_checkpoint_was_split(tmp_path, shards):
    """Sharding is a storage detail; the recovered checkpoint must not see it."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model()
    single = reg._stage_moe_directory(tmp_path / "one", model, shards=1)
    mutils._prepare_moe_gguf_export_directory(single, model=model)
    expected = _staged(single)

    path = reg._stage_moe_directory(tmp_path / f"split{shards}", model, shards=shards)
    mutils._prepare_moe_gguf_export_directory(path, model=model)
    got = _staged(path)

    assert set(got) == set(expected)
    for name, tensor in expected.items():
        assert mutils._mlx_arrays_match(got[name], tensor), name


@pytest.mark.parametrize("shards", [1, 2, 3])
def test_the_index_still_describes_the_directory_it_indexes(tmp_path, shards):
    """A GGUF converter reads the index first; a name it lists must be findable."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model()
    path = reg._stage_moe_directory(tmp_path, model, shards=shards)
    assert mutils._prepare_moe_gguf_export_directory(path, model=model) > 0

    weight_map = json.loads(
        (path / "model.safetensors.index.json").read_text())["weight_map"]
    shard_names = {
        file.name: set(mutils.mx.load(str(file)))
        for file in sorted(path.glob("*.safetensors"))
    }
    assert set(weight_map) == set().union(*shard_names.values())
    for name, file in weight_map.items():
        assert name in shard_names[file], f"{name} is not in {file}"


def test_no_stacked_expert_name_survives_the_pass(tmp_path):
    """The whole point: llama.cpp has no mapping for a `switch_mlp` tensor."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model()
    path = reg._stage_moe_directory(tmp_path, model, shards=2)
    assert any(".switch_mlp." in name for name in _staged(path))

    assert mutils._prepare_moe_gguf_export_directory(path, model=model) > 0
    assert not any(".switch_mlp." in name for name in _staged(path))


def test_a_single_expert_checkpoint_is_refused_rather_than_half_recovered(tmp_path):
    """One expert leaves the stack ambiguous, so the pass declines: the export is then
    exactly what it is without this pass, which is the contract."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model(num_experts=1)
    path = reg._stage_moe_directory(tmp_path, model, shards=2)
    before = _digest(path)

    assert mutils._prepare_moe_gguf_export_directory(path, model=model) == 0
    assert _digest(path) == before


@pytest.mark.parametrize("num_experts", [2, 3, 16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_the_recovered_checkpoint_is_the_one_the_model_was_saved_from(
    tmp_path, num_experts, with_bias
):
    """The oracle: any preimage sanitize() accepts is not enough, it has to be *the*
    preimage, under the per-expert HF names the converter reads."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model(num_experts=num_experts, with_bias=with_bias)
    path = reg._stage_moe_directory(tmp_path, model, shards=2)
    staged = _staged(path)

    assert mutils._prepare_moe_gguf_export_directory(path, model=model) > 0
    recovered = _staged(path)

    for layer in range(2):
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            for expert in range(num_experts):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{leaf}.weight"
                assert name in recovered, name

    replayed = model.sanitize(dict(recovered))
    for name, tensor in staged.items():
        assert mutils._mlx_arrays_match(replayed[name], tensor), name


def test_a_helper_called_from_inside_a_comprehension_is_still_read():
    """The vocabulary walk must not depend on the interpreter: before PEP 709 (3.12) a
    comprehension compiles to its own code object, so a helper called from inside one --
    the ordinary shape for a rename -- is absent from the enclosing `co_names`."""
    import unsloth_zoo.mlx.utils as mutils

    class Model:
        def sanitize(self, weights):
            return _renamed_through_a_comprehension(weights)

    vocabulary = mutils._mlx_sanitizer_vocabulary(mutils._mlx_moe_sanitizers(Model()))
    assert "model.layers." in vocabulary
    assert "language_model.model.layers." in vocabulary


def _renamed_through_a_comprehension(weights):
    # The call this test is about: reached only from inside the comprehension.
    return {_a_spelling_only_the_helper_holds(name): tensor
            for name, tensor in weights.items()}


def _a_spelling_only_the_helper_holds(name):
    return name.replace("model.layers.", "language_model.model.layers.")


def test_experts_come_back_in_the_order_the_sanitizer_stacked_them(tmp_path):
    """Expert 3's weights must be expert 3's, not expert 0's: a permutation replays
    identically only if the sanitizer's own order is reproduced."""
    import unsloth_zoo.mlx.utils as mutils

    reg = _regressions()
    model = reg._make_moe_model(num_experts=3)
    path = reg._stage_moe_directory(tmp_path, model, shards=1)
    stacked = dict(_staged(path))
    assert mutils._prepare_moe_gguf_export_directory(path, model=model) > 0
    recovered = _staged(path)

    for layer in range(2):
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            source = stacked[f"model.layers.{layer}.mlp.switch_mlp.{leaf}.weight"]
            for expert in range(3):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{leaf}.weight"
                assert mutils._mlx_arrays_match(recovered[name], source[expert]), name


def test_a_merge_survives_a_sanitizer_that_offsets_a_norm_in_place(tmp_path):
    """mlx-vlm's Qwen3.5-VL-MoE splits `experts.gate_up_proj` into two `switch_mlp`
    tensors and offsets a 1-D norm with `+=` in the same sanitize
    (mlx_vlm/models/qwen3_5_moe/qwen3_5_moe.py:38-48 and :89), writing through to the
    caller's array. The merge proof's floor probe shares marker objects with the set it
    compares, so an uncopied floor sat one offset ahead of every recipe, no merge was
    ever proved, and the export kept the MLX spelling llama.cpp cannot map.
    """
    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx

    class Model:
        def __init__(self):
            self.checkpoint = {
                "model.layers.0.mlp.fused.weight":
                    mx.reshape(mx.arange(8, dtype=mx.float32), (4, 2)),
                "model.layers.0.post_attention_layernorm.weight":
                    mx.arange(2, dtype=mx.float32) + 5.0,
            }
            self.expected = self.sanitize(dict(self.checkpoint))

        def named_modules(self):
            yield "", self

        def sanitize(self, weights):
            out = {}
            for name, tensor in weights.items():
                if "fused" in name:
                    for leaf, half in zip(("left", "right"),
                                          mx.split(tensor, 2, axis=0)):
                        out[name.replace("fused", leaf)] = half
                else:
                    out[name] = tensor
            for name in list(out):
                if name.endswith("layernorm.weight") and out[name].ndim == 1:
                    out[name] += 1.0
            return out

    reg = _regressions()
    model = Model()
    path = reg._stage_moe_directory(tmp_path, model, shards=1)

    assert mutils._prepare_moe_gguf_export_directory(path, model=model) > 0
    assert sorted(_staged(path)) == sorted(model.checkpoint)


def test_a_layout_the_vlm_pass_already_inverted_is_not_inverted_again(
    tmp_path, monkeypatch
):
    """The two passes run back to back over one directory, so the second reads what the
    first wrote. An adjacent-axis move is its own inverse, so a second inversion replays
    to exactly what is on disk and the confirmation accepts it: the tensor would ship
    transposed off the HF layout llama.cpp reads. `depthwise_conv1d.weight` is the shape
    both passes claim (_vlm_gguf_tensor_candidates transposes (0, 2, 1); (2, 1) is in
    _MOE_TENSOR_LAYOUTS)."""
    import json

    import unsloth_zoo.mlx.utils as mutils

    mx = mutils.mx
    name = "vision_tower.depthwise_conv1d.weight"
    hf = mx.reshape(mx.arange(2 * 3 * 4, dtype=mx.float32), (2, 3, 4))

    class Model:
        @staticmethod
        def sanitize(weights):
            return {n: (mx.moveaxis(t, 2, 1) if getattr(t, "ndim", 0) == 3 else t)
                    for n, t in weights.items()}

        def named_modules(self):
            yield "", self

    model = Model()
    path = tmp_path / "merged"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "fake_vlm"}))
    mx.save_safetensors(str(path / "model-00001-of-00001.safetensors"),
                        {name: Model.sanitize({name: hf})[name]},
                        metadata={"format": "mlx"})
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model-00001-of-00001.safetensors"}}))
    monkeypatch.setattr(mutils, "_build_mlx_vlm_sanitize_pipelines",
                        lambda config, model=None: [[(Model, None)]])

    norm_offsets = mutils._mlx_sanitizer_norm_offsets(model)
    relaid_out = set()
    assert mutils._prepare_mlx_gguf_export_directory(
        path, model=model, replay_sanitizers=True, norm_offsets=norm_offsets,
        relaid_out=relaid_out) == 1
    assert relaid_out == {name}
    assert mutils._prepare_moe_gguf_export_directory(
        path, model=model, source_norm_offsets=norm_offsets,
        source_layouts=relaid_out) == 0

    assert mutils._mlx_arrays_match(_staged(path)[name], hf)
