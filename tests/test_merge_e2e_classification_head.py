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

"""End-to-end merge correctness for a trained head the base checkpoint does not have.

`merged_16bit` takes its weights from the base checkpoint and rewrites the shards in
place, so a `modules_to_save` head trained on a causal-LM base (sequence classification,
token classification) had no key to be written into and was dropped. The Step-7 count
excluded it from both sides, so nothing raised: the export simply reloaded with a fresh
random head and the wrong label count.

The load-bearing assertion is that the reloaded head is bit-identical to the trained one.
A shape match alone passes against a randomly initialized head of the right size.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import _merge_e2e_helpers as H

LABELS = {0: "neg", 1: "neu", 2: "pos", 3: "mixed", 4: "other"}


def _labels(num_labels):
    """id2label sized to the head; a short map would silently redefine num_labels."""
    return {i: LABELS.get(i, f"label_{i}") for i in range(num_labels)}


def _distinct_head(out_features, in_features):
    """A head no random initializer would produce, so a re-init cannot pass by luck."""
    return torch.arange(out_features * in_features, dtype=torch.float32).reshape(
        out_features, in_features) / 97.0


def _build_seqcls_case(tmp_path, *, num_labels=5, dtype=torch.float32, labels=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")

    base = str(tmp_path / "base")
    spec = H.make_spec("llama")
    torch.manual_seed(H.SEED)
    AutoModelForCausalLM.from_config(spec.config).to(torch.float32).save_pretrained(
        base, safe_serialization=True)

    config = AutoConfig.from_pretrained(base)
    config.num_labels = num_labels
    if labels:
        config.id2label = _labels(num_labels)
        config.label2id = {v: k for k, v in config.id2label.items()}
        config.problem_type = "single_label_classification"
    torch.manual_seed(H.SEED)
    model = AutoModelForSequenceClassification.from_config(config).to(torch.float32)
    model.config._name_or_path = base

    peft_model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"], modules_to_save=["score"],
        task_type="SEQ_CLS"))
    H.seed_lora(peft_model)

    head = _distinct_head(num_labels, spec.config.hidden_size)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if name.endswith("score.modules_to_save.default.weight"):
                param.copy_(head)

    out = str(tmp_path / "out")
    H.run_merge(peft_model, base, out, save_dtype=dtype)
    return base, out, head


def _exported_config(out):
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def test_trained_head_survives_the_merge(tmp_path):
    """The regression: on the unfixed code `score.weight` is absent and the reload is random."""
    from transformers import AutoModelForSequenceClassification

    _, out, head = _build_seqcls_case(tmp_path)

    written = H.read_safetensors_dir(out)
    assert "score.weight" in written, sorted(written)

    reloaded, info = AutoModelForSequenceClassification.from_pretrained(
        out, output_loading_info=True, local_files_only=True)
    assert [k for k in info["missing_keys"] if "score" in k] == []
    assert torch.equal(reloaded.score.weight.detach().float(), head)


def test_label_maps_and_architecture_follow_the_head(tmp_path):
    _, out, _ = _build_seqcls_case(tmp_path)

    data = _exported_config(out)
    assert data["architectures"] == ["LlamaForSequenceClassification"]
    assert data["id2label"] == {str(k): v for k, v in _labels(5).items()}
    assert data["label2id"] == {v: k for k, v in _labels(5).items()}
    assert data["problem_type"] == "single_label_classification"


def test_the_head_bias_travels_with_its_weight():
    """Unit level on purpose: Llama builds `score` with bias=False, so an end-to-end case
    cannot show this. Heads that do carry a bias must not have it left behind."""
    import collections
    from unsloth_zoo.saving_utils import LoraStats, _unbacked_trained_tensors

    class _BiasedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modules_to_save = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 3, bias=True)})

    stats = LoraStats(None, None, None, 0)
    stats.module = _BiasedHead()
    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    lora_weights["score"] = stats

    found = _unbacked_trained_tensors(lora_weights, {"model.embed_tokens.weight"},
                                      "LlamaForCausalLM")
    assert set(found) == {"score.weight", "score.bias"}
    assert tuple(found["score.bias"].shape) == (3,)


@pytest.mark.parametrize("num_labels", [2, 3, 5, 17])
def test_any_label_count_round_trips(tmp_path, num_labels):
    from transformers import AutoModelForSequenceClassification

    _, out, head = _build_seqcls_case(tmp_path, num_labels=num_labels)
    reloaded = AutoModelForSequenceClassification.from_pretrained(out, local_files_only=True)
    assert reloaded.config.num_labels == num_labels
    assert torch.equal(reloaded.score.weight.detach().float(), head)


def test_the_backbone_merge_is_unchanged_by_the_seeding(tmp_path):
    """Rewriting a shard to add the head must not disturb the tensors already in it."""
    base, out, _ = _build_seqcls_case(tmp_path)

    base_tensors = H.read_safetensors_dir(base)
    written = H.read_safetensors_dir(out)
    assert set(written) == set(base_tensors) | {"score.weight"}
    for key, tensor in base_tensors.items():
        if any(t in key for t in ("q_proj", "v_proj")):
            continue                                    # LoRA targets, expected to differ
        assert torch.equal(written[key], tensor), key


def test_a_half_precision_export_keeps_the_head(tmp_path):
    from transformers import AutoModelForSequenceClassification

    _, out, head = _build_seqcls_case(tmp_path, dtype=torch.bfloat16)
    written = H.read_safetensors_dir(out)
    assert written["score.weight"].dtype == torch.bfloat16
    reloaded = AutoModelForSequenceClassification.from_pretrained(out, local_files_only=True)
    got = reloaded.score.weight.detach().float()
    assert torch.allclose(got, head, atol=1e-2, rtol=1e-2)


def test_a_plain_causal_lm_export_is_untouched(tmp_path):
    """No `modules_to_save` head, so nothing is seeded and no head fields are written."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    H.set_offline_cpu_env()
    if not H.family_available("llama"):
        pytest.skip("llama unavailable")
    base, out = str(tmp_path / "base"), str(tmp_path / "out")
    spec = H.make_spec("llama")
    torch.manual_seed(H.SEED)
    AutoModelForCausalLM.from_config(spec.config).to(torch.float32).save_pretrained(
        base, safe_serialization=True)
    model = AutoModelForCausalLM.from_pretrained(base)
    model.config._name_or_path = base
    peft_model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    H.seed_lora(peft_model)
    H.run_merge(peft_model, base, out, save_dtype=torch.float32)

    assert set(H.read_safetensors_dir(out)) == set(H.read_safetensors_dir(base))
    data = _exported_config(out)
    assert data["architectures"] == ["LlamaForCausalLM"]
    assert "id2label" not in data or len(data["id2label"]) == 2   # transformers' own default


# ---- unit level -------------------------------------------------------------------

def test_only_modules_to_save_is_seeded_not_an_unbacked_lora():
    """A LoRA target missing from the base has no trained tensor of its own to lose, so it
    must stay excluded; only `modules_to_save` carries weights that exist nowhere else."""
    import collections
    from unsloth_zoo.saving_utils import LoraStats, _unbacked_trained_tensors

    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    lora_weights["vision_tower.layers.0.q_proj"] = LoraStats(None, None, None, 0)
    assert _unbacked_trained_tensors(lora_weights, {"model.layers.0.q_proj.weight"},
                                     "LlamaForCausalLM") == {}


def test_a_backed_head_is_not_seeded_twice():
    """When the base already ships the key, the ordinary in-place merge owns it."""
    import collections
    from unsloth_zoo.saving_utils import LoraStats, _unbacked_trained_tensors

    class _Saved(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modules_to_save = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 3)})

    stats = LoraStats(None, None, None, 0)
    stats.module = _Saved()
    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    lora_weights["score"] = stats
    assert _unbacked_trained_tensors(lora_weights, {"score.weight"}, "LlamaForCausalLM") == {}
    assert "score.weight" in _unbacked_trained_tensors(
        lora_weights, {"model.embed_tokens.weight"}, "LlamaForCausalLM")


@pytest.mark.parametrize("name", ["lm_head", "embed_tokens", "model.language_model.lm_head"])
def test_backbone_tensors_are_never_seeded(name):
    """gemma3 ties its text embeddings, so that architecture has no `lm_head` slot at all.
    Seeding one put an unexpected key in the export and broke the text_only resize case on
    transformers 4.51.3, which is why this exclusion exists rather than a general rule."""
    import collections
    from unsloth_zoo.saving_utils import LoraStats, _unbacked_trained_tensors

    class _Saved(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modules_to_save = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 3)})

    stats = LoraStats(None, None, None, 0)
    stats.module = _Saved()
    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    lora_weights[name] = stats
    assert _unbacked_trained_tensors(lora_weights, {"model.layers.0.q_proj.weight"},
                                     "LlamaForCausalLM") == {}


def test_a_sharded_export_records_the_head_in_its_index(tmp_path):
    """transformers resolves a sharded checkpoint through the index alone, so a key absent
    from the weight_map is invisible even though the bytes are on disk."""
    from unsloth_zoo.saving_utils import _add_keys_to_index

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({
        "metadata": {"total_size": 8},
        "weight_map": {"model.embed_tokens.weight": "model-00001-of-00002.safetensors"},
    }), encoding="utf-8")

    seeded = {"score.weight": "model-00002-of-00002.safetensors"}
    assert _add_keys_to_index(str(tmp_path), seeded) is True
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    assert weight_map["score.weight"] == "model-00002-of-00002.safetensors"
    assert weight_map["model.embed_tokens.weight"] == "model-00001-of-00002.safetensors"

    # Idempotent: a second pass has nothing to add, so nothing is re-uploaded.
    assert _add_keys_to_index(str(tmp_path), seeded) is False


def test_an_unsharded_export_has_no_index_to_touch(tmp_path):
    from unsloth_zoo.saving_utils import _add_keys_to_index
    assert _add_keys_to_index(str(tmp_path), {"score.weight": "model.safetensors"}) is False


def test_seeding_refuses_rather_than_dropping_the_head_when_disk_is_short(tmp_path, monkeypatch):
    """Adding a key rewrites the shard through a temp copy, and appending moves every offset
    so there is no in-place fallback. Refusing beats silently dropping the head, which is the
    bug this whole path exists to fix."""
    import collections
    import shutil as _shutil
    from safetensors.torch import save_file
    from unsloth_zoo.saving_utils import LoraStats, _seed_unbacked_trained_tensors

    save_file({"model.embed_tokens.weight": torch.zeros(4, 4)},
              str(tmp_path / "model.safetensors"), metadata={"format": "pt"})

    class _Saved(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.modules_to_save = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 3)})

    stats = LoraStats(None, None, None, 0)
    stats.module = _Saved()
    lora_weights = collections.defaultdict(lambda: LoraStats(None, None, None, 0))
    lora_weights["score"] = stats

    monkeypatch.setattr(_shutil, "disk_usage",
                        lambda _p: type("U", (), {"free": 1024})())
    with pytest.raises(RuntimeError, match="not enough free disk"):
        _seed_unbacked_trained_tensors(str(tmp_path), ["model.safetensors"],
                                       lora_weights, "LlamaForCausalLM")
    # The shard must be left exactly as it was, not half-rewritten.
    assert set(H.read_safetensors_dir(str(tmp_path))) == {"model.embed_tokens.weight"}
    assert not list(tmp_path.glob("*.unsloth_seed_tmp"))


def test_the_corrected_config_is_re_uploaded_when_pushing():
    """Step 2 uploads config.json before the head exists and Step 7's folder re-upload is
    skipped in low-disk mode, so without this the remote keeps the causal-LM config beside
    shards that hold the head."""
    import inspect
    from unsloth_zoo.saving_utils import merge_and_overwrite_lora

    src = inspect.getsource(merge_and_overwrite_lora)
    seeded = src.split("_seeded_head_keys", 1)[1]
    block = seeded.split("for filename in ProgressBar", 1)[0]
    assert 'upload_items("config.json")' in block, (
        "the seeding block no longer re-uploads config.json; a low-disk push would leave "
        "the remote config describing the base architecture instead of the trained head."
    )


def test_the_mxfp4_rewrite_counts_a_seeded_head(tmp_path):
    """An mxfp4 base with save_method='merged_16bit' routes to _merge_and_overwrite_lora_mxfp4,
    whose every `count += 1` was gated on lora_A. A seeded head has no lora_A, so it counted
    zero there while the Step-7 `_count_backed_lora_modules` counted it as backed, and the
    export aborted on the count mismatch. The dense path has always had this branch.
    """
    import collections
    from safetensors.torch import save_file
    from unsloth_zoo.saving_utils import LoraStats, _merge_and_overwrite_lora_mxfp4

    head = torch.full((3, 4), 0.5)

    class _Saved(torch.nn.Module):
        def __init__(self):
            super().__init__()
            linear = torch.nn.Linear(4, 3, bias=False)
            with torch.no_grad():
                linear.weight.copy_(head)
            self.modules_to_save = torch.nn.ModuleDict({"default": linear})

    stats = LoraStats(None, None, None, 0)
    stats.module = _Saved()
    lora_weights = collections.defaultdict(lambda: None)
    lora_weights["score"] = stats

    # A 16-bit tensor coexisting with the mxfp4 pair is exactly where a seeded head lands.
    fname = "model.safetensors"
    save_file({"score.weight": torch.zeros(3, 4, dtype=torch.bfloat16)},
              str(tmp_path / fname), metadata={"format": "pt"})

    count, keys = _merge_and_overwrite_lora_mxfp4(
        save_directory=str(tmp_path), filename=fname, lora_weights=lora_weights,
        output_dtype=torch.bfloat16, model_class_name="LlamaForSequenceClassification",
    )
    assert count == 1, (
        f"the mxfp4 rewrite counted {count} saved modules for one modules_to_save head; "
        "Step-7 would see one more backed module than saved and abort the export"
    )
    written = H.read_safetensors_dir(str(tmp_path))["score.weight"]
    assert torch.equal(written.float(), head), "the trained head was not written"


def test_appending_to_a_shard_leaves_the_existing_tensors_byte_identical(tmp_path):
    """`_stream_rewrite_resized_shard` gained an append path; the copy path must not move."""
    from safetensors.torch import save_file
    from unsloth_zoo.saving_utils import _stream_rewrite_resized_shard

    src = str(tmp_path / "model.safetensors")
    original = {"a": torch.randn(4, 5), "b": torch.arange(6, dtype=torch.float32)}
    save_file(original, src, metadata={"format": "pt"})

    with open(src, "rb") as f:
        length_of_header = int.from_bytes(f.read(8), "little")
        header_metadata = json.loads(f.read(length_of_header))

    added = {"score.weight": torch.full((3, 5), 0.25)}
    dst = str(tmp_path / "out.safetensors")
    _stream_rewrite_resized_shard(src, dst, header_metadata, length_of_header, added)

    from safetensors.torch import load_file
    got = load_file(dst)
    assert set(got) == {"a", "b", "score.weight"}
    for key, tensor in original.items():
        assert torch.equal(got[key], tensor), key
    assert torch.equal(got["score.weight"], added["score.weight"])
