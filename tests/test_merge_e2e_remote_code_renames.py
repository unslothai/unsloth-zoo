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

"""A trust_remote_code model whose checkpoint is renamed on load must still merge.

LFM3-Audio-2B ships `lfm.model.layers.N...` on disk and its remote modeling file registers a
Transformers 5 `WeightRenaming` onto `model.language_model.` under its `model_type`. The class
lives in `transformers_modules`, not `transformers`, so the merge never found that registry
entry, matched no LoRA module, and wrote the base weights back as a successful merge.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch

import _merge_e2e_helpers as H

import unsloth_zoo.saving_utils as SU
from unsloth_zoo.saving_utils import PartialLoraMergeError

_MODEL_TYPE = "zoo_test_renamed_llama"
_CLASS_NAME = "ZooTestRenamedForCausalLM"
_MODULE_NAME = "transformers_modules.zoo_test_repo.modeling_renamed"
_DISK_PREFIX = "lfm.body."
_LIVE_PREFIX = "model."
_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _require_registry():
    try:
        from transformers.conversion_mapping import register_checkpoint_conversion_mapping  # noqa: F401
        from transformers.core_model_loading import WeightRenaming  # noqa: F401
    except Exception:
        pytest.skip("no Transformers 5 conversion registry")
    if not H.family_available("llama"):
        pytest.skip("llama unavailable in this transformers")


@pytest.fixture
def remote_class():
    """A model class defined the way trust_remote_code defines one: in a `transformers_modules`
    module, with its renaming registered under its own model_type."""
    _require_registry()
    import transformers as T
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    config_class = type("ZooTestRenamedConfig", (T.LlamaConfig,), {"model_type": _MODEL_TYPE})
    model_class = type(_CLASS_NAME, (T.LlamaForCausalLM,), {"config_class": config_class})
    module = types.ModuleType(_MODULE_NAME)
    setattr(module, _CLASS_NAME, model_class)
    model_class.__module__ = _MODULE_NAME
    sys.modules[_MODULE_NAME] = module
    register_checkpoint_conversion_mapping(
        _MODEL_TYPE,
        [WeightRenaming(source_patterns = r"^lfm\.body\.", target_patterns = _LIVE_PREFIX)],
        overwrite = True,
    )
    try:
        yield model_class, config_class
    finally:
        sys.modules.pop(_MODULE_NAME, None)


def _stage(work_dir, model_class, config_class):
    """Tiny base saved under the legacy prefix, plus a seeded LoRA on the live module paths."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file

    H.set_offline_cpu_env()
    base_dir = os.path.join(work_dir, "base")
    out_dir = os.path.join(work_dir, "merged")
    cfg = config_class(
        hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
        num_attention_heads = 4, num_key_value_heads = 2, vocab_size = 64,
        max_position_embeddings = 64, tie_word_embeddings = False,
    )
    torch.manual_seed(H.SEED)
    model = model_class(cfg).to(torch.float32)
    model.save_pretrained(base_dir, safe_serialization = True)
    model.config._name_or_path = base_dir

    shard = os.path.join(base_dir, "model.safetensors")
    tensors = load_file(shard)
    renamed = {
        (_DISK_PREFIX + key[len(_LIVE_PREFIX):] if key.startswith(_LIVE_PREFIX) else key): value
        for key, value in tensors.items()
    }
    save_file(renamed, shard, metadata = {"format": "pt"})

    torch.manual_seed(H.SEED)
    peft_model = get_peft_model(model, LoraConfig(
        r = 8, lora_alpha = 16, lora_dropout = 0.0, bias = "none", target_modules = _TARGETS,
    ))
    H.seed_lora(peft_model)
    return base_dir, out_dir, renamed, peft_model, H.extract_adapted(peft_model)


def test_registry_mapping_is_found_for_a_remote_code_class(remote_class):
    assert SU._get_checkpoint_conversion_mapping(_CLASS_NAME) == {r"^lfm\.body\.": _LIVE_PREFIX}


def test_remote_code_merge_applies_the_registered_rename(tmp_path, remote_class):
    base_dir, out_dir, base_tensors, peft_model, adapted = _stage(str(tmp_path), *remote_class)
    assert all(key.startswith(_LIVE_PREFIX) for key in adapted), sorted(adapted)[:3]

    H.run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)
    merged = H.read_safetensors_dir(out_dir)
    assert set(merged) == set(base_tensors)

    n_adapted = 0
    for live_key, record in adapted.items():
        disk_key = _DISK_PREFIX + live_key[len(_LIVE_PREFIX):]
        reference = (
            base_tensors[disk_key].to(torch.float64)
            + record.alpha * (record.lora_B.to(torch.float64) @ record.lora_A.to(torch.float64))
        )
        torch.testing.assert_close(merged[disk_key].to(torch.float64), reference, atol = 1e-5, rtol = 1e-5)
        n_adapted += 1
    assert n_adapted == 2 * len(_TARGETS)
    for key in base_tensors:
        if not any(key == _DISK_PREFIX + k[len(_LIVE_PREFIX):] for k in adapted):
            assert torch.equal(merged[key], base_tensors[key]), key


def test_a_merge_that_places_no_lora_module_refuses(tmp_path, remote_class, monkeypatch):
    """No bridge at all (a rename the merge cannot see) must refuse, not write the base back."""
    base_dir, out_dir, base_tensors, peft_model, _ = _stage(str(tmp_path), *remote_class)
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)

    with pytest.raises(PartialLoraMergeError, match = "none of the"):
        H.run_merge(peft_model, base_dir, out_dir, save_dtype = torch.float32)

    if os.path.isdir(out_dir):
        for key, tensor in H.read_safetensors_dir(out_dir).items():
            assert torch.equal(tensor, base_tensors[key]), f"{key} was rewritten before the refusal"


def _lora_weights(entries):
    import collections
    weights = collections.defaultdict(lambda: SU.LoraStats(None, None, None, 0))
    weights.update(entries)
    return weights


def _lora(out_features, in_features):
    module = torch.nn.Linear(in_features, out_features, bias = False)
    return SU.LoraStats(module, torch.zeros(2, in_features), torch.zeros(out_features, 2), 1.0)


def test_adapters_only_on_an_omitted_tower_still_save(monkeypatch):
    """Every target in a component the export leaves out: no tensor anywhere to rename to."""
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)
    lora = _lora_weights({f"model.vision_tower.layers.{i}.attn.q_proj": _lora(48, 48) for i in range(2)})
    shapes = {f"model.layers.{i}.self_attn.q_proj": (32, 32) for i in range(2)}
    disk = ({k + ".weight" for k in shapes}, shapes)
    assert SU._check_lora_merge_is_complete(None, [], lora, "LlamaForCausalLM", disk = disk) == {}


def test_adapters_whose_weights_sit_under_another_prefix_refuse(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)
    monkeypatch.setattr(SU, "_get_checkpoint_conversion_mapping", lambda name: {})
    lora = _lora_weights({f"model.layers.{i}.self_attn.q_proj": _lora(32, 32) for i in range(2)})
    shapes = {f"lfm.body.layers.{i}.self_attn.q_proj": (32, 32) for i in range(2)}
    disk = ({k + ".weight" for k in shapes}, shapes)
    with pytest.raises(PartialLoraMergeError, match = "none of the"):
        SU._check_lora_merge_is_complete(None, [], lora, "LlamaForCausalLM", disk = disk)


def test_the_merged_class_wins_over_a_same_named_remote_class(remote_class, monkeypatch):
    """Another trust_remote_code model loaded earlier may define a class of the same name."""
    model_class, _ = remote_class
    other_config = type("OtherConfig", (model_class.config_class,), {"model_type": "zoo_test_other"})
    other = type(_CLASS_NAME, (model_class,), {"config_class": other_config,
                                               "_checkpoint_conversion_mapping": {r"^other\.": "model."}})
    module = types.ModuleType("transformers_modules.aaa_other_repo.modeling_other")
    setattr(module, _CLASS_NAME, other)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(SU._MERGE_MODEL_CLASSES, _CLASS_NAME, model_class)
    assert SU._get_checkpoint_conversion_mapping(_CLASS_NAME) == {r"^lfm\.body\.": _LIVE_PREFIX}


def test_an_omitted_tower_sharing_names_and_shapes_with_the_text_model_still_saves(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ALLOW_PARTIAL_LORA_MERGE", raising = False)
    lora = _lora_weights({
        f"model.vision_tower.layers.{i}.self_attn.q_proj": _lora(32, 32) for i in range(2)
    })
    shapes = {f"model.language_model.layers.{i}.self_attn.q_proj": (32, 32) for i in range(2)}
    disk = ({k + ".weight" for k in shapes}, shapes)
    assert SU._check_lora_merge_is_complete(None, [], lora, "LlamaForCausalLM", disk = disk) == {}
