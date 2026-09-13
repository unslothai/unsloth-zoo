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

"""GPU-free regression tests for the vLLM >= 0.25.0 LoRA weights-mapper fix.

vLLM 0.25.0 folds the QKV/MLP fusion (q_proj/k_proj/v_proj -> qkv_proj,
gate/up -> gate_up_proj) into WeightsMapper.orig_to_new_stacked. The LoRA name
parser calls mapper._map_name which drops the shard id, so the constituent
projections collapse onto one fused key and collide in the in-memory LoRA tensor
dict, crashing GRPO fast_inference=True with IndexError during activation.

The fix in vllm_lora_worker_manager.py drops the stacked maps while keeping
genuine renames. vLLM's helper for that is get_unstacked_mapper() on 0.25.0 -
0.28.x and was renamed to get_rename_mapper() in 0.29.0, so both names are
tried, then the orig_to_new_stacked field is cleared directly as a last resort.
These tests drive the real WorkerLoRAManager._load_adapter with light fakes (no
GPU, no real vLLM init) and assert which mapper reaches the loader, for both the
in-memory and local-checkpoint paths and both loader signatures.
"""

import dataclasses
import types
import pytest

import unsloth_zoo.vllm_lora_worker_manager as wm


class _StackedMapper:
    """Fake vLLM 0.25.0 - 0.28.x WeightsMapper exposing get_unstacked_mapper()."""

    method_name = "get_unstacked_mapper"

    def __init__(self):
        self.calls = 0
        self.unstacked = object()  # distinct sentinel returned by the method

    def get_unstacked_mapper(self):
        self.calls += 1
        return self.unstacked


class _RenameMapper(_StackedMapper):
    """Fake vLLM >= 0.29.0 WeightsMapper: the helper is get_rename_mapper()."""

    method_name = "get_rename_mapper"
    get_unstacked_mapper = None  # gone in 0.29.0

    def get_rename_mapper(self):
        self.calls += 1
        return self.unstacked


@dataclasses.dataclass
class _FieldOnlyMapper:
    """A future vLLM that renames the helper again but keeps the field."""

    orig_to_new_stacked: dict
    orig_to_new_substr: dict


class _FakePEFTHelper:
    @staticmethod
    def from_dict(config):
        return _FakePEFTHelper()

    @staticmethod
    def from_local_dir(lora_dir, max_position_embeddings, *args, **kwargs):
        return _FakePEFTHelper()

    def validate_legal(self, lora_config):
        return None


def _make_recording_lora_model_cls(record, *, new_signature):
    """Fake _lora_model_cls whose loaders record the kwargs they receive.

    ``new_signature`` toggles whether the loader exposes ``model_vocab_size``
    (newer vLLM) so we cover _load_adapter's signature branch without installing
    multiple vLLM versions.
    """
    if new_signature:
        def _loader(lora_model_id, peft_helper, dtype, weights_mapper,
                    tensors=None, lora_dir=None, device=None,
                    expected_lora_modules=None, model_vocab_size=None):
            record["weights_mapper"] = weights_mapper
            record["tensors"] = tensors
            record["lora_dir"] = lora_dir
            return types.SimpleNamespace(extra_vocab_size=0, id=lora_model_id)
    else:
        def _loader(lora_model_id, peft_helper, dtype, weights_mapper,
                    tensors=None, lora_dir=None, device=None,
                    expected_lora_modules=None, target_embedding_padding=None,
                    embedding_modules=None, embedding_padding_modules=None):
            record["weights_mapper"] = weights_mapper
            record["tensors"] = tensors
            record["lora_dir"] = lora_dir
            return types.SimpleNamespace(extra_vocab_size=0, id=lora_model_id)

    return types.SimpleNamespace(from_lora_tensors=_loader,
                                 from_local_checkpoint=_loader)


def _make_manager(record, *, mapper, set_mapper_attr=True, new_signature=True):
    """Construct a WorkerLoRAManager via __new__ with only what _load_adapter reads."""
    mgr = object.__new__(wm.WorkerLoRAManager)

    model = types.SimpleNamespace(supported_lora_modules=[], packed_modules_mapping={})
    if set_mapper_attr:
        model.hf_to_vllm_mapper = mapper

    mgr._adapter_manager = types.SimpleNamespace(model=model)
    mgr.max_position_embeddings = 2048
    mgr.lora_config = types.SimpleNamespace(lora_dtype=None, lora_extra_vocab_size=0)
    mgr.vocab_size = 32000
    mgr.embedding_modules = {}
    mgr.embedding_padding_modules = []
    mgr._lora_model_cls = _make_recording_lora_model_cls(record, new_signature=new_signature)
    return mgr


def _in_memory_request():
    return types.SimpleNamespace(
        lora_path=None, lora_config={}, config={},
        lora_tensors={"base_model.model.layer.q_proj.lora_A.weight": object()},
        lora_int_id=1,
    )


def _checkpoint_request():
    return types.SimpleNamespace(
        lora_path="/tmp/does-not-need-to-exist", lora_config={}, config={},
        lora_tensors=None, lora_int_id=2,
    )


@pytest.fixture(autouse=True)
def _patch_vllm_helpers(monkeypatch):
    """Swap in fakes for the vLLM helpers _load_adapter calls, GPU/vLLM-free."""
    monkeypatch.setattr(wm, "PEFTHelper", _FakePEFTHelper, raising=False)
    monkeypatch.setattr(wm, "get_adapter_absolute_path", lambda p: p, raising=False)
    monkeypatch.setattr(wm, "LoRAModel", types.SimpleNamespace, raising=False)


@pytest.mark.parametrize("mapper_cls", [_StackedMapper, _RenameMapper])
@pytest.mark.parametrize("new_signature", [True, False])
def test_unstacked_mapper_used_for_in_memory_tensors(mapper_cls, new_signature):
    record = {}
    mapper = mapper_cls()
    mgr = _make_manager(record, mapper=mapper, new_signature=new_signature)

    mgr._load_adapter(_in_memory_request())

    assert mapper.calls == 1, f"{mapper_cls.method_name} must be called exactly once"
    assert record["weights_mapper"] is mapper.unstacked
    assert record["weights_mapper"] is not mapper
    assert record["weights_mapper"] is not None
    assert record["tensors"] is not None, "should hit the in-memory branch"


@pytest.mark.parametrize("mapper_cls", [_StackedMapper, _RenameMapper])
def test_unstacked_mapper_used_for_local_checkpoint(mapper_cls):
    record = {}
    mapper = mapper_cls()
    mgr = _make_manager(record, mapper=mapper)

    mgr._load_adapter(_checkpoint_request())

    assert mapper.calls == 1
    assert record["weights_mapper"] is mapper.unstacked
    assert record["tensors"] is None, "should hit the local-checkpoint branch"
    assert record["lora_dir"] == "/tmp/does-not-need-to-exist"


def test_rename_mapper_wins_when_both_helpers_exist():
    # A version exposing both must not be served by the deprecated spelling.
    record = {}

    class _Both(_StackedMapper):
        def get_rename_mapper(self):
            self.calls += 1
            return self.unstacked

    mapper = _Both()
    mgr = _make_manager(record, mapper=mapper)
    mgr._load_adapter(_in_memory_request())

    assert mapper.calls == 1
    assert record["weights_mapper"] is mapper.unstacked


def test_stacked_field_is_cleared_when_no_helper_exists():
    # Belt and braces for a third rename: strip orig_to_new_stacked ourselves
    # and leave every other mapping untouched.
    record = {}
    mapper = _FieldOnlyMapper(
        orig_to_new_stacked={".q_proj": (".qkv_proj", "q")},
        orig_to_new_substr={"visual.": "vision_tower."},
    )
    mgr = _make_manager(record, mapper=mapper)
    mgr._load_adapter(_in_memory_request())

    got = record["weights_mapper"]
    assert got.orig_to_new_stacked == {}
    assert got.orig_to_new_substr == {"visual.": "vision_tower."}
    # The original must not be mutated; it belongs to the model.
    assert mapper.orig_to_new_stacked == {".q_proj": (".qkv_proj", "q")}


def test_legacy_mapper_without_unstack_is_forwarded_unchanged():
    # vLLM < 0.25.0: no helper and no stacked field -> pass through intact
    record = {}
    legacy_mapper = object()
    mgr = _make_manager(record, mapper=legacy_mapper)

    mgr._load_adapter(_in_memory_request())

    assert record["weights_mapper"] is legacy_mapper


def test_noncallable_unstacked_attribute_is_forwarded_unchanged():
    # Defensive: an attribute exists but is not callable -> do not invoke it
    record = {}
    weird_mapper = types.SimpleNamespace(get_unstacked_mapper="not-callable")
    mgr = _make_manager(record, mapper=weird_mapper)

    mgr._load_adapter(_in_memory_request())

    assert record["weights_mapper"] is weird_mapper


def test_none_mapper_is_forwarded():
    record = {}
    mgr = _make_manager(record, mapper=None, set_mapper_attr=True)

    mgr._load_adapter(_in_memory_request())

    assert record["weights_mapper"] is None


def test_model_without_mapper_attribute_is_supported():
    record = {}
    mgr = _make_manager(record, mapper=None, set_mapper_attr=False)

    mgr._load_adapter(_in_memory_request())

    assert record["weights_mapper"] is None
