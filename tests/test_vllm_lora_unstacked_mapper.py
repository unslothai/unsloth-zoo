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

The fix in vllm_lora_worker_manager.py calls WeightsMapper.get_unstacked_mapper()
(present only on vLLM >= 0.25.0) to drop the stacked maps while keeping genuine
renames. These tests drive the real WorkerLoRAManager._load_adapter with light
fakes (no GPU, no real vLLM init) and assert which mapper reaches the loader,
for both the in-memory and local-checkpoint paths and both loader signatures.
"""

import types
import pytest

import unsloth_zoo.vllm_lora_worker_manager as wm


class _StackedMapper:
    """Fake vLLM >= 0.25.0 WeightsMapper exposing get_unstacked_mapper()."""

    def __init__(self):
        self.calls = 0
        self.unstacked = object()  # distinct sentinel returned by the method

    def get_unstacked_mapper(self):
        self.calls += 1
        return self.unstacked


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


@pytest.mark.parametrize("new_signature", [True, False])
def test_unstacked_mapper_used_for_in_memory_tensors(new_signature):
    record = {}
    mapper = _StackedMapper()
    mgr = _make_manager(record, mapper=mapper, new_signature=new_signature)

    mgr._load_adapter(_in_memory_request())

    assert mapper.calls == 1, "get_unstacked_mapper must be called exactly once"
    assert record["weights_mapper"] is mapper.unstacked
    assert record["weights_mapper"] is not mapper
    assert record["weights_mapper"] is not None
    assert record["tensors"] is not None, "should hit the in-memory branch"


def test_unstacked_mapper_used_for_local_checkpoint():
    record = {}
    mapper = _StackedMapper()
    mgr = _make_manager(record, mapper=mapper)

    mgr._load_adapter(_checkpoint_request())

    assert mapper.calls == 1
    assert record["weights_mapper"] is mapper.unstacked
    assert record["tensors"] is None, "should hit the local-checkpoint branch"
    assert record["lora_dir"] == "/tmp/does-not-need-to-exist"


def test_legacy_mapper_without_unstack_is_forwarded_unchanged():
    # vLLM < 0.25.0: mapper has no get_unstacked_mapper -> pass through intact
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
