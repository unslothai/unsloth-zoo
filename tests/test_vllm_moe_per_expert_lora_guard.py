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
"""Per-expert 2D MoE LoRA (`experts.<i>.<proj>`) is servable on a 2D vLLM model; engine lookups are stubbed."""
import types

import pytest

import unsloth_zoo.vllm_utils as vu


PER_EXPERT_KEYS = [
    "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight",
    "base_model.model.model.layers.0.mlp.experts.0.up_proj.lora_B.weight",
    "base_model.model.model.layers.0.mlp.experts.7.down_proj.lora_A.weight",
    "base_model.model.model.layers.1.block_sparse_moe.experts.3.w1.lora_A.weight",
]
STACKED_KEYS = [
    "base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight",
    "base_model.model.model.layers.0.mlp.experts.lora_B.weight",
    "base_model.model.model.layers.0.mlp.experts.gate_up_proj.lora_A.weight",
]
ATTENTION_KEYS = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]


@pytest.fixture
def engine(monkeypatch):
    state = types.SimpleNamespace(is_3d = False, mixed = False, unmatched = [])

    def vllm_model(*a, **k):
        cls = type("FakeVllmMoe", (), {"is_3d_moe_weight": state.is_3d})
        return cls()

    monkeypatch.setattr(vu, "_get_vllm_model_runner", lambda model: object())
    monkeypatch.setattr(vu, "_get_vllm_lora_model", vllm_model)
    monkeypatch.setattr(
        vu,
        "_get_vllm_lora_manager",
        lambda model, runner: types.SimpleNamespace(_enable_mixed_moe_lora_format = state.mixed),
    )
    monkeypatch.setattr(vu, "_unmatched_lora_keys", lambda model, keys: list(state.unmatched))
    monkeypatch.delenv("UNSLOTH_DISABLE_LORA_NAME_CHECK", raising = False)
    return state


def _model(**attrs):
    return types.SimpleNamespace(config = types.SimpleNamespace(), **attrs)


@pytest.mark.parametrize("key", PER_EXPERT_KEYS)
def test_per_expert_keys_are_classified(key):
    assert vu._is_moe_expert_lora_key(key)
    assert vu._is_moe_per_expert_lora_key(key)


@pytest.mark.parametrize("key", STACKED_KEYS + ATTENTION_KEYS)
def test_stacked_and_dense_keys_are_not_per_expert(key):
    assert not vu._is_moe_per_expert_lora_key(key)


def test_per_expert_adapter_is_served_by_the_2d_wrapper(engine):
    vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS + ATTENTION_KEYS, "the training model", {"r": 16})


def test_per_expert_adapter_on_a_3d_model_is_refused(engine):
    engine.is_3d = True
    with pytest.raises(NotImplementedError, match = "per-expert"):
        vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 16})
    engine.mixed = True
    vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 16})


def test_stacked_adapter_still_needs_the_3d_wrapper(engine):
    with pytest.raises(NotImplementedError, match = "is_3d_moe_weight"):
        vu._check_lora_is_servable(_model(), STACKED_KEYS, "the training model", {"r": 16})
    with pytest.raises(NotImplementedError, match = "stacked"):
        vu._check_lora_is_servable(
            _model(), PER_EXPERT_KEYS + STACKED_KEYS, "the training model", {"r": 16}
        )
    engine.is_3d = True
    vu._check_lora_is_servable(_model(), STACKED_KEYS, "the training model", {"r": 16})


def test_per_expert_adapter_keeps_the_rank_and_bitsandbytes_limits(engine):
    with pytest.raises(NotImplementedError, match = "128"):
        vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 256})
    with pytest.raises(NotImplementedError, match = "bitsandbytes"):
        vu._check_lora_is_servable(
            _model(is_loaded_in_4bit = True), PER_EXPERT_KEYS, "the training model", {"r": 16}
        )


def test_per_expert_adapter_still_goes_through_the_name_check(engine):
    engine.unmatched = [(PER_EXPERT_KEYS[0], "model.layers.0.mlp.experts.0.gate_proj")]
    with pytest.raises(RuntimeError, match = "does not have"):
        vu._check_lora_is_servable(_model(), PER_EXPERT_KEYS, "the training model", {"r": 16})


# --- the real name check, unstubbed: only what vLLM itself provides is faked below --------
# With no reachable manager the check must rebuild vLLM's `experts -> experts.<i>.<proj>` expansion.
import sys

import torch.nn as nn


class _LinearBase(nn.Module): pass
class _MoERunner(nn.Module): pass
class _BaseLayerWithLoRA(nn.Module): pass


def _parse_fine_tuned_lora_name(name, weights_mapper = None):
    # vllm/lora/utils.py parse_fine_tuned_lora_name, without the mapper (none is set here).
    start_index = 2 if name.startswith("base_model.model.") else 0
    parts = name.split(".")
    if parts[-1] == "weight" and parts[-2] in ("lora_A", "lora_B"):
        return ".".join(parts[start_index:-2]), parts[-2] == "lora_A"
    raise ValueError(f"{name} is unsupported LoRA weight")


def _get_packed_modules_mapping(model):
    # vllm/model_executor/utils.py: the model's static mapping, no expert expansion.
    return dict(getattr(model, "packed_modules_mapping", None) or {})


def _expert_mapping(names, num_experts, prefix):
    # FusedMoE.make_expert_params_mapping (v0.11.2) / RoutedExperts.build_expert_params_mapping (main)
    gate, down, up = names
    return [
        ("w13_" if w in (gate, up) else "w2_", f"experts.{i}.{w}.{prefix}", i, shard)
        for i in range(num_experts)
        for shard, w in (("w1", gate), ("w2", down), ("w3", up))
    ]


def _process_packed_modules_mapping_v0112(model):
    # vllm/lora/utils.py v0.11.2: get_moe_expert_mapping checks the model, then its children.
    if not any(isinstance(m, _MoERunner) for m in model.modules()):
        return _get_packed_modules_mapping(model)
    get = getattr(model, "get_expert_mapping", None)
    mapping = get() if get is not None else []
    if not mapping:
        raise AttributeError("To support LoRA for MoE model, 'get_expert_mapping' must be implemented")
    packed = _get_packed_modules_mapping(model)
    packed["experts"] = [w.rstrip(".") for _, w, _, _ in mapping]
    return packed


def _process_packed_modules_mapping_main(model, force_2d_moe = False, enable_moe_shared_loras = False):
    # vllm/lora/utils.py main: first named module with get_expert_mapping (RoutedExperts).
    packed = _get_packed_modules_mapping(model)
    if not any(isinstance(m, _MoERunner) for m in model.modules()): return packed
    if (not model.is_3d_moe_weight) or force_2d_moe:
        for _, module in model.named_modules():
            get = getattr(module, "get_expert_mapping", None)
            if get is not None: break
        packed["experts"] = [w.rstrip(".") for _, w, _, _ in get()]
    return packed


class _Experts(_MoERunner):
    def __init__(self, names, num_experts, own_mapping):
        super().__init__()
        self.names, self.num_experts = names, num_experts
        self.lora_base_layer_prefix = ""
        if own_mapping:  # main: RoutedExperts.get_expert_mapping
            self.get_expert_mapping = lambda: _expert_mapping(
                self.names, self.num_experts, self.lora_base_layer_prefix
            )


class _Wrapped(_BaseLayerWithLoRA):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer


class _Node(nn.Module): pass


ARCHS = {
    # transformers 4.x Qwen3-MoE and Mixtral, as PEFT names the per-expert adapters
    "qwen3_moe": ("mlp", ("gate_proj", "down_proj", "up_proj")),
    "mixtral": ("block_sparse_moe", ("w1", "w2", "w3")),
}
NUM_EXPERTS, NUM_LAYERS = 4, 2


def _per_expert_keys(arch):
    parent, projs = ARCHS[arch]
    keys = []
    for layer in range(NUM_LAYERS):
        keys.append(f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight")
        keys.append(f"base_model.model.model.layers.{layer}.self_attn.o_proj.lora_B.weight")
        for expert in range(NUM_EXPERTS):
            for proj in projs:
                for ab in ("lora_A", "lora_B"):
                    keys.append(
                        f"base_model.model.model.layers.{layer}.{parent}.experts.{expert}.{proj}.{ab}.weight"
                    )
    return keys


def _build_engine(arch, flavor):
    """A 2D MoE vLLM model plus the LoRAModelManager vLLM would build for it."""
    parent, projs = ARCHS[arch]
    on_main = flavor == "main"

    class _VllmMoe(nn.Module):
        is_3d_moe_weight = False
        packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    root = _VllmMoe()
    if not on_main:  # v0.11.2: get_expert_mapping lives on the CausalLM
        root.get_expert_mapping = lambda: _expert_mapping(projs, NUM_EXPERTS, "")
    root.add_module("model", _Node())
    root.model.add_module("layers", nn.ModuleList())
    experts = []
    for _ in range(NUM_LAYERS):
        layer = _Node()
        layer.add_module("self_attn", _Node())
        layer.self_attn.add_module("qkv_proj", _LinearBase())
        layer.self_attn.add_module("o_proj", _LinearBase())
        layer.add_module(parent, _Node())
        experts.append(_Experts(projs, NUM_EXPERTS, own_mapping = on_main))
        getattr(layer, parent).add_module("experts", experts[-1])
        root.model.layers.append(layer)

    # LoRAModelManager.__init__: packed mapping first, then _create_lora_modules wraps.
    process = _process_packed_modules_mapping_main if on_main else _process_packed_modules_mapping_v0112
    packed = process(root)
    modules = {}
    for name, module in list(root.named_modules()):
        if not isinstance(module, (_LinearBase, _MoERunner)): continue
        owner_name, _, leaf = name.rpartition(".")
        owner = root.get_submodule(owner_name)
        owner._modules[leaf] = _Wrapped(module)
        modules[name] = owner._modules[leaf]
    if on_main:  # FusedMoEWithLoRA.__init__ sets this on the wrapped RoutedExperts
        for e in experts: e.lora_base_layer_prefix = "base_layer."
    manager = types.SimpleNamespace(
        modules = modules, packed_modules_mapping = packed, _enable_mixed_moe_lora_format = False,
    )
    return root, manager, process


@pytest.fixture
def fake_vllm(monkeypatch):
    def install(process):
        mods = {
            "vllm": {},
            "vllm.lora": {},
            "vllm.lora.utils": {
                "parse_fine_tuned_lora_name": _parse_fine_tuned_lora_name,
                "process_packed_modules_mapping": process,
            },
            "vllm.lora.layers": {},
            "vllm.lora.layers.base": {"BaseLayerWithLoRA": _BaseLayerWithLoRA},
            "vllm.model_executor": {},
            "vllm.model_executor.utils": {"get_packed_modules_mapping": _get_packed_modules_mapping},
            "vllm.model_executor.layers": {},
            "vllm.model_executor.layers.linear": {"LinearBase": _LinearBase},
            "vllm.model_executor.layers.fused_moe": {"MoERunner": _MoERunner},
        }
        for name, attrs in mods.items():
            module = types.ModuleType(name)
            module.__path__ = []
            for k, v in attrs.items(): setattr(module, k, v)
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("UNSLOTH_DISABLE_LORA_NAME_CHECK", raising = False)
    return install


def _trainer(monkeypatch, vllm_model, manager, lora_config = None):
    runner = types.SimpleNamespace(model = vllm_model)
    if manager is not None:
        runner.lora_manager = types.SimpleNamespace(_adapter_manager = manager)
    elif lora_config is not None:  # WorkerLoRAManager without a built LoRAModelManager
        runner.lora_manager = types.SimpleNamespace(lora_config = lora_config)
    monkeypatch.setattr(vu, "_get_vllm_model_runner", lambda model: runner)
    return _model()


@pytest.mark.parametrize("with_manager", [True, False], ids = ["manager", "no_manager"])
@pytest.mark.parametrize("flavor", ["v0.11.2", "main"])
@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_per_expert_keys_resolve_through_the_real_name_check(monkeypatch, fake_vllm, arch, flavor, with_manager):
    vllm_model, manager, process = _build_engine(arch, flavor)
    fake_vllm(process)
    model = _trainer(monkeypatch, vllm_model, manager if with_manager else None)
    keys = _per_expert_keys(arch)

    assert vu._unmatched_lora_keys(model, keys) == []
    vu._check_lora_is_servable(model, keys, "the training model", {"r": 16})

    # and the check still bites: an expert index or projection vLLM does not have is caught
    parent, projs = ARCHS[arch]
    bad = [
        f"base_model.model.model.layers.0.{parent}.experts.{NUM_EXPERTS}.{projs[0]}.lora_A.weight",
        f"base_model.model.model.layers.1.{parent}.experts.0.not_a_proj.lora_A.weight",
    ]
    assert sorted(n for _, n in vu._unmatched_lora_keys(model, keys + bad)) == [
        f"model.layers.0.{parent}.experts.{NUM_EXPERTS}.{projs[0]}",
        f"model.layers.1.{parent}.experts.0.not_a_proj",
    ]
    with pytest.raises(RuntimeError, match = "does not have"):
        vu._check_lora_is_servable(model, keys + bad, "the training model", {"r": 16})


# --- no reachable manager on a 3D model in mixed mode ------------------------------------
# The fallback must pass force_2d_moe = enable_mixed_moe_lora_format like LoRAModelManager.
@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_mixed_mode_3d_model_without_manager_expands_per_expert_names(monkeypatch, fake_vllm, arch):
    vllm_model, _, _ = _build_engine(arch, "main")
    type(vllm_model).is_3d_moe_weight = True
    fake_vllm(_process_packed_modules_mapping_main)
    lora_config = types.SimpleNamespace(enable_mixed_moe_lora_format = True, enable_moe_shared_loras = False)
    model = _trainer(monkeypatch, vllm_model, None, lora_config)
    keys = _per_expert_keys(arch)

    assert vu._moe_expert_lora_refusal_reason(model, {"r": 16}, per_expert = True) is None
    assert vu._unmatched_lora_keys(model, keys) == []
    vu._check_lora_is_servable(model, keys, "the training model", {"r": 16})

    # without mixed mode the same 3D model still refuses per-expert keys
    lora_config.enable_mixed_moe_lora_format = False
    with pytest.raises(NotImplementedError, match = "per-expert"):
        vu._check_lora_is_servable(model, keys, "the training model", {"r": 16})


def test_v0112_builder_without_flag_parameters_is_called_with_the_model_only(monkeypatch, fake_vllm):
    vllm_model, _, process = _build_engine("qwen3_moe", "v0.11.2")
    fake_vllm(process)
    lora_config = types.SimpleNamespace(enable_mixed_moe_lora_format = True, enable_moe_shared_loras = True)
    model = _trainer(monkeypatch, vllm_model, None, lora_config)
    assert vu._unmatched_lora_keys(model, _per_expert_keys("qwen3_moe")) == []


# --- path loading: per-expert keys are not lone stacked halves ---------------------------
def test_per_expert_keys_are_not_lone_stacked_expert_halves():
    keys = [k for k in _per_expert_keys("qwen3_moe") if "experts" in k]
    assert vu._moe_lone_expert_lora_prefixes(keys) == []
    # a real lone stacked half is still found
    lone = ["base_model.model.model.layers.0.mlp.experts.lora_A.weight",
            "base_model.model.model.layers.0.mlp.experts.lora_B.weight"]
    assert vu._moe_lone_expert_lora_prefixes(keys + lone) == ["base_model.model.model.layers.0.mlp.experts"]


def test_saved_per_expert_adapter_is_loaded_by_path(monkeypatch, tmp_path):
    keys = _per_expert_keys("mixtral")
    monkeypatch.setattr(vu, "_check_lora_rank_fits", lambda model, save_directory: None)
    monkeypatch.setattr(vu, "_saved_adapter_lora_keys", lambda save_directory: list(keys))
    monkeypatch.setattr(vu, "get_peft_config", lambda save_directory: {"r": 16})
    monkeypatch.setattr(vu, "_remap_moe_expert_lora_keys", lambda model, state_dict: state_dict)
    monkeypatch.setattr(vu, "_check_lora_is_servable", lambda *a, **k: None)
    monkeypatch.setattr(vu, "LORA_REQUEST_ID", 1, raising = False)

    def materialised(save_directory):
        raise AssertionError("the whole checkpoint was loaded in the trainer process")
    monkeypatch.setattr(vu, "_saved_adapter_lora_tensors", materialised)

    class _LoRARequest:
        def __init__(self, name, int_id, path = None, lora_tensors = None, lora_config = None):
            self.path, self.lora_tensors = path, lora_tensors
    for name in ("vllm", "vllm.lora", "vllm.lora.request"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.lora.request"].LoRARequest = _LoRARequest

    request = vu._build_lora_request(_model(), str(tmp_path), False, 1)
    assert request.path == str(tmp_path) and request.lora_tensors is None
