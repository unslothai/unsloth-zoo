# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""merged_16bit export of compressed-tensors INT4 / INT8 ``pack-quantized`` checkpoints (W4A16 / W8A16)."""

import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

pytest.importorskip("compressed_tensors")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from unsloth_zoo import saving_utils  # noqa: E402
from unsloth_zoo.saving_utils import (  # noqa: E402
    _compressed_int_disk_view, _compressed_int_pack_schemes, _disk_module_shapes,
    _plan_compressed_int_rewrite, _rewrite_compressed_int_shard, merge_and_overwrite_lora,
)

PACKED = ("weight_packed", "weight_scale", "weight_shape", "weight_zero_point", "weight_g_idx")
VARIANTS = {
    "sym_g32": dict(num_bits = 4, strategy = "group", group_size = 32, symmetric = True),
    "asym_g64": dict(num_bits = 4, strategy = "group", group_size = 64, symmetric = False),
    "actorder_g32": dict(num_bits = 4, strategy = "group", group_size = 32, symmetric = True, actorder = "group"),
    "int8_channel": dict(num_bits = 8, strategy = "channel", group_size = None, symmetric = True),
    "asym_channel": dict(num_bits = 4, strategy = "channel", group_size = None, symmetric = False),
}


def _group(weights, targets = ("Linear",)):
    weights = dict(dict(type = "int", dynamic = False, observer = "minmax", block_structure = None), **weights)
    return {"targets": list(targets), "weights": weights, "input_activations": None, "output_activations": None}


def _quant(*groups, ignore = ("lm_head",)):
    return {
        "quant_method": "compressed-tensors", "format": "pack-quantized", "quantization_status": "compressed",
        "ignore": list(ignore), "config_groups": {f"group_{i}": g for i, g in enumerate(groups)},
    }


def _scheme(group):
    from compressed_tensors.quantization import QuantizationScheme

    weights = dict(group["weights"])
    if weights.get("actorder") == "group":
        try:
            return QuantizationScheme.model_validate(dict(group, weights = weights))
        except Exception:
            weights["actorder"] = "weight"  # removed spelling on newer compressed-tensors
    return QuantizationScheme.model_validate(dict(group, weights = weights))


def _pack(weight, group, seed = 0):
    """compressed-tensors' own quantize + pack of one Linear weight."""
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization.utils import calculate_qparams

    scheme = _scheme(group)
    args = scheme.weights
    out_f, in_f = weight.shape
    wf = weight.float()
    g_idx = None
    if args.strategy == "channel":
        grouped = wf[:, None, :]
    else:
        gs = args.group_size
        if group["weights"].get("actorder"):
            g_idx = (torch.randperm(in_f, generator = torch.Generator().manual_seed(seed)) // gs).to(torch.int32)
            grouped = wf[:, torch.argsort(g_idx)].reshape(out_f, in_f // gs, gs)
        else:
            grouped = wf.reshape(out_f, in_f // gs, gs)
    scale, zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
    state = {"weight": wf, "weight_scale": scale.to(torch.bfloat16)}
    if not args.symmetric:
        state["weight_zero_point"] = zp.to(torch.int8)
    if g_idx is not None:
        state["weight_g_idx"] = g_idx
    return BaseCompressor.get_value_from_registry("pack-quantized").compress(state, scheme)


def _decompress(packed, group):
    from compressed_tensors.compressors import BaseCompressor

    return BaseCompressor.get_value_from_registry("pack-quantized").decompress(dict(packed), _scheme(group))["weight"]


def _tiny_llama():
    config = transformers.LlamaConfig(
        hidden_size = 128, intermediate_size = 256, num_hidden_layers = 2, num_attention_heads = 4,
        num_key_value_heads = 2, vocab_size = 96, max_position_embeddings = 64, tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    return config, transformers.LlamaForCausalLM(config).to(torch.bfloat16)


def _write_packed_llama(root, pick_group, shards = 1, quant = None):
    """Packs every Linear but lm_head; returns (packed_dir, config, {module: decoded weight})."""
    config, model = _tiny_llama()
    linears = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != "lm_head"}
    tensors, decoded, groups = {}, {}, []
    for key, value in model.state_dict().items():
        module = key[: -len(".weight")]
        if module not in linears:
            tensors[key] = value.contiguous()
            continue
        group = pick_group(module)
        if group not in groups:
            groups.append(group)
        packed = _pack(value, group, seed = len(decoded))
        decoded[module] = _decompress(packed, group).to(torch.bfloat16)
        tensors.update({module + "." + k: v.contiguous() for k, v in packed.items()})
    os.makedirs(root, exist_ok = True)
    keys = sorted(tensors)
    names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)] if shards > 1 else ["model.safetensors"]
    weight_map = {}
    for i, name in enumerate(names):
        part = {k: tensors[k] for k in keys[i::shards]}
        save_file(part, os.path.join(root, name), metadata = {"format": "pt"})
        weight_map.update({k: name for k in part})
    if shards > 1:
        with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
    cfg = config.to_dict()
    cfg["quantization_config"] = quant or _quant(*groups)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(cfg, f)
    return root, config, decoded


def _read_dir(path):
    out = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(path, name), framework = "pt", device = "cpu") as f:
                out.update({k: f.get_tensor(k) for k in f.keys()})
    return out


def _peft_on_decoded(config, packed_dir, decoded, dtype):
    """The in-memory model a loader builds from the packed checkpoint, with a seeded LoRA."""
    from peft import LoraConfig, get_peft_model

    model = transformers.LlamaForCausalLM(config)
    state = _read_dir(packed_dir)
    state = {k: v for k, v in state.items() if not k.rpartition(".")[2] in PACKED}
    state.update({m + ".weight": w for m, w in decoded.items()})
    model.load_state_dict(state, strict = True)
    model = model.to(dtype)
    model.config._name_or_path = packed_dir
    torch.manual_seed(1)
    peft = get_peft_model(model, LoraConfig(
        r = 4, lora_alpha = 8, lora_dropout = 0.0, target_modules = ["q_proj", "v_proj", "up_proj", "down_proj"],
    ))
    with torch.no_grad():
        for name, param in peft.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    return peft.eval()


def _merge(peft, packed_dir, out_dir, dtype):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    return merge_and_overwrite_lora(
        get_model_name = lambda *a, **k: packed_dir, model = peft, tokenizer = None,
        save_directory = out_dir, save_method = "merged_16bit", output_dtype = dtype, push_to_hub = False,
    )


def _lora_of(peft):
    out = {}
    for name, module in peft.named_modules():
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
            key = name.removeprefix("base_model.model.")
            out[key] = (module.lora_A["default"].weight.detach(), module.lora_B["default"].weight.detach(),
                        module.scaling["default"])
    return out


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_merged_export_is_decompress_plus_lora_delta(variant, tmp_path, monkeypatch):
    monkeypatch.setattr(saving_utils, "_active_merge_device", lambda: "cpu")
    group = _group(VARIANTS[variant])
    packed_dir, config, decoded = _write_packed_llama(str(tmp_path / "packed"), lambda m: group)
    peft = _peft_on_decoded(config, packed_dir, decoded, torch.bfloat16)
    loras = _lora_of(peft)
    base = {k: v for k, v in _read_dir(packed_dir).items() if k.rpartition(".")[2] not in PACKED}
    out = str(tmp_path / "merged")
    _merge(peft, packed_dir, out, torch.bfloat16)

    got = _read_dir(out)
    assert not any(k.rpartition(".")[2] in PACKED for k in got)
    assert "quantization_config" not in json.load(open(os.path.join(out, "config.json")))
    assert set(got) == set(base) | {m + ".weight" for m in decoded}
    n_merged = 0
    for module, weight in decoded.items():
        if module in loras:
            A, B, scaling = loras[module]
            want = weight.float().addmm_(B.float(), A.float(), alpha = scaling).to(torch.bfloat16)
            n_merged += 1
        else:
            want = weight
        assert torch.equal(got[module + ".weight"], want), module
    for key, value in base.items():
        assert torch.equal(got[key], value), key
    assert n_merged == 2 * 4


@pytest.mark.parametrize("variant", ["asym_g64", "actorder_g32", "int8_channel"])
def test_merged_export_reloads_with_plain_transformers(variant, tmp_path):
    group = _group(VARIANTS[variant])
    packed_dir, config, decoded = _write_packed_llama(str(tmp_path / "packed"), lambda m: group, shards = 2)
    peft = _peft_on_decoded(config, packed_dir, decoded, torch.float32)
    ids = torch.randint(0, config.vocab_size, (2, 12), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = peft(input_ids = ids).logits
    out = str(tmp_path / "merged")
    _merge(peft, packed_dir, out, torch.float32)
    index = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    assert not any(k.rpartition(".")[2] in PACKED for k in index)
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(out, dtype = torch.float32)
    with torch.no_grad():
        got = reloaded(input_ids = ids).logits
    torch.testing.assert_close(got, want, atol = 1e-4, rtol = 1e-4)
    with torch.no_grad(), peft.disable_adapter():
        assert (peft(input_ids = ids).logits - want).abs().max() > 1e-2  # the LoRA moves the logits


def test_mixed_int4_int8_groups_decode_each_tensor_with_its_own_group(tmp_path):
    int4 = _group(VARIANTS["asym_g64"], targets = ["re:.*self_attn.*"])
    int8 = _group(VARIANTS["int8_channel"], targets = ["re:.*mlp.*"])
    packed_dir, _, decoded = _write_packed_llama(
        str(tmp_path), lambda m: int4 if "self_attn" in m else int8, shards = 2,
    )
    filenames = sorted(f for f in os.listdir(packed_dir) if f.endswith(".safetensors"))
    schemes = _compressed_int_pack_schemes(packed_dir)
    plan = _plan_compressed_int_rewrite(packed_dir, filenames, schemes)
    assert {plan["bases"][m]["num_bits"] for m in decoded} == {4, 8}
    for m in decoded:
        assert plan["bases"][m]["num_bits"] == (4 if "self_attn" in m else 8)
    view = _compressed_int_disk_view(packed_dir, filenames, plan)
    for name in filenames:
        _rewrite_compressed_int_shard(packed_dir, name, plan)
    assert view == _disk_module_shapes(packed_dir, filenames)
    got = _read_dir(packed_dir)
    for m, weight in decoded.items():
        assert torch.equal(got[m + ".weight"], weight), m


def test_companions_in_another_shard_are_read_before_any_rewrite(tmp_path):
    group = _group(VARIANTS["asym_g64"])
    packed_dir, _, decoded = _write_packed_llama(str(tmp_path), lambda m: group)
    tensors = _read_dir(packed_dir)
    moved = "model.layers.0.self_attn.q_proj."
    other = {k: tensors.pop(k) for k in list(tensors) if k.startswith(moved) and k != moved + "weight_packed"}
    os.remove(os.path.join(packed_dir, "model.safetensors"))
    save_file(tensors, os.path.join(packed_dir, "a.safetensors"))
    save_file(other, os.path.join(packed_dir, "b.safetensors"))
    plan = _plan_compressed_int_rewrite(packed_dir, ["a.safetensors", "b.safetensors"], _compressed_int_pack_schemes(packed_dir))
    _rewrite_compressed_int_shard(packed_dir, "a.safetensors", plan)
    _rewrite_compressed_int_shard(packed_dir, "b.safetensors", plan)
    got = _read_dir(packed_dir)
    assert torch.equal(got[moved + "weight"], decoded[moved[:-1]])
    assert not any(k.rpartition(".")[2] in PACKED for k in got)


@pytest.mark.parametrize("change, match", [
    (lambda q: q["config_groups"]["group_0"].update(input_activations = {"num_bits": 8, "type": "int"}), "quantizes activations"),
    (lambda q: q["config_groups"]["group_0"]["weights"].update(type = "float"), "only int 4 / 8-bit"),
    (lambda q: q["config_groups"]["group_0"]["weights"].update(num_bits = 2), "only int 4 / 8-bit"),
    (lambda q: q["config_groups"]["group_0"]["weights"].update(strategy = "block", block_structure = [128, 128]), "only int 4 / 8-bit"),
    (lambda q: q.update(kv_cache_scheme = {"num_bits": 8, "type": "float"}), "KV cache"),
    (lambda q: q.update(sparsity_config = {"format": "sparse-24-bitmask"}), "sparsity-compressed"),
])
def test_unsupported_int_schemes_refuse_before_anything_is_written(tmp_path, change, match):
    quant = _quant(_group(VARIANTS["sym_g32"]))
    change(quant)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant}))
    with pytest.raises(RuntimeError, match = match):
        _compressed_int_pack_schemes(str(tmp_path))


def test_a_tensor_no_group_describes_refuses(tmp_path):
    group = _group(VARIANTS["sym_g32"])
    packed_dir, _, _ = _write_packed_llama(str(tmp_path), lambda m: group)
    with open(os.path.join(packed_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"]["config_groups"]["group_0"]["weights"]["num_bits"] = 8
    with open(os.path.join(packed_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    with pytest.raises(RuntimeError, match = "matches none"):
        _plan_compressed_int_rewrite(packed_dir, ["model.safetensors"], _compressed_int_pack_schemes(packed_dir))


def test_nvfp4_and_mixed_formats_still_refuse(tmp_path):
    group = _group(VARIANTS["sym_g32"])
    for fmt in ("nvfp4-pack-quantized", "float-quantized"):
        quant = _quant(group)
        quant["config_groups"]["group_1"] = dict(group, format = fmt)
        packed_dir, config, decoded = _write_packed_llama(str(tmp_path / fmt), lambda m: group, quant = quant)
        peft = _peft_on_decoded(config, packed_dir, decoded, torch.bfloat16)
        out = tmp_path / (fmt + "_out")
        with pytest.raises(RuntimeError, match = "not supported"):
            _merge(peft, packed_dir, str(out), torch.bfloat16)
        assert not out.exists() or not any(out.iterdir())


def test_the_int_decode_runs_after_the_completeness_check_and_reads_config_first():
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(inspect.unwrap(saving_utils.merge_and_overwrite_lora)))
    calls = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node.lineno)
    assert max(calls["_check_lora_merge_is_complete"]) < min(calls["_rewrite_compressed_int_shard"])
    assert max(calls["_compressed_int_pack_schemes"]) < min(calls["_remove_quantization_config"])
