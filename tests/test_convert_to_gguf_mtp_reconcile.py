from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest


def _load_llama_cpp_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "unsloth_zoo" / "llama_cpp.py"
    spec = importlib.util.spec_from_file_location(
        "llama_cpp_under_test_mtp_reconcile",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def llama_cpp():
    return _load_llama_cpp_module()


def _write_index(model_dir: Path, tensor_name: str) -> None:
    (model_dir / "model-00001-of-00001.safetensors").touch()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    tensor_name: "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding = "utf-8",
    )


@pytest.mark.parametrize(
    "tensor_name",
    (
        "model.layers.24.eh_proj.weight",
        "model.language_model.layers.24.eh_proj.weight",
        "language_model.mtp.fc.weight",
        "model.language_model.mtp.layers.0.mlp.down_proj.weight",
    ),
)
def test_has_mtp_weight_tensors_normalizes_converter_prefixes(llama_cpp, tmp_path, tensor_name):
    _write_index(tmp_path, tensor_name)

    assert llama_cpp._has_mtp_weight_tensors(tmp_path, 24) is True


def test_has_mtp_weight_tensors_reads_single_pytorch_checkpoint(llama_cpp, tmp_path):
    torch = pytest.importorskip("torch")
    torch.save(
        {"model.layers.24.eh_proj.weight": torch.ones(1)},
        tmp_path / "pytorch_model.bin",
    )

    assert llama_cpp._has_mtp_weight_tensors(tmp_path, 24) is True


def test_has_mtp_weight_tensors_reads_each_unindexed_safetensors_part(llama_cpp, tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safetensors_torch.save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.ones(1)},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    safetensors_torch.save_file(
        {"model.language_model.layers.24.eh_proj.weight": torch.ones(1)},
        tmp_path / "model-00002-of-00002.safetensors",
    )

    assert llama_cpp._has_mtp_weight_tensors(tmp_path, 24) is True


def test_has_mtp_weight_tensors_matches_converter_index_precedence(llama_cpp, tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    safetensors_torch.save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.ones(1)},
        tmp_path / "model.safetensors",
    )
    _write_index(tmp_path, "model.layers.24.eh_proj.weight")

    assert llama_cpp._has_mtp_weight_tensors(tmp_path, 24) is True


def _write_converter(path: Path) -> Path:
    converter = path / "fake_convert.py"
    converter.write_text(
        textwrap.dedent(
            """
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--outfile")
            parser.add_argument("--outtype")
            parser.add_argument("--split-max-size")
            parser.add_argument("model_dir")
            args = parser.parse_args()
            Path(args.outfile).write_bytes(b"GGUF")
            """
        ),
        encoding = "utf-8",
    )
    return converter


@pytest.mark.parametrize("has_mtp", (False, True))
def test_convert_to_gguf_reconciles_mtp_config_to_index(llama_cpp, tmp_path, has_mtp):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "mtp_num_hidden_layers": 1,
                "unsloth_fixed_mtp": True,
                "text_config": {
                    "num_hidden_layers": 24,
                    "mtp_num_hidden_layers": 1,
                    "unsloth_fixed_mtp": True,
                },
            }
        ),
        encoding = "utf-8",
    )
    tensor_name = (
        "model.language_model.layers.24.eh_proj.weight"
        if has_mtp
        else "model.language_model.layers.23.self_attn.q_proj.weight"
    )
    _write_index(model_dir, tensor_name)

    llama_cpp.convert_to_gguf(
        model_name = str(tmp_path / "output.gguf"),
        input_folder = str(model_dir),
        converter_location = str(_write_converter(tmp_path)),
        quantization_type = "bf16",
    )

    updated = json.loads(config_path.read_text(encoding = "utf-8"))
    assert "unsloth_fixed_mtp" not in updated
    assert "unsloth_fixed_mtp" not in updated["text_config"]
    assert ("mtp_num_hidden_layers" in updated) is has_mtp
    assert ("mtp_num_hidden_layers" in updated["text_config"]) is has_mtp


def test_convert_to_gguf_always_removes_null_internal_marker(llama_cpp, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "unsloth_fixed_mtp": None,
            }
        ),
        encoding = "utf-8",
    )

    llama_cpp.convert_to_gguf(
        model_name = str(tmp_path / "output.gguf"),
        input_folder = str(model_dir),
        converter_location = str(_write_converter(tmp_path)),
        quantization_type = "bf16",
    )

    updated = json.loads(config_path.read_text(encoding = "utf-8"))
    assert "unsloth_fixed_mtp" not in updated


def test_convert_to_gguf_does_not_rewrite_config_after_index_error(llama_cpp, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "mtp_num_hidden_layers": 1,
        "unsloth_fixed_mtp": True,
        "num_hidden_layers": 24,
    }
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding = "utf-8")
    (model_dir / "model.safetensors").touch()
    (model_dir / "model.safetensors.index.json").write_text("{", encoding = "utf-8")

    with pytest.raises(RuntimeError, match="config.json.*was not changed"):
        llama_cpp.convert_to_gguf(
            model_name = str(tmp_path / "output.gguf"),
            input_folder = str(model_dir),
            converter_location = str(tmp_path / "unused.py"),
        )

    assert json.loads(config_path.read_text(encoding = "utf-8")) == config


def test_convert_to_gguf_rejects_malformed_layer_count_before_rewrite(llama_cpp, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "mtp_num_hidden_layers": 1,
        "unsloth_fixed_mtp": True,
        "num_hidden_layers": "24",
    }
    config_path = model_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding = "utf-8")

    with pytest.raises(ValueError, match="positive integer.*config.json.*was not changed"):
        llama_cpp.convert_to_gguf(
            model_name = str(tmp_path / "output.gguf"),
            input_folder = str(model_dir),
            converter_location = str(tmp_path / "unused.py"),
        )

    assert json.loads(config_path.read_text(encoding = "utf-8")) == config
