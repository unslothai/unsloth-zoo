"""convert_to_gguf matches whole shard filenames, so a neighbour like `old-<name>-00001-of-00002.gguf` is never adopted."""

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
        "llama_cpp_under_test_shard_validation",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def llama_cpp():
    return _load_llama_cpp_module()


def _write_model(model_dir: Path) -> None:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {"num_hidden_layers": 24},
            }
        ),
        encoding = "utf-8",
    )
    (model_dir / "model-00001-of-00001.safetensors").touch()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.language_model.layers.23.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding = "utf-8",
    )


def _write_splitting_converter(path: Path, shards: int) -> Path:
    """A converter that writes `shards` split files and never the bare --outfile."""
    converter = path / "fake_split_convert.py"
    converter.write_text(
        textwrap.dedent(
            f"""
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--outfile")
            parser.add_argument("--outtype")
            parser.add_argument("--split-max-size")
            parser.add_argument("--no-mtp", action="store_true")
            parser.add_argument("model_dir")
            args = parser.parse_args()
            stem = args.outfile[: -len(".gguf")]
            for index in range(1, {shards} + 1):
                Path(f"{{stem}}-{{index:05d}}-of-{shards:05d}.gguf").write_bytes(b"GGUF")
            """
        ),
        encoding = "utf-8",
    )
    return converter


def _touch_shards(directory: Path, stem: str, total: int) -> None:
    for index in range(1, total + 1):
        (directory / f"{stem}-{index:05d}-of-{total:05d}.gguf").write_bytes(b"OLD")


def _convert(llama_cpp, tmp_path: Path, own_shards: int):
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    return llama_cpp.convert_to_gguf(
        model_name = str(tmp_path / "out" / "model"),
        input_folder = str(model_dir),
        converter_location = str(_write_splitting_converter(tmp_path, own_shards)),
        quantization_type = "bf16",
    )


@pytest.mark.parametrize("neighbour_total", (None, 2, 3))
def test_only_this_exports_shards_are_collected(llama_cpp, tmp_path, neighbour_total):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    if neighbour_total is not None:
        _touch_shards(out_dir, "old-model.BF16", neighbour_total)

    outputs, _ = _convert(llama_cpp, tmp_path, own_shards = 2)

    assert sorted(Path(p).name for p in outputs) == [
        "model.BF16-00001-of-00002.gguf",
        "model.BF16-00002-of-00002.gguf",
    ]


def test_a_neighbours_shards_are_not_adopted_when_the_export_wrote_none(llama_cpp, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _touch_shards(out_dir, "old-model.BF16", 2)

    with pytest.raises(RuntimeError, match = "not created"):
        _convert(llama_cpp, tmp_path, own_shards = 0)
