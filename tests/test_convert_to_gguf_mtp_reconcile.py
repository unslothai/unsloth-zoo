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


def _write_converter(path: Path, *, supports_no_mtp: bool = True) -> Path:
    converter = path / "fake_convert.py"
    command_log = path / "converter_commands.jsonl"
    converter.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import json
            import sys
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--outfile")
            parser.add_argument("--outtype")
            parser.add_argument("--split-max-size")
            {('parser.add_argument("--no-mtp", action="store_true")' if supports_no_mtp else '')}
            parser.add_argument("--mmproj", action="store_true")
            parser.add_argument("model_dir")
            args = parser.parse_args()
            with Path({str(command_log)!r}).open("a", encoding="utf-8") as log:
                log.write(json.dumps(sys.argv[1:]) + "\\n")
            Path(args.outfile).write_bytes(b"GGUF")
            """
        ),
        encoding = "utf-8",
    )
    return converter


def _read_converter_commands(path: Path) -> list[list[str]]:
    return [
        json.loads(line)
        for line in (path / "converter_commands.jsonl").read_text(encoding = "utf-8").splitlines()
    ]


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
    # The declaration survives either way: kept as-is with MTP tensors, and kept
    # because `--no-mtp` carries the intent without them.
    assert "mtp_num_hidden_layers" in updated
    assert "mtp_num_hidden_layers" in updated["text_config"]
    command, = _read_converter_commands(tmp_path)
    assert ("--no-mtp" in command) is not has_mtp


def test_convert_to_gguf_disables_missing_mtp_only_for_vlm_text(llama_cpp, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "mtp_num_hidden_layers": 1,
                "text_config": {
                    "num_hidden_layers": 24,
                    "mtp_num_hidden_layers": 1,
                },
            }
        ),
        encoding = "utf-8",
    )
    _write_index(model_dir, "model.language_model.layers.23.self_attn.q_proj.weight")

    llama_cpp.convert_to_gguf(
        model_name = str(tmp_path / "output.gguf"),
        input_folder = str(model_dir),
        converter_location = str(_write_converter(tmp_path)),
        supported_vision_archs = {"Qwen3_5ForConditionalGeneration"},
        quantization_type = "bf16",
        is_vlm = True,
    )

    text_command, mmproj_command = _read_converter_commands(tmp_path)
    assert "--no-mtp" in text_command
    assert "--mmproj" not in text_command
    assert "--no-mtp" not in mmproj_command
    assert "--mmproj" in mmproj_command


def test_convert_to_gguf_omits_no_mtp_for_legacy_converter(llama_cpp, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "mtp_num_hidden_layers": 1,
                "num_hidden_layers": 24,
            }
        ),
        encoding = "utf-8",
    )
    _write_index(model_dir, "model.layers.23.self_attn.q_proj.weight")

    llama_cpp.convert_to_gguf(
        model_name = str(tmp_path / "output.gguf"),
        input_folder = str(model_dir),
        converter_location = str(_write_converter(tmp_path, supports_no_mtp = False)),
        quantization_type = "bf16",
    )

    command, = _read_converter_commands(tmp_path)
    assert "--no-mtp" not in command


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


@pytest.mark.parametrize(
    "location",
    [None, 123, "", "\x00bad"],
    ids = ["none", "not-a-path", "empty", "nul-byte"],
)
def test_converter_support_probe_never_raises_on_an_unusable_location(llama_cpp, location):
    """Anything unreadable is False. Only OSError was caught, so a None
    location raised TypeError out of convert_to_gguf."""
    assert llama_cpp._converter_supports_no_mtp(location) is False


def test_converter_support_probe_reads_a_directory_as_unsupported(llama_cpp, tmp_path):
    assert llama_cpp._converter_supports_no_mtp(str(tmp_path)) is False


def _write_arch_gated_converter(path: Path) -> Path:
    """Declares `--no-mtp` but refuses it for this architecture: llama.cpp
    25558268 added the flag and its allowlist in the same commit."""
    converter = path / "arch_gated_convert.py"
    command_log = path / "converter_commands.jsonl"
    converter.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import json
            import sys
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--outfile")
            parser.add_argument("--outtype")
            parser.add_argument("--split-max-size")
            parser.add_argument("--no-nextn", "--no-mtp", dest="no_mtp", action="store_true")
            parser.add_argument("--mmproj", action="store_true")
            parser.add_argument("model_dir")
            args = parser.parse_args()
            with Path({str(command_log)!r}).open("a", encoding="utf-8") as log:
                log.write(json.dumps(sys.argv[1:]) + "\\n")
            if args.no_mtp:
                sys.stderr.write(
                    "ERROR:hf-to-gguf:--mtp / --no-nextn are not supported "
                    "for LlamaForCausalLM\\n"
                )
                raise SystemExit(1)
            Path(args.outfile).write_bytes(b"GGUF")
            """
        ),
        encoding = "utf-8",
    )
    return converter


def test_convert_to_gguf_drops_no_mtp_when_the_architecture_refuses_it(llama_cpp, tmp_path):
    """`_mtp_declared` reads config.json only, so Qwen3.5 weights republished
    under `Qwen3ForCausalLM` sent `--no-mtp` to a converter that rejects it,
    turning a working export into a failure citing a flag nobody passed."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "mtp_num_hidden_layers": 1,
                "unsloth_fixed_mtp": True,
                "num_hidden_layers": 24,
            }
        ),
        encoding = "utf-8",
    )
    _write_index(model_dir, "model.layers.23.self_attn.q_proj.weight")
    output = tmp_path / "output.gguf"

    llama_cpp.convert_to_gguf(
        model_name = str(output),
        input_folder = str(model_dir),
        converter_location = str(_write_arch_gated_converter(tmp_path)),
        quantization_type = "bf16",
    )

    first, second = _read_converter_commands(tmp_path)
    assert "--no-mtp" in first        # tried, because the option is declared
    assert "--no-mtp" not in second   # retried without it once refused
    assert output.read_bytes() == b"GGUF"


def test_convert_to_gguf_does_not_retry_an_unrelated_converter_failure(llama_cpp, tmp_path):
    """The retry is keyed on the architecture refusal, not on any failure."""
    assert llama_cpp._converter_rejected_no_mtp(
        "ERROR:hf-to-gguf:--mtp / --no-nextn are not supported for LlamaForCausalLM"
    ) is True
    assert llama_cpp._converter_rejected_no_mtp("MemoryError: out of memory") is False
    assert llama_cpp._converter_rejected_no_mtp("INFO: exporting with --no-mtp") is False
    assert llama_cpp._converter_rejected_no_mtp("") is False
    assert llama_cpp._drop_no_mtp(["x", "--no-mtp", "y"]) == ["x", "y"]
    assert llama_cpp._drop_no_mtp(["x", "y"]) is None


def test_convert_to_gguf_stays_idempotent_across_repeated_exports(llama_cpp, tmp_path):
    """A retry, or a second export from the same folder, must behave like the first.

    Deleting `mtp_num_hidden_layers` made the next run read no declaration, omit
    `--no-mtp`, and hit the converter assertion the flag exists to avoid.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "mtp_num_hidden_layers": 1,
                "unsloth_fixed_mtp": True,
                "text_config": {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1},
            }
        ),
        encoding = "utf-8",
    )
    _write_index(model_dir, "model.language_model.layers.23.self_attn.q_proj.weight")
    converter = str(_write_converter(tmp_path))

    for run in range(3):
        llama_cpp.convert_to_gguf(
            model_name = str(tmp_path / f"output{run}.gguf"),
            input_folder = str(model_dir),
            converter_location = converter,
            quantization_type = "bf16",
        )

    # Every run saw the declaration and sent the flag; the internal marker still goes.
    updated = json.loads(config_path.read_text(encoding = "utf-8"))
    assert "unsloth_fixed_mtp" not in updated
    assert updated["mtp_num_hidden_layers"] == 1
    commands = _read_converter_commands(tmp_path)
    assert len(commands) == 3
    assert all("--no-mtp" in command for command in commands)


def test_convert_to_gguf_still_strips_the_declaration_for_legacy_converters(llama_cpp, tmp_path):
    """Without the flag there is nothing to carry the intent, so the key must go."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "mtp_num_hidden_layers": 1,
                "text_config": {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1},
            }
        ),
        encoding = "utf-8",
    )
    _write_index(model_dir, "model.language_model.layers.23.self_attn.q_proj.weight")

    llama_cpp.convert_to_gguf(
        model_name = str(tmp_path / "output.gguf"),
        input_folder = str(model_dir),
        converter_location = str(_write_converter(tmp_path, supports_no_mtp = False)),
        quantization_type = "bf16",
    )

    updated = json.loads(config_path.read_text(encoding = "utf-8"))
    assert "mtp_num_hidden_layers" not in updated
    assert "mtp_num_hidden_layers" not in updated["text_config"]
    command, = _read_converter_commands(tmp_path)
    assert "--no-mtp" not in command
