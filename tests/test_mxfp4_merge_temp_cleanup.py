"""mxfp4 merge spools tensors to mmapped .pt files; Windows cannot delete a mapped file, so they
leaked into TEMP. Simulates that rule; no .pt may survive and the merged bytes must not change."""
import hashlib
import os
import tempfile
import weakref
from collections import defaultdict

import pytest
import torch


def _make_shard(save_dir, fname):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    kb = "model.layers.0.mlp.experts.gate_up_proj"
    save_file(
        {
            kb + "_blocks": torch.randint(0, 256, (2, 6, 4, 16), dtype = torch.uint8),
            kb + "_scales": torch.randint(120, 135, (2, 6, 4), dtype = torch.uint8),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype = torch.bfloat16),
            "model.norm.weight": torch.randn(8, dtype = torch.bfloat16),
        },
        os.path.join(save_dir, fname),
        metadata = {"format": "pt"},
    )


def _merge(save_dir, fname):
    import unsloth_zoo.saving_utils as sv

    sv._merge_and_overwrite_lora_mxfp4(
        save_directory = save_dir,
        filename = fname,
        lora_weights = defaultdict(lambda: None),
        output_dtype = torch.bfloat16,
        model_class_name = "GptOssForCausalLM",
    )
    with open(os.path.join(save_dir, fname), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _spooled(spool_dir):
    return sorted(n for n in os.listdir(spool_dir) if n.endswith(".pt"))


def test_mxfp4_merge_removes_spooled_files_when_mapped(tmp_path, monkeypatch):
    pytest.importorskip("transformers.integrations.mxfp4")
    fname = "model-00001-of-00001.safetensors"

    posix_dir = tmp_path / "posix"
    posix_dir.mkdir()
    _make_shard(str(posix_dir), fname)
    posix_spool = tmp_path / "posix_spool"
    posix_spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(posix_spool))
    expected = _merge(str(posix_dir), fname)
    assert _spooled(str(posix_spool)) == []

    # Windows rule: a file stays undeletable while a tensor mapped from it is alive.
    win_dir = tmp_path / "windows"
    win_dir.mkdir()
    _make_shard(str(win_dir), fname)
    win_spool = tmp_path / "windows_spool"
    win_spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(win_spool))

    mapped = defaultdict(list)
    real_load = torch.load
    real_remove = os.remove

    def load(f, *args, **kwargs):
        out = real_load(f, *args, **kwargs)
        if kwargs.get("mmap") and isinstance(f, str):
            mapped[os.path.abspath(f)].append(weakref.ref(out))
        return out

    def remove(path, *args, **kwargs):
        if any(ref() is not None for ref in mapped.get(os.path.abspath(path), ())):
            raise PermissionError(13, "The process cannot access the file", path)
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(os, "remove", remove)
    got = _merge(str(win_dir), fname)
    monkeypatch.undo()

    assert mapped, "the merge did not spool through torch.load(mmap = True)"
    assert _spooled(str(win_spool)) == [], "spooled .pt files leaked"
    assert got == expected, "merged shard bytes changed"
