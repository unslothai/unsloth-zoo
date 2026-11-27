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

"""A GGUF repo with no tokenizer files must give the CPU backend a working tokenizer.

Such repos (unsloth/Qwen3-0.6B-GGUF) ship config.json next to the .gguf files, whose metadata holds the tokenizer.
Loaded without `gguf_file`, transformers 5 builds an empty tokenizer from config.json (vocab 1, so every forward fails
on 0 tokens) and 4.57 raises. The CPU loader now reads the tokenizer from the GGUF when the repo has none.
"""

import json
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("gguf")


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def write_tiny_gguf_repo(folder, arch = "qwen3", name = "tiny.gguf"):
    """config.json + a float32 GGUF whose metadata holds a byte-level BPE tokenizer: a GGUF repo with no tokenizer
    files. Returns the vocabulary size."""
    gguf = pytest.importorskip("gguf")
    import numpy as np

    hidden, inter, heads, kv = 16, 32, 2, 1
    # transformers reads no qwen3 key_length from a GGUF, so its head_dim stays at the default 128.
    head_dim = 128 if arch == "qwen3" else hidden // heads
    tokens = _bytes_to_unicode() + ["Ġt", "he", "<|endoftext|>"]
    vocab = len(tokens)
    os.makedirs(folder, exist_ok = True)
    w = gguf.GGUFWriter(os.path.join(folder, name), arch)
    w.add_context_length(64)
    w.add_embedding_length(hidden)
    w.add_block_count(1)
    w.add_feed_forward_length(inter)
    w.add_head_count(heads)
    w.add_head_count_kv(kv)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_dimension_count(head_dim)
    w.add_rope_freq_base(10000.0)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2" if arch == "qwen3" else "default")
    w.add_token_list(tokens)
    w.add_token_types([1] * (vocab - 1) + [3])
    w.add_token_merges(["Ġ t", "h e"])
    w.add_eos_token_id(vocab - 1)
    w.add_bos_token_id(vocab - 1)
    rng = np.random.default_rng(0)

    def add(tensor, *shape):
        w.add_tensor(tensor, (rng.standard_normal(shape) * 0.02).astype(np.float32))

    add("token_embd.weight", vocab, hidden)
    add("output.weight", vocab, hidden)
    for norm in ("output_norm", "blk.0.attn_norm", "blk.0.ffn_norm"):
        w.add_tensor(f"{norm}.weight", np.ones(hidden, dtype = np.float32))
    if arch == "qwen3":
        w.add_tensor("blk.0.attn_q_norm.weight", np.ones(head_dim, dtype = np.float32))
        w.add_tensor("blk.0.attn_k_norm.weight", np.ones(head_dim, dtype = np.float32))
    add("blk.0.attn_q.weight", heads * head_dim, hidden)
    add("blk.0.attn_k.weight", kv * head_dim, hidden)
    add("blk.0.attn_v.weight", kv * head_dim, hidden)
    add("blk.0.attn_output.weight", hidden, heads * head_dim)
    add("blk.0.ffn_gate.weight", inter, hidden)
    add("blk.0.ffn_up.weight", inter, hidden)
    add("blk.0.ffn_down.weight", hidden, inter)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    config = {
        "architectures": ["Qwen3ForCausalLM" if arch == "qwen3" else "LlamaForCausalLM"],
        "model_type": arch,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "head_dim": head_dim,
        "num_hidden_layers": 1,
        "vocab_size": vocab,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "eos_token_id": vocab - 1,
        "bos_token_id": vocab - 1,
        "torch_dtype": "float32",
    }
    with open(os.path.join(folder, "config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f)
    return vocab


def _as_hub_cache(folder, cache_dir, repo_id = "unsloth-test/tiny-GGUF"):
    """The folder as a cached Hub snapshot of `repo_id`, so a repo-id load resolves offline."""
    repo = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo / "snapshots" / ("0" * 40)
    snapshot.mkdir(parents = True)
    for f in Path(folder).iterdir():
        (snapshot / f.name).write_bytes(f.read_bytes())
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("0" * 40)
    return repo_id



def _load(name, **kwargs):
    from unsloth_zoo.cpu.loader import FastCPUModel

    return FastCPUModel.from_pretrained(name, gguf_file = "tiny.gguf", max_seq_length = 64, load_in_4bit = False,
                                        dtype = torch.float32, **kwargs)


def _forward(model, tokenizer):
    ids = tokenizer("hello there, the test", return_tensors = "pt").input_ids
    assert ids.shape[1] > 0
    with torch.no_grad():
        loss = model(input_ids = ids, labels = ids).loss
    assert torch.isfinite(loss)


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_gguf_only_folder_gets_the_gguf_tokenizer(tmp_path, capsys, arch):
    vocab = write_tiny_gguf_repo(tmp_path, arch)
    model, tokenizer = _load(str(tmp_path))
    assert "the tokenizer is read from tiny.gguf" in capsys.readouterr().out
    assert len(tokenizer) >= vocab - 1
    assert tokenizer.pad_token is not None
    _forward(model, tokenizer)


@pytest.fixture
def hub_offline(monkeypatch):
    """Offline whatever ran earlier in the worker: transformers 4.x looks up adapter_config.json with only the token,
    reads the flag once at import, and huggingface_hub < 1.0 enforces it only in sessions built after it is set. The
    online session made first is the order that sent this made-up repo id to the Hub (401 / 404)."""
    import huggingface_hub.constants

    try:
        from huggingface_hub.utils import get_session, reset_sessions
    except ImportError:  # huggingface_hub >= 1.0 reads the flag per request
        get_session = reset_sessions = None
    if get_session is not None:
        get_session()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    import transformers.utils.hub as transformers_hub

    if hasattr(transformers_hub, "_is_offline_mode"):
        monkeypatch.setattr(transformers_hub, "_is_offline_mode", True)
    if reset_sessions is not None:
        reset_sessions()
    yield
    if reset_sessions is not None:
        reset_sessions()  # the next session is built once the flag is back


def test_gguf_only_repo_id_gets_the_gguf_tokenizer(tmp_path, monkeypatch, hub_offline):
    import huggingface_hub.constants

    vocab = write_tiny_gguf_repo(tmp_path / "src", "qwen3")
    cache = tmp_path / "hub"
    repo_id = _as_hub_cache(tmp_path / "src", cache)
    # transformers reads the GGUF tokenizer from the default cache, whatever cache_dir says.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    model, tokenizer = _load(repo_id, cache_dir = str(cache), local_files_only = True)
    assert len(tokenizer) >= vocab - 1
    _forward(model, tokenizer)


def test_a_repo_with_tokenizer_files_keeps_its_own(tmp_path, monkeypatch):
    import transformers

    write_tiny_gguf_repo(tmp_path, "qwen3")
    (tmp_path / "tokenizer.json").write_text("{}")
    seen = []

    class _Tokenizer:
        pad_token, pad_token_id, eos_token = "<pad>", 0, "</s>"

        def __len__(self):
            return 259

    def fake(name, **kwargs):
        seen.append(kwargs)
        return _Tokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    _load(str(tmp_path))
    assert seen and all("gguf_file" not in k for k in seen)
