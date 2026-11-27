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

"""A `gguf_file` load on the CPU backend must give a trainable model.

From transformers 5.17 a GGUF load keeps `GgufHfQuantizer` attached (`is_trainable = False`) unless it is asked to
dequantize, so Trainer refuses LoRA and full finetuning. The CPU loader passes `GgufConfig(dequantize = True)`.
"""

import inspect

import pytest
import torch

try:
    from transformers import GgufConfig
except ImportError:
    GgufConfig = None

needs_gguf_config = pytest.mark.skipif(
    GgufConfig is None or "dequantize" not in inspect.signature(GgufConfig.__init__).parameters,
    reason = "transformers < 5.17 has no GGUF quantizer",
)


class _StubTokenizer:
    pad_token, pad_token_id, eos_token, padding_side = "<pad>", 0, "</s>", "right"

    def __len__(self):
        return 32

    def save_pretrained(self, *a, **k):
        pass


def _write_tiny_llama_gguf(path, gguf):
    import numpy as np

    hidden, inter, vocab, heads = 16, 32, 32, 2
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_context_length(64)
    writer.add_embedding_length(hidden)
    writer.add_block_count(1)
    writer.add_feed_forward_length(inter)
    writer.add_head_count(heads)
    writer.add_head_count_kv(heads)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(hidden // heads)
    writer.add_vocab_size(vocab)
    writer.add_tokenizer_model("llama")
    writer.add_token_list([f"t{i}" for i in range(vocab)])
    writer.add_token_scores([0.0] * vocab)
    writer.add_token_types([1] * vocab)
    rng = np.random.default_rng(0)

    def add(name, *shape):
        writer.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))

    add("token_embd.weight", vocab, hidden)
    add("output.weight", vocab, hidden)
    writer.add_tensor("output_norm.weight", np.ones(hidden, dtype = np.float32))
    writer.add_tensor("blk.0.attn_norm.weight", np.ones(hidden, dtype = np.float32))
    writer.add_tensor("blk.0.ffn_norm.weight", np.ones(hidden, dtype = np.float32))
    for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
        add(f"blk.0.{name}.weight", hidden, hidden)
    add("blk.0.ffn_gate.weight", inter, hidden)
    add("blk.0.ffn_up.weight", inter, hidden)
    add("blk.0.ffn_down.weight", hidden, inter)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@needs_gguf_config
def test_gguf_dequantize_config():
    from unsloth_zoo.cpu.loader import _gguf_dequantize_config

    config = _gguf_dequantize_config()
    assert isinstance(config, GgufConfig) and config.dequantize is True


@needs_gguf_config
@pytest.mark.parametrize("fixed", [False, True])
def test_cpu_gguf_load_is_trainable(monkeypatch, tmp_path, capsys, fixed):
    gguf = pytest.importorskip("gguf")
    import transformers
    from transformers.trainer_utils import validate_quantization_for_training
    from unsloth_zoo.cpu import loader
    from unsloth_zoo.cpu.loader import FastCPUModel

    _write_tiny_llama_gguf(tmp_path / "tiny.gguf", gguf)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _StubTokenizer())
    if not fixed:
        # The loader as it was: no GgufConfig, so the quantizer stays and Trainer refuses.
        monkeypatch.setattr(loader, "_gguf_dequantize_config", lambda: None)
    # load_in_4bit is left at its default (True): a GGUF load must ignore it.
    model, _ = FastCPUModel.from_pretrained(str(tmp_path), gguf_file = "tiny.gguf", dtype = torch.float32)
    assert "load_in_4bit is ignored" in capsys.readouterr().out
    if not fixed:
        assert getattr(model, "hf_quantizer", None) is not None
        with pytest.raises(ValueError, match = "purely quantized|do not support training"):
            validate_quantization_for_training(model)
        return
    assert getattr(model, "hf_quantizer", None) is None
    validate_quantization_for_training(model)
    assert all(p.is_floating_point() for p in model.parameters())
    ids = torch.randint(0, 32, (1, 8))
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    loss = model(input_ids = ids, labels = ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "q_proj" in n)
