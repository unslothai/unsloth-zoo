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

"""On real Metal: the MLX decision model against a torch reference of laya's DecisionModel, and the Qwen-based readouts on a tiny decoder."""

from __future__ import annotations

import contextlib
import copy
import json
import random
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mlx_simulation import mlx_is_simulated  # noqa: E402

if mlx_is_simulated():
    pytest.skip("needs real MLX: mx.fast attention and RoPE", allow_module_level = True)

from mlx.nn import Dropout, Embedding, LayerNorm, Linear, quantize  # noqa: E402
from mlx.utils import tree_flatten, tree_map  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from unsloth_zoo.mlx.decision import (  # noqa: E402
    ClefNetwork,
    _KevModel,
    _LabelModel,
    _LayerwiseStep,
    _MarkerStep,
    _clef_record_loss,
    _load_joint_head,
    _merge_lora,
    _soft_cross_entropy,
    add_lora_adapters,
    clef_logits,
    clef_option_keys,
    clef_training_item,
    clef_training_network,
    collate_decisions,
    decision_logits,
    load_decision_model,
    load_trainable_decision_model,
    save_clef_model,
    save_decision_model,
)
from unsloth_zoo.mlx.generate import generation_mode  # noqa: E402
from unsloth_zoo.mlx.trainer import MLXDecisionTrainer, MLXTrainingConfig, _default_flow_evaluates_final_step, _length_grouped_batches  # noqa: E402
from unsloth_zoo.mlx.utils import _forward_text_hidden_states  # noqa: E402


class _Reference(torch.nn.Module):
    # laya.common.DecisionModel up to its decision logits.
    def __init__(self, encoder, head_layers):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = torch.nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.0, batch_first = True, norm_first = True)
        self.head = torch.nn.TransformerEncoder(layer, head_layers, enable_nested_tensor = False)
        self.type_emb = torch.nn.Embedding(3, d)
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(d), torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 1)
        )
        self.act_head = torch.nn.Sequential(torch.nn.Linear(d + 4, 256), torch.nn.GELU(), torch.nn.Linear(256, 2))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask = ~attention_mask.bool())
        index = marker_pos.clamp(min = 0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, index)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


# 1 and 0 head layers exercise the trimmed last layer alone and the no-head gather.
@pytest.fixture(scope = "module", params = [2, 1, 0])
def checkpoint(request, tmp_path_factory):
    torch.manual_seed(0)
    config = transformers.ModernBertConfig(
        vocab_size = 97,
        hidden_size = 128,
        intermediate_size = 96,
        num_hidden_layers = 4,
        num_attention_heads = 2,
        # Short window and distinct thetas, so the sliding layers and their RoPE base both matter.
        local_attention = 8,
        global_attn_every_n_layers = 3,
        rope_parameters = {
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 50.0},
        },
        # transformers 4.x reads only these, 5.x only rope_parameters.
        global_rope_theta = 160000.0,
        local_rope_theta = 50.0,
        pad_token_id = 0,
    )
    reference = _Reference(transformers.ModernBertModel(config), head_layers = request.param).eval()
    for p in reference.parameters():
        # Default init leaves LayerNorm at identity and biases at zero, which would hide their mapping.
        p.data.add_(0.1 * torch.randn_like(p))
    folder = tmp_path_factory.mktemp("laya")
    (folder / "encoder").mkdir()
    (folder / "tokenizer").mkdir()
    (folder / "tokenizer" / "vocab.txt").write_text("a")
    config.save_pretrained(folder / "encoder")
    (folder / "rl_agent_config.json").write_text(json.dumps({"head_layers": request.param}))
    save_file({k: v.contiguous() for k, v in reference.state_dict().items()}, folder / "model.safetensors")
    return reference, folder


def _batch():
    # Row 1 is padded far past the window, so its padding queries have no valid key in range.
    lengths, markers = [30, 11], [[3, 17, 26], [2, 9, 0]]
    ids = np.zeros((2, 30), np.int64)
    rng = np.random.default_rng(0)
    for row, length in enumerate(lengths):
        ids[row, :length] = rng.integers(1, 97, length)
    return {
        "input_ids": ids,
        "attention_mask": (np.arange(30)[None] < np.array(lengths)[:, None]).astype(np.int64),
        "marker_pos": np.array(markers),
        "marker_mask": np.array([[True, True, True], [True, True, False]]),
        "qtype": np.array([0, 2]),
    }


def test_logits_match_the_torch_reference(checkpoint):
    reference, folder = checkpoint
    batch = _batch()
    with torch.inference_mode():
        expected = reference(**{k: torch.from_numpy(v) for k, v in batch.items()}).numpy()
    model = load_decision_model(folder)
    got = model.logits(batch)
    np.testing.assert_allclose(got, expected, atol = 2e-5, rtol = 0)
    np.testing.assert_array_equal(np.array(model(**{k: mx.array(v) for k, v in batch.items()}).astype(mx.float32)), got)
    assert got[1, 2] == -1e4


def test_julia_checkpoint_loads_from_its_own_config_file(checkpoint, tmp_path):
    folder = shutil.copytree(checkpoint[1], tmp_path / "julia")
    (folder / "rl_agent_config.json").rename(folder / "julia_config.json")
    np.testing.assert_array_equal(load_decision_model(folder).logits(_batch()), load_decision_model(checkpoint[1]).logits(_batch()))


def test_casting_load_leaves_no_source_buffers_cached(checkpoint):
    folder = checkpoint[1]
    assert {v.dtype for v in mx.load(str(folder / "model.safetensors")).values()} == {mx.float32}
    model = load_decision_model(folder, compute_dtype = mx.float16)
    mx.synchronize()
    assert mx.get_cache_memory() < (folder / "model.safetensors").stat().st_size // 10
    del model


@pytest.mark.parametrize("dtype, atol", [(mx.float16, 1e-2), (mx.bfloat16, 5e-2)])
def test_reduced_precision_keeps_norms_in_float32(checkpoint, dtype, atol, tmp_path):
    reference, folder = checkpoint
    batch = _batch()
    with torch.inference_mode():
        expected = reference(**{k: torch.from_numpy(v) for k, v in batch.items()}).numpy()
    # Published checkpoints store every tensor, norms included, in float16.
    half = shutil.copytree(folder, tmp_path / "half")
    stored = mx.load(str(folder / "model.safetensors"))
    mx.save_safetensors(str(half / "model.safetensors"), {k: v.astype(mx.float16) for k, v in stored.items()})
    model = load_decision_model(folder, compute_dtype = dtype)
    for loaded in (model, load_decision_model(half, compute_dtype = dtype)):
        for module in loaded.modules():
            if isinstance(module, (Embedding, LayerNorm, Linear)):
                want = mx.float32 if isinstance(module, LayerNorm) else dtype
                assert {p.dtype for _, p in tree_flatten(module.parameters())} == {want}
    assert mx.array_equal(model.encoder.final_norm.weight, stored["encoder.final_norm.weight"])
    assert model.encoder.layers[0].attn.Wqkv(mx.zeros((1, 1, 128))).dtype == dtype
    assert model.encoder.embeddings(mx.zeros((1, 2), mx.int32)).dtype == mx.float32
    assert model.encoder.layers[1](mx.zeros((1, 2, 128)), None).dtype == mx.float32
    np.testing.assert_allclose(model.logits(batch), expected, atol = atol, rtol = 0)


def test_load_drains_generation_streams_before_clearing_the_cache(checkpoint, monkeypatch):
    import sys
    import types

    stream = mx.new_stream(mx.default_device())
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", types.SimpleNamespace(generation_stream = stream))
    calls = []
    synchronize, clear_cache = mx.synchronize, mx.clear_cache
    monkeypatch.setattr(mx, "synchronize", lambda *a: calls.append(("sync", a)) or synchronize(*a))
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append(("clear", ())) or clear_cache())
    load_decision_model(checkpoint[1])
    clear = calls.index(("clear", ()))
    assert ("sync", (stream,)) in calls[:clear] and ("sync", ()) in calls[:clear]


def _decoder(quantized = False, bits = 8):
    from mlx_lm.models import qwen3_5

    text = dict(model_type = "qwen3_5_text", hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1, vocab_size = 512)
    # The default recurrent state is the full model's, megabytes per token whatever the hidden size.
    text.update(linear_num_value_heads = 4, linear_num_key_heads = 2, linear_key_head_dim = 32, linear_value_head_dim = 16)
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type = "qwen3_5", text_config = text))
    model.set_dtype(mx.bfloat16)
    if quantized:
        quantize(model, group_size = 64, bits = bits)
    return model


def test_lora_adapter_is_merged_into_the_decoder_by_layer_name(tmp_path):
    model, weights = _decoder(), str(tmp_path / "adapter_model.safetensors")
    before = dict(tree_flatten(model.parameters()))
    # The prefixes PEFT writes for a causal LM, its inner text model and a vision-language wrapper.
    targets = {"layers.3.self_attn.q_proj": "base_model.model.model.", "layers.1.linear_attn.out_proj": "base_model.model.", "layers.1.mlp.down_proj": "base_model.model.model.language_model."}
    tensors = {}
    for name, prefix in targets.items():
        rows, columns = before[f"language_model.model.{name}.weight"].shape
        tensors[f"{prefix}{name}.lora_A.weight"], tensors[f"{prefix}{name}.lora_B.weight"] = mx.random.normal((4, columns)), mx.random.normal((rows, 4))
    wrong_shape = {"base_model.model.layers.1.mlp.up_proj.lora_A.weight": mx.zeros((4, 64)), "base_model.model.layers.1.mlp.up_proj.lora_B.weight": mx.zeros((64, 4))}
    refused = [(model, problem, {}) for problem in ({"use_dora": True}, {"peft_type": "LOHA"}, {"bias": "all"})]
    for base, problem, stray in [*refused, (model, {}, wrong_shape), (_decoder(quantized = True), {}, {}), (model, {}, {})]:
        (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 4, "lora_alpha": 12, **problem}))
        mx.save_safetensors(weights, {**tensors, **stray})
        with pytest.raises(ValueError) if base is not model or problem or stray else contextlib.nullcontext():
            _merge_lora(base, tmp_path)
    after = dict(tree_flatten(model.parameters()))
    assert {name for name in before if not mx.array_equal(before[name], after[name])} == {f"language_model.model.{name}.weight" for name in targets}
    for name, prefix in targets.items():
        weight = f"language_model.model.{name}.weight"
        delta = tensors[f"{prefix}{name}.lora_B.weight"] @ tensors[f"{prefix}{name}.lora_A.weight"]
        assert after[weight].dtype == mx.bfloat16 and mx.abs(after[weight] - (before[weight].astype(mx.float32) + 3 * delta).astype(mx.bfloat16)).max().item() <= 2**-7


def test_shared_prefix_continuations_match_a_full_forward():
    reader, rng = object.__new__(_LabelModel), np.random.default_rng(0)
    reader.model = _decoder()
    prefix = rng.integers(1, 500, 40).tolist()
    prompts = [prefix + rng.integers(1, 500, n).tolist() for n in (1, 7, 23)]
    with generation_mode(reader.model):
        shared = [np.array(hidden.astype(mx.float32)) for hidden in reader._hidden_states(prompts)]
        full = [np.array(reader._hidden(ids).astype(mx.float32)) for ids in prompts]
    for got, expected in zip(shared, full):
        np.testing.assert_allclose(got, expected, atol = 2e-2)


@pytest.mark.parametrize("quantized", [False, True])
def test_label_scores_are_the_output_head_rows_of_the_last_token(quantized):
    reader, ids, picks = _LabelModel.__new__(_LabelModel), list(range(40, 63)), mx.array([300, 7, 301])
    reader.model, reader.labels = _decoder(quantized), [("A", 300), ("B", 7), ("C", 301), ("D", 511)]
    scores = reader._read(SimpleNamespace(options = [None] * 3), ids, reader._hidden(ids))
    logits, hidden = reader.model(mx.array(ids)[None])[0, -1], _forward_text_hidden_states(reader.model, mx.array(ids)[None])[0, -1]
    gap = lambda expected: mx.abs(mx.array(scores) - expected.astype(mx.float32)).max().item()
    assert len(scores) == 3 and gap(logits[picks]) <= 2**-5 and (quantized or gap(reader.model.language_model.lm_head.weight[picks].astype(mx.float32) @ hidden.astype(mx.float32)) == 0)


def test_kev_scores_each_option_end_against_the_last_token(tmp_path, monkeypatch):
    import os
    import re
    import types
    from pathlib import Path

    texts, specials, decoder = [], {"<|box_end|>": 500}, _decoder()
    encode = lambda text, add_special_tokens: texts.append(text) or [specials.get(piece, 300 if piece.startswith("<|") else ord(piece) % 256) for piece in re.findall(r"<\|\w+\|>|.", text, re.S)]
    load = lambda self, source, revision, *args, **kwargs: vars(self).update(base = (source, revision), seen = Path(source).is_dir() and sorted(os.listdir(source)), model = decoder, tokenizer = types.SimpleNamespace(encode = encode))
    monkeypatch.setattr(_KevModel, "_load", load)
    for name, content in {"adapter_config.json": {"base_model_name_or_path": "org/base", "revision": None}, "training_config.json": {"base_revision": "pinned"}}.items():
        (tmp_path / name).write_text(json.dumps(content))
    torch.manual_seed(0)
    head = {name: torch.randn(shape) / 16 for name, shape in {"q.weight": (16, 64), "q.bias": (16,), "k.weight": (16, 64), "k.bias": (16,)}.items()}
    torch.save({"head": head, "head_dim": 16, "temperature": 0.5}, tmp_path / "head.pt")
    kev = load_decision_model(tmp_path)
    questions = {"level": {"type": "score", "instructions": "s", "criteria": ["low", None]}, "sure": {"type": "noul", "instructions": "n", "criteria": {"false": "N"}}, "pick": {"type": "choice", "instructions": ["i", 2], "criteria": {"x<|box_end|>": {"d": True}, "y": None, "z": "last"}}}
    result = kev.answer({"a": {"b": [1, {"c": "s", "e": 2}]}, "n": None}, questions)
    asked = ("s<|box_start|>low<|box_end|><|box_start|><|box_end|>", "n<|box_start|>no: N<|box_end|><|box_start|>yes<|box_end|>", "- i\n- 2<|box_start|>x<¦box_end¦>: d: True<|box_end|><|box_start|>y<|box_end|><|box_start|>z: last<|box_end|>")
    assert kev.base == ("org/base", "pinned") and texts[-3:] == [f"<|fim_prefix|>a:\n  b:\n    - 1\n    - c: s\n      e: 2\nn: <|fim_middle|>{rest}<|fim_suffix|>" for rest in asked]
    ids = encode(texts[-1], False)
    # The engine reads hidden states under generation_mode, whose fused kernels differ slightly from a plain forward.
    with generation_mode(kev.model):
        hidden = np.array(_forward_text_hidden_states(kev.model, mx.array(ids)[None])[0].astype(mx.float32))
    q, k = (hidden[rows] @ head[f"{name}.weight"].numpy().T + head[f"{name}.bias"].numpy() for name, rows in (("q", -1), ("k", [i for i, token in enumerate(ids) if token == 500])))
    weights = np.exp((scores := k @ q / 4 / 0.5) - scores.max())
    assert list(result["answers"]["pick"]["probabilities"].values()) == pytest.approx(weights / weights.sum(), abs = 1e-4) and result["usage"]["input_tokens"] == sum(len(encode(text, False)) for text in texts[-4:-1])
    # A conversion ships the merged decoder, here with the head as safetensors and its settings beside it.
    for name in ("adapter_config.json", "head.pt"):
        (tmp_path / name).unlink()
    save_file(head, tmp_path / "kev_head.safetensors")
    save_file({"decoder": torch.zeros(1)}, tmp_path / "model.safetensors")
    for name, content in {"config.json": {}, "kev_config.json": {"head_dim": 16, "temperature": 0.5}}.items():
        (tmp_path / name).write_text(json.dumps(content))
    merged = load_decision_model(tmp_path)
    assert merged.base[1] is None and merged.seen == ["config.json", "kev_config.json", "model.safetensors", "training_config.json"] and merged.answer({"a": {"b": [1, {"c": "s", "e": 2}]}, "n": None}, questions) == result
    save_file({**head, "extra": torch.zeros(1)}, tmp_path / "kev_head.safetensors")
    with pytest.raises(ValueError, match = "no Kev pointer head"):
        load_decision_model(tmp_path)


class _RoutingReference(torch.nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.query_norm, self.memory_norm, self.feedforward_norm = (torch.nn.LayerNorm(width) for _ in range(3))
        self.attention = torch.nn.MultiheadAttention(width, heads, batch_first = True)
        self.feedforward = torch.nn.Sequential(torch.nn.Linear(width, feedforward), torch.nn.GELU(), torch.nn.Dropout(0.0), torch.nn.Linear(feedforward, width))

    def forward(self, queries, memory):
        queries = queries + self.attention(self.query_norm(queries), self.memory_norm(memory), self.memory_norm(memory), need_weights = False)[0]
        return queries + self.feedforward(self.feedforward_norm(queries))


class _JointReference(torch.nn.Module):
    """Clef's joint head as its repo defines it, for one request; parameter names are the checkpoint's."""

    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        for name in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
            setattr(self, f"{name}_projection", torch.nn.Linear(hidden_size, width, bias = False))
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList(_RoutingReference(width, heads, feedforward) for _ in range(routing_layers))
        self.layers = torch.nn.ModuleList(torch.nn.TransformerDecoderLayer(width, heads, feedforward, 0.0, "gelu", batch_first = True, norm_first = True) for _ in range(layers))
        self.option_summary_norm, self.field_norm, self.option_norm = (torch.nn.LayerNorm(width) for _ in range(3))
        self.residual_scorer = torch.nn.Sequential(torch.nn.Linear(4 * width, width), torch.nn.GELU(), torch.nn.Dropout(0.0), torch.nn.Linear(width, 1))
        self.prior_logit_scale, self.joint_logit_scale, self.residual_gate = (torch.nn.Parameter(torch.zeros(())) for _ in range(3))

    def forward(self, hidden, embedding, ids, question_spans, option_spans, types):
        normalize, hidden = torch.nn.functional.normalize, self.hidden_norm(hidden)
        memory, overall = self.memory_projection(hidden)[None], hidden[-1]
        questions = torch.stack([hidden[start:end].mean(0) for start, end in question_spans])
        lexical = [torch.stack([embedding[ids[start:end]].mean(0) for start, end in spans]) for spans in option_spans]
        routed = torch.cat([
            self.option_context_projection(torch.stack([hidden[start:end].mean(0) for start, end in spans])) + self.option_lexical_projection(rows) + self.option_question_projection(question)
            for spans, rows, question in zip(option_spans, lexical, questions)
        ])[None]
        for layer in self.evidence_layers:
            routed = layer(routed, memory)
        routed = torch.split(routed[0], [len(spans) for spans in option_spans])
        fields = self.question_projection(questions)
        summaries = torch.stack([torch.softmax(options @ field / options.shape[-1] ** 0.5, 0) @ options for field, options in zip(fields, routed)])
        fields = (fields + self.option_summary_norm(summaries) + self.global_projection(overall) + self.type_embedding(torch.tensor(types)))[None]
        for layer in self.layers:
            fields = layer(fields, memory)
        logits = []
        for field, question, rows, options in zip(self.field_norm(fields[0]), questions, lexical, routed):
            options, field = self.option_norm(options), field.expand(len(options), -1)
            prior = self.prior_logit_scale.clamp(max = np.log(100.0)).exp() * (normalize(rows, dim = -1) @ normalize(question + overall, dim = -1))
            residual = self.residual_scorer(torch.cat([field, options, field * options, (field - options).abs()], -1)).squeeze(-1)
            joint = self.joint_logit_scale.clamp(max = np.log(100.0)).exp() * torch.cosine_similarity(field, options, dim = -1) + residual
            logits.append(prior + torch.sigmoid(self.residual_gate) * joint)
        return logits


@pytest.mark.parametrize("quantized", [False, True])
def test_clef_answers_every_question_from_one_prompt_like_its_reference_head(tmp_path, monkeypatch, quantized):
    from unsloth_zoo.mlx.decision import ClefModel

    torch.manual_seed(0)
    config = {"hidden_size": 64, "width": 32, "routing_layers": 2, "layers": 2, "heads": 4, "feedforward": 48}
    reference = _JointReference(**config).eval()
    for parameter in reference.parameters():
        parameter.data.add_(0.3 * torch.randn_like(parameter))
    (tmp_path / "joint_head_config.json").write_text(json.dumps(config))
    save_file({name: value.contiguous() for name, value in reference.state_dict().items()}, tmp_path / "joint_head.safetensors")
    # Vowels take two tokens, so a span counted in characters reads the wrong rows.
    tokens = lambda text: [token for c in text for token in ([ord(c) % 512, 7] if c in "aeiou" else [ord(c) % 512])]
    pieces, encode = [], lambda text, add_special_tokens: pieces.append(text) or tokens(text)
    monkeypatch.setattr(ClefModel, "_load", lambda self, *args: vars(self).update(model = _decoder(quantized), tokenizer = SimpleNamespace(encode = encode)))
    questions = {"route": {"type": "choice", "instructions": {"b": 1, "a": "é"}, "criteria": {"ship": "late", "bill": None}}, "level": {"type": "score", "instructions": "How bad?", "criteria": ["fine", "bad", "awful"]}, "angry": {"type": "noul", "instructions": "Angry?", "criteria": {"false": "calm"}}}
    model = load_decision_model(tmp_path)
    result = model.answer({"z": [1, None], "y": "s"}, questions)
    # Keys sorted, compact JSON, a choice's options sorted, noul yes first with a default description.
    marked = [
        '{"a":"é","b":1}', '{"option_id":"bill"}', '{"description":"late","option_id":"ship"}',
        "How bad?", '{"description":"fine","option_id":"0"}', '{"description":"bad","option_id":"1"}', '{"description":"awful","option_id":"2"}',
        "Angry?", '{"description":"The proposition is true or the answer is yes.","option_id":"true"}', '{"description":"calm","option_id":"false"}',
    ]
    assert "".join(pieces) == (
        "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n" + '{"y":"s","z":[1,null]}\n\nSCHEMA FIELDS:\n'
        f"\nFIELD 1\nID: route\nTYPE: choice\nINSTRUCTION: {marked[0]}\nALLOWED OPTIONS:\nOPTION 1: {marked[1]}\nOPTION 2: {marked[2]}\nEND FIELD\n"
        f"\nFIELD 2\nID: level\nTYPE: score\nINSTRUCTION: {marked[3]}\nALLOWED OPTIONS:\nOPTION 1: {marked[4]}\nOPTION 2: {marked[5]}\nOPTION 3: {marked[6]}\nEND FIELD\n"
        f"\nFIELD 3\nID: angry\nTYPE: noul\nINSTRUCTION: {marked[7]}\nALLOWED OPTIONS:\nOPTION 1: {marked[8]}\nOPTION 2: {marked[9]}\nEND FIELD\n"
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    )
    starts = np.cumsum([0, *(len(tokens(piece)) for piece in pieces)])
    spans = [(int(starts[pieces.index(text)]), int(starts[pieces.index(text) + 1])) for text in marked]
    ids = torch.tensor(tokens("".join(pieces)))
    with generation_mode(model.model):
        hidden = torch.from_numpy(np.array(_forward_text_hidden_states(model.model, mx.array(ids.numpy())[None])[0].astype(mx.float32)))
    head = model.model.language_model.lm_head
    # An MLX conversion quantizes the output embedding the options are read from.
    weight = mx.dequantize(head.weight, head.scales, head.biases, group_size = 64, bits = 8) if quantized else head.weight
    embedding = torch.from_numpy(np.array(weight.astype(mx.float32)))
    with torch.inference_mode():
        route, level, angry = reference(hidden, embedding, ids, [spans[0], spans[3], spans[7]], [spans[1:3], spans[4:7], spans[8:]], [1, 2, 0])
    for name, logits in (("route", route), ("level", level)):
        assert list(result["answers"][name]["probabilities"].values()) == pytest.approx(torch.softmax(logits, 0).tolist(), abs = 1e-4)
    assert list(result["answers"]["route"]["probabilities"]) == ["bill", "ship"] and result["answers"]["angry"]["noul"] == pytest.approx(torch.softmax(angry, 0)[0].item(), abs = 1e-4) and result["usage"]["input_tokens"] == len(ids)


def test_prompts_of_a_request_run_their_shared_prefix_once(monkeypatch):
    from unsloth_zoo.mlx import utils

    reader, shared = _LabelModel.__new__(_LabelModel), list(range(40, 80))
    reader.model = _decoder()
    # The last prompt is the shared prefix itself, so one of its tokens is left to continue with.
    prompts = [shared + [7, 8, 9], shared + [7], shared + [300, 301, 302, 303], shared]
    calls, forward = [], utils._forward_text_hidden_states
    monkeypatch.setattr(utils, "_forward_text_hidden_states", lambda model, inputs, **kwargs: calls.append((inputs.shape[1], kwargs)) or forward(model, inputs, **kwargs))
    with generation_mode(reader.model):
        got = list(reader._hidden_states(prompts))
        continued, calls[:] = list(calls), []
        want = [reader._hidden(ids) for ids in prompts]
        assert all(mx.array_equal(a, b) for a, b in zip(reader._hidden_states([prompts[0]]), want)) and all(mx.array_equal(a, reader._hidden(ids)) for a, ids in zip(reader._hidden_states([[1, 2, 3], [1, 2, 9]]), ([1, 2, 3], [1, 2, 9])))
    # One prompt, or a shared prefix too short to pay for itself, takes the plain pass.
    assert not any("cache" in kwargs for _, kwargs in calls)
    # Prompts that part before the shortest one ends share only up to where they part.
    with generation_mode(reader.model):
        calls.clear()
        list(reader._hidden_states([shared[:20] + [7] * 5, shared[:20] + [9] * 8]))
    assert [length for length, _ in calls] == [20, 5, 8]
    assert [length for length, _ in continued] == [39, 4, 2, 5, 1] and len({id(kwargs["cache"]) for _, kwargs in continued}) == 5
    assert [kwargs["position_ids"][:, 0].tolist() for _, kwargs in continued[1:]] == [[list(range(39, len(ids)))] * 3 for ids in prompts]
    for a, b in zip(got, want):
        assert a.shape == b.shape and mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() <= 2**-4 * mx.abs(b.astype(mx.float32)).max().item()


def test_saved_checkpoint_matches_the_torch_state_dict(checkpoint, tmp_path):
    reference, folder = checkpoint
    model = load_decision_model(folder)
    model.update(tree_map(lambda v: v + 0.25, model.parameters()))
    save_decision_model(model, tmp_path, folder, {"head_layers": len(model.head.layers), "fine_tuned": True})

    saved = load_file(tmp_path / "model.safetensors")
    assert {v.dtype for v in saved.values()} == {torch.float16}
    # The MLX model holds neither the act head nor the temperature buffer; they are carried over.
    expected = {
        k: (v if k.startswith("act_head.") or k == "temperature" else v + 0.25).half()
        for k, v in reference.state_dict().items()
    }
    assert saved.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(saved[name], value), name
    assert json.loads((tmp_path / "rl_agent_config.json").read_text())["fine_tuned"] is True
    assert (tmp_path / "encoder" / "config.json").is_file() and (tmp_path / "tokenizer" / "vocab.txt").is_file()
    np.testing.assert_array_equal(load_decision_model(tmp_path).logits(_batch()), _half(model).logits(_batch()))


def _half(model):
    # What a float16 round trip of the weights leaves, back in float32.
    model.update(tree_map(lambda v: v.astype(mx.float16).astype(mx.float32), model.parameters()))
    return model


def test_save_replaces_a_linked_weights_file(checkpoint, tmp_path):
    _, folder = checkpoint
    shutil.copytree(folder, tmp_path / "snapshot")
    blob = tmp_path / "blob"
    (tmp_path / "snapshot" / "model.safetensors").rename(blob)
    (tmp_path / "snapshot" / "model.safetensors").symlink_to(blob)
    before = blob.read_bytes()
    save_decision_model(load_decision_model(tmp_path / "snapshot"), tmp_path / "snapshot", tmp_path / "snapshot")
    assert blob.read_bytes() == before
    assert not (tmp_path / "snapshot" / "model.safetensors").is_symlink()
    assert sorted(p.name for p in (tmp_path / "snapshot").iterdir()) == sorted(p.name for p in folder.iterdir())


@pytest.mark.parametrize("bad", [1e6, float("nan")])
def test_save_refuses_weights_float16_cannot_hold(checkpoint, tmp_path, bad):
    model = load_decision_model(checkpoint[1])
    model.type_emb.weight = model.type_emb.weight * bad
    with pytest.raises(ValueError, match = "type_emb.weight"):
        save_decision_model(model, tmp_path, checkpoint[1])
    assert not (tmp_path / "rl_agent_config.json").exists()


def _items():
    rng = np.random.default_rng(1)
    rows = [(30, [3, 17, 26], 0), (11, [2, 9], 2), (20, [4, 12, 18], 1), (25, [5, 21], 2), (16, [3, 8, 14], 0), (28, [6, 19], 1)]
    return [
        {"input_ids": rng.integers(1, 97, length).tolist(), "markers": markers, "qtype": qtype, "target": rng.dirichlet(np.ones(len(markers))).tolist()}
        for length, markers, qtype in rows
    ]


def _config(**kwargs):
    defaults = dict(per_device_train_batch_size = 2, gradient_accumulation_steps = 2, num_train_epochs = 2, max_steps = 0, learning_rate = 1e-3, weight_decay = 0.1, lr_scheduler_type = "constant", warmup_steps = 0, max_grad_norm = 1.0)
    return MLXTrainingConfig(**{**defaults, **kwargs})


def _without_dropout(model):
    for module in model.modules():
        if isinstance(module, Dropout):
            module._p_1 = 1.0
    return model


def _parameters(model):
    return {name: np.array(value.astype(mx.float32)) for name, value in tree_flatten(model.parameters())}


def test_training_forward_matches_eval_until_dropout_applies(checkpoint):
    model = load_trainable_decision_model(checkpoint[1], full_finetuning = True)
    batch = collate_decisions(_items(), 0)
    batch.pop("target")
    model.eval()
    want = np.array(model(**batch))
    model.train()
    dropped = np.array(model(**batch))
    np.testing.assert_allclose(np.array(_without_dropout(model)(**batch)), want, atol = 1e-4)
    if model.head.layers:
        assert np.abs(dropped - want)[np.array(batch["marker_mask"])].min() > 1e-4


def test_decay_skips_what_transformers_skips_by_class_and_by_name():
    import mlx.nn as nn
    from unsloth_zoo.mlx.trainer import _no_decay_names

    class Gated(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight, self.dt_bias = mx.ones(2), mx.zeros(2)

    block = nn.Module()
    block.proj, block.ln, block.norm, block.q_norm, block.gate = Linear(2, 2), LayerNorm(2), Gated(), nn.RMSNorm(2), Gated()
    block.normalize = Linear(2, 2, bias = False)
    assert _no_decay_names(block) == {"proj.bias", "ln.weight", "ln.bias", "norm.weight", "norm.dt_bias", "q_norm.weight", "gate.dt_bias"}


@pytest.mark.parametrize("gradient_checkpointing", [True, False])
def test_two_training_steps_match_torch_adamw(checkpoint, gradient_checkpointing):
    reference, folder = checkpoint
    # One epoch of three pairs at accumulation 2: a full step, then a step from the odd micro-batch alone.
    items = _items()
    order = _length_grouped_batches([len(item["input_ids"]) for item in items], 2, random.Random(3407))
    torch_model = copy.deepcopy(reference).train()
    torch_model.encoder.config.reference_compile = False
    norms = {f"{n}.{p}" for n, m in torch_model.named_modules() if isinstance(m, torch.nn.LayerNorm) for p, _ in m.named_parameters()}
    groups = {}
    for name, param in torch_model.named_parameters():
        groups.setdefault((name.startswith("encoder."), name not in norms and "bias" not in name), []).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": 1e-3 if encoder else 3e-4, "weight_decay": 0.1 if decay else 0.0} for (encoder, decay), params in groups.items()],
        betas = (0.8, 0.95), eps = 1e-6,
    )
    steps = []
    for rows in (order[0] + order[1], order[2]):
        batch = {k: torch.from_numpy(np.array(v)) for k, v in collate_decisions([items[i] for i in rows], 0).items()}
        target = batch.pop("target")
        optimizer.zero_grad()
        loss = -(target * torch.log_softmax(torch_model(**batch), -1)).sum(-1).mean()
        loss.backward()
        steps.append((loss.item(), torch.nn.utils.clip_grad_norm_(torch_model.parameters(), 1.0).item()))
        optimizer.step()

    model = _without_dropout(load_trainable_decision_model(folder, full_finetuning = True, gradient_checkpointing = gradient_checkpointing))
    recorder = _Recorder()
    args = _config(num_train_epochs = 1, adam_beta1 = 0.8, adam_beta2 = 0.95, adam_epsilon = 1e-6)
    MLXDecisionTrainer(model, args, items, head_learning_rate = 3e-4, callbacks = [recorder]).train()
    np.testing.assert_allclose([(log["loss"], log["grad_norm"]) for log in recorder.logs[:2]], steps, rtol = 2e-3)
    got, want = _parameters(model), torch_model.state_dict()
    for name in ("encoder.layers.1.attn.Wqkv.weight", "encoder.layers.1.mlp_norm.weight", "scorer.0.weight", "scorer.1.weight", "scorer.1.bias", "type_emb.weight"):
        # Adam's first steps are about one learning rate per element whatever the gradient's size.
        assert (np.abs(got[name] - want[name].numpy()) > 3e-5).mean() < 0.01, name


def test_lora_adapters_target_encoder_linears_and_merge_on_save(checkpoint, tmp_path):
    _, folder = checkpoint
    model = add_lora_adapters(load_trainable_decision_model(folder), r = 4, lora_alpha = 8, target_modules = ["Wqkv", "mlp.Wo"])
    trainable = {name for name, _ in tree_flatten(model.trainable_parameters())}
    adapters = {f"encoder.layers.{i}.{path}.{half}" for i in range(4) for path in ("attn.Wqkv", "mlp.Wo") for half in ("lora_a", "lora_b")}
    assert trainable == adapters | {name for name in _parameters(model) if not name.startswith("encoder.")}
    with pytest.raises(RuntimeError, match = "already"):
        add_lora_adapters(model)
    with pytest.raises(ValueError, match = "matches no"):
        add_lora_adapters(load_trainable_decision_model(folder), target_modules = ["Wxyz"])

    for module in model.modules():
        if "lora_b" in module:
            module.lora_b = mx.random.normal(module.lora_b.shape) * 0.1
    adapter = model.encoder.layers[3].mlp.Wo
    merged = np.array(adapter.linear.weight).astype(np.float32) + 2.0 * (np.array(adapter.lora_b).T @ np.array(adapter.lora_a).T)
    save_decision_model(model, tmp_path, folder)
    saved = load_file(tmp_path / "model.safetensors")
    np.testing.assert_allclose(saved["encoder.layers.3.mlp.Wo.weight"].float().numpy(), merged, atol = 2e-3)
    assert saved.keys() == load_file(folder / "model.safetensors").keys()
    assert {name for name, _ in tree_flatten(model.trainable_parameters())} == trainable
    model.eval()
    np.testing.assert_allclose(load_decision_model(tmp_path, compute_dtype = mx.float16).logits(_batch()), model.logits(_batch()), atol = 5e-2)


def test_a_fresh_run_draws_dropout_from_its_seed(checkpoint, tmp_path):
    losses = []
    for stray in (1, 2):
        mx.random.seed(0)
        model = add_lora_adapters(load_trainable_decision_model(checkpoint[1]), r = 4, lora_dropout = 0.3)
        mx.random.seed(stray)
        trainer = MLXDecisionTrainer(model, _config(output_dir = str(tmp_path / str(stray)), max_steps = 2), _items())
        losses.append(trainer.train().metrics["train_loss"])
    assert losses[0] == losses[1]


@pytest.mark.parametrize("target_modules, objective", [("all-linear", None), (r"layers\.0\.attn\.Wqkv", (0.1, 0.5, 0.0))])
def test_layerwise_gradients_match_one_graph_under_dropout(checkpoint, target_modules, objective):
    import mlx.nn as nn

    # With one early adapter, the frozen layers after it must still pass the gradient down.
    model = add_lora_adapters(load_trainable_decision_model(checkpoint[1], gradient_checkpointing = False), r = 4, lora_dropout = 0.3, target_modules = target_modules)
    batch = collate_decisions(_items(), 0)
    mx.random.seed(7)
    want_loss, want = _MarkerStep(model, 0, False, objective)(batch)
    key, far = mx.random.state[0].tolist(), {}
    for compile in (False, True):
        mx.random.seed(7)
        loss, got = _LayerwiseStep(model, compile, objective)(batch)
        assert abs(loss.item() - want_loss.item()) < 1e-3 and mx.random.state[0].tolist() == key
        want_flat = dict(tree_flatten(want))
        assert {name for name, _ in tree_flatten(got)} == set(want_flat)
        for name, value in tree_flatten(got):
            reference = np.array(want_flat[name])
            far[compile, name] = (np.abs(np.array(value) - reference) > 1e-3 + 0.05 * np.abs(reference)).mean()
    # float16 noise moves under 1% of a tensor's elements on an M3, and 7% was seen on an M1; other dropout masks move about 80%.
    assert not {where: round(share, 3) for where, share in far.items() if share >= 0.25}


def test_decision_logits_match_the_forward_item_by_item(checkpoint):
    model, items = load_trainable_decision_model(checkpoint[1]), _items()
    with pytest.raises(KeyError):
        decision_logits(model, [{"input_ids": [1]}], 0)
    got = decision_logits(model, items, 0, batch_size = 4)
    assert model.training
    model.eval()
    for item, row in zip(items, got, strict = True):
        want = np.array(model(**{k: v for k, v in collate_decisions([item], 0).items() if k != "target"}))[0]
        np.testing.assert_allclose(row, want, atol = 2e-2)
        assert row.shape == (len(item["markers"]),)


@pytest.fixture
def clef(tmp_path, monkeypatch):
    from unsloth_zoo.mlx.decision import _TYPE_IDS, ClefModel

    torch.manual_seed(0)
    config = {"hidden_size": 64, "width": 32, "routing_layers": 2, "layers": 2, "heads": 4, "feedforward": 48}
    reference = _JointReference(**config)
    for parameter in reference.parameters():
        parameter.data.add_(0.3 * torch.randn_like(parameter))
    (tmp_path / "joint_head_config.json").write_text(json.dumps(config))
    save_file({name: value.contiguous() for name, value in reference.state_dict().items()}, tmp_path / "joint_head.safetensors")
    encode = lambda text, add_special_tokens: [ord(c) % 512 for c in text]
    monkeypatch.setattr(ClefModel, "_load", lambda self, *args: vars(self).update(model = (mx.random.seed(7), _decoder(args[5], bits = 4))[1], tokenizer = SimpleNamespace(encode = encode)))
    pipeline = load_decision_model(tmp_path)
    questions = {"route": {"type": "choice", "instructions": "where", "criteria": {"c": "z", "a": "x", "b": None}}, "ok": {"type": "noul", "instructions": "fine?"}}
    assert [clef_option_keys(pipeline, question) for question in questions.values()] == [["a", "b", "c"], ["true", "false"]]

    def item(state, max_length = None, count = 2):
        record = clef_training_item(pipeline, state, dict(list(questions.items())[:count]), max_length)
        assert record["types"] == [_TYPE_IDS.index(kind) for kind in ("choice", "noul")[:count]]
        return {**record, "targets": [[0.0, 0.25, 0.75], [1.0, 0.0]][:count]}

    return pipeline, reference, item


def test_clef_loss_and_head_gradients_match_the_reference_head(clef):
    pipeline, reference, item = clef
    record, network = item("hello"), ClefNetwork(pipeline)
    network.encoder.freeze()
    network.encoder.language_model.lm_head.unfreeze()
    loss, grads = _clef_record_loss_and_grad(network, record)
    ids = torch.tensor(record["input_ids"])
    hidden = torch.from_numpy(np.array(_forward_text_hidden_states(pipeline.model, mx.array(ids.numpy())[None])[0].astype(mx.float32)))
    embedding = torch.from_numpy(np.array(pipeline.model.language_model.lm_head.weight.astype(mx.float32)))
    logits = reference(hidden, embedding, ids, record["question_spans"], record["option_spans"], record["types"])
    expected = -sum((torch.tensor(target) * torch.log_softmax(row, 0)).sum() for row, target in zip(logits, record["targets"]))
    expected.backward()
    assert loss.item() == pytest.approx(expected.item(), rel = 1e-4)
    got = dict(tree_flatten(grads["head"]))
    assert not mx.any(grads["encoder"]["language_model"]["lm_head"]["weight"]).item()
    for name, parameter in reference.named_parameters():
        np.testing.assert_allclose(np.array(got[name]), parameter.grad.numpy(), atol = 2e-4, rtol = 2e-3, err_msg = name)


@pytest.mark.parametrize("encoder_lr, head_lr", [(1e-2, 0.0), (0.0, 1e-2)])
def test_clef_trains_decoder_adapters_and_head_at_their_own_rates(clef, encoder_lr, head_lr):
    pipeline, _, item = clef
    items = [item("hello " * 20), item("bye", count = 1)]
    network = clef_training_network(pipeline, r = 4, lora_alpha = 4)
    before, first = _parameters(network), sum(_clef_record_loss(network, record).item() for record in items) / 3
    adapted = {name.split(".")[-2] for name in before if name.endswith("lora_a")}
    assert {"q_proj", "in_proj_qkv", "down_proj"} <= adapted and not adapted & {"in_proj_a", "in_proj_b", "lm_head"}
    args = MLXTrainingConfig(per_device_train_batch_size = 2, max_steps = 3, learning_rate = encoder_lr, warmup_steps = 0, logging_steps = 1, compile = False)
    trainer = MLXDecisionTrainer(network, args, items, eval_dataset = items, head_learning_rate = head_lr)
    trainer.train()
    moved = {name.split(".")[0] for name, value in _parameters(network).items() if not np.array_equal(value, before[name])}
    assert moved == ({"encoder"} if encoder_lr else {"head"}) and network.training
    assert trainer.state.log_history[0]["loss"] == pytest.approx(first, rel = 1e-3)
    assert trainer.evaluate()["eval_loss"] == pytest.approx(sum(_clef_record_loss(network, record).item() for record in items) / 3, rel = 1e-3)
    assert [len(row) for row in clef_logits(network, items)[0]] == [3, 2]


def _reference_loss(rows, ordinal, smoothing, brier, weight):
    # The loss of unsloth's torch DecisionTrainer, one decision (its logits and target) at a time.
    losses, apart = [], []
    for z, t in ((torch.as_tensor(np.array(z), dtype = torch.float32), torch.as_tensor(np.array(t), dtype = torch.float32)) for z, t in rows):
        p, levels = torch.softmax(z, 0), torch.arange(len(z)).float()
        losses.append(-(((1 - smoothing) * t + smoothing / len(z)) * p.log()).sum() + brier * ((p - t) ** 2).sum())
        apart.append((p[:, None] * (levels[:, None] - levels[None, :]).abs() * t[None, :]).sum() / max(len(z) - 1, 1) * ordinal[len(apart)])
    return (sum(losses) / len(losses) + weight * sum(apart) / max(sum(ordinal), 1)).item()


@pytest.mark.parametrize("objective", [(0.1, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.7), (0.1, 0.5, 0.7)])
def test_decision_objectives_match_the_torch_loss(clef, objective):
    scored = {"grade": {"type": "score", "instructions": "how good", "criteria": ["bad", "fine", "good", "great"]}, "ok": {"type": "noul", "instructions": "fine?"}, "rank": {"type": "score", "instructions": "rank", "criteria": ["low", "mid", "high"]}}
    items = [clef[2]("bye"), {**clef_training_item(clef[0], "hello there", scored), "targets": [[0.0, 0.1, 0.9, 0.0], [0.3, 0.7], [0.2, 0.8, 0.0]]}, clef[2]("hi", count = 1)]
    network, kinds = ClefNetwork(clef[0]), [kind == 2 for record in items for kind in record["types"]]
    want = _reference_loss(zip((row for rows in clef_logits(network, items) for row in rows), (t for record in items for t in record["targets"])), kinds, *objective)
    with network.training_run():
        loss, grads = network.decision_step(objective = objective)(items)
    assert loss.item() == pytest.approx(want, rel = 2e-3) and any(mx.any(value).item() for _, value in tree_flatten(grads["head"]))
    trainer = MLXDecisionTrainer(network, _config(per_device_eval_batch_size = 3), items, items, label_smoothing = objective[0], brier_weight = objective[1], ordinal_weight = objective[2])
    assert trainer.evaluate()["eval_loss"] == pytest.approx(want, rel = 2e-3)


def test_clef_kl_penalty_holds_on_to_the_starting_model_and_fields_are_permuted_in_training(clef):
    import random

    pipeline, questions = clef[0], {"grade": {"type": "score", "instructions": "how good", "criteria": ["bad", "fine", "good", "great"]}, "ok": {"type": "noul", "instructions": "fine?"}, "pick": {"type": "choice", "instructions": "which", "criteria": {"a": "x", "b": "y", "c": "z"}}}
    record = {**clef_training_item(pipeline, "hello there", questions), "targets": [[0.0, 0.1, 0.9, 0.0], [0.3, 0.7], [0.2, 0.8, 0.0]]}
    with pytest.raises(NotImplementedError, match = "LoRA Clef"):
        MLXDecisionTrainer(ClefNetwork(pipeline), _config(), [record], kl_weight = 2.0)
    network = clef_training_network(pipeline, r = 4, lora_alpha = 4)
    started = clef_logits(network, [record])[0]
    trainer = MLXDecisionTrainer(network, _config(), [record], [record], kl_weight = 2.0)
    plain = lambda: MLXDecisionTrainer(network, _config(), [record], [record]).evaluate()["eval_loss"]
    assert trainer.evaluate()["eval_loss"] == pytest.approx(plain(), rel = 1e-3)
    network.update(tree_map(lambda value: value + 0.3 * mx.random.normal(value.shape).astype(value.dtype), network.trainable_parameters()))
    log = lambda rows: [torch.log_softmax(torch.as_tensor(row), 0) for row in rows]
    kl = sum((ref.exp() * (ref - now)).sum().item() for ref, now in zip(log(started), log(clef_logits(network, [record])[0]))) / 3
    assert kl > 1e-2 and trainer.evaluate()["eval_loss"] - plain() == pytest.approx(2.0 * kl, rel = 2e-2)
    with network.training_run():
        steps = [network.decision_step(reference = reference)([record]) for reference in (trainer._reference, None)]
    assert any(not mx.allclose(with_kl, without, rtol = 1e-2, atol = 1e-4).item() for (_, with_kl), (_, without) in zip(*(tree_flatten(grads["head"]) for _, grads in steps)))
    assert steps[0][0].item() - steps[1][0].item() == pytest.approx(2.0 * kl, rel = 0.1) and all(module.scale for _, module in network.encoder.named_modules() if "lora_a" in module)
    # The item as clef_training_item builds it is permutable; each question keeps its targets and options whatever order the epoch draws.
    orders = set()
    for epoch in range(8):
        shuffled = network.permuted_item(record, random.Random(f"0-{epoch}-0"))
        orders.add(tuple(shuffled["types"]))
        assert [len(target) for target in shuffled["targets"]] == [len(spans) for spans in shuffled["option_spans"]] == [{2: 4, 0: 2, 1: 3}[kind] for kind in shuffled["types"]]
    draws = []
    object.__setattr__(network, "permuted_item", lambda row, rng: draws.append(rng.random()) or row)
    for _ in range(2):
        MLXDecisionTrainer(network, _config(per_device_train_batch_size = 1, gradient_accumulation_steps = 1, max_steps = 2, seed = 5), [record], [record], permute_fields = True).train()
    # Training only, a new order each epoch, and the same ones in a run with the same seed.
    assert len(orders) > 1 and len(draws) == 4 and draws[:2] == draws[2:] and draws[0] != draws[1]


def test_evaluation_weighs_records_alike_and_hands_metrics_logits_and_soft_targets(clef):
    pipeline, network, seen = clef[0], ClefNetwork(clef[0]), []
    short = {**clef_training_item(pipeline, "x", {"ok": {"type": "noul", "instructions": "fine?"}}), "targets": [[1.0, 0.0]]}
    items = [clef[2]("hello"), short]
    def metrics(prediction):
        seen.append(prediction)
        return {"hits": float((prediction[0].argmax(-1) == prediction[1].argmax(-1)).mean()), "eval_width": prediction[0].shape[1], "loss": -1.0}

    config = _config(per_device_eval_batch_size = 1)
    trainer = MLXDecisionTrainer(network, config, items, items, compute_metrics = metrics, preprocess_logits_for_metrics = lambda logits, labels: 2 * logits)
    got, want = trainer.evaluate(), [_clef_record_loss(network, item).item() / len(item["targets"]) for item in items]
    assert got["eval_loss"] == pytest.approx(sum(want) / 2, rel = 1e-3) and got["eval_width"] == 3 and "eval_hits" in got and got["eval_runtime"] > 0
    logits, flat = seen[0].predictions, [row for rows in clef_logits(network, items) for row in rows]
    np.testing.assert_allclose(seen[0].label_ids, [[0.0, 0.25, 0.75], [1.0, 0.0, 0.0], [1.0, 0.0, -100.0]])
    np.testing.assert_allclose(logits[1, :2], 2 * flat[1], rtol = 2e-3)
    # Past a question's own options: -1e4 inside its batch (doubled here), -100 where a later batch is narrower.
    assert logits[1, 2] == -2e4 and logits[2, 2] == -100 and logits.shape == (3, 3)
    told = SimpleNamespace(on_predict = lambda *_, metrics, **__: seen.append(metrics))
    output = MLXDecisionTrainer(network, config, items, callbacks = [told], preprocess_logits_for_metrics = lambda logits, labels: logits.argmax(-1)).predict(items)
    assert seen[-1] is output.metrics and output.metrics["test_loss"] == pytest.approx(sum(want) / 2, rel = 1e-3) and output.predictions.shape == (3,) and output.label_ids.shape == (3, 3) and trainer.evaluate(items[1:], "held")["held_loss"] == pytest.approx(want[1], rel = 1e-3)


def test_trainer_follows_the_evaluation_and_logging_schedule_it_is_given(checkpoint):
    model, recorder = add_lora_adapters(load_trainable_decision_model(checkpoint[1]), r = 4), _Recorder()
    args = _config(eval_steps = 2, max_steps = 5, per_device_eval_batch_size = 6)
    args.eval_strategy, args.eval_delay, args.logging_strategy = "steps", 3, "epoch"
    trainer = MLXDecisionTrainer(model, args, _items(), _items(), callbacks = [recorder], label_smoothing = 0.2)
    trainer.train()
    history = trainer.state.log_history
    # Two steps an epoch: evaluation waits for step 3 and, as transformers 5 does, takes the last step too; the loss
    # is logged as each epoch ends, the one the run stops in included.
    evaluated = [log for log in history if "eval_loss" in log]
    assert [log["step"] for log in evaluated] == [4, 5][: 1 + _default_flow_evaluates_final_step()] and [log["step"] for log in history if "learning_rate" in log] == [2, 4, 5]
    model.eval()
    assert evaluated[-1]["eval_loss"] == pytest.approx(_soft_cross_entropy(model, collate_decisions(_items(), 0), (0.2, 0.0, 0.0)).item(), rel = 1e-3)
    args = _config(logging_steps = 5)
    args.dataloader_drop_last, args.logging_first_step, args.eval_strategy = True, True, "no"
    trainer = MLXDecisionTrainer(model, args, _items()[:5], _items())
    trainer.train()
    # The item past two full batches is dropped: one step an epoch, the first is logged, and nothing is evaluated.
    assert trainer.state.global_step == 2 and [log["step"] for log in trainer.state.log_history if "learning_rate" in log or "eval_loss" in log] == [1]
    # A second run on the same trainer starts its own count.
    assert trainer.train().global_step == 2 and len(trainer.state.log_history) == 2


def test_clef_full_fine_tune_trains_the_decoder_but_not_the_output_embedding(clef, monkeypatch):
    clef[0].model.vision_tower = Linear(2, 2)
    network, checkpointed = clef_training_network(clef[0], full_finetuning = True, gradient_checkpointing = False), []
    monkeypatch.setattr("unsloth_zoo.mlx.utils.apply_gradient_checkpointing", checkpointed.append)
    with network.training_run():
        names = [name for name, _ in tree_flatten(network.trainable_parameters())]
        _, grads = _clef_record_loss_and_grad(network, clef[2]("hello"))
    assert all(mx.any(grads["encoder"]["language_model"]["model"]["layers"][index]["mlp"]["down_proj"]["weight"]).item() for index in (0, 3))
    assert any(name.startswith("encoder.") for name in names) and not any("lm_head" in name or "vision" in name for name in names) and not checkpointed


def _clef_checkpoint_tensors(decoder):
    # As the real checkpoint differs from the loaded model: its names, channels-first kernels, norm weights stored one lower.
    tensors = {"model.visual.proj.weight": mx.ones((2, 3), mx.bfloat16), "lm_head.weight": decoder.language_model.lm_head.weight}
    for name, value in tree_flatten(decoder.language_model.model.parameters()):
        tensors[f"model.language_model.{name}"] = value.swapaxes(1, 2) if value.ndim == 3 else value - 1 if "norm" in name else value
    return tensors


@pytest.mark.parametrize("mode", ["adapters", "qlora", "embedding", "full"])
def test_saved_clef_holds_the_trained_decoder_and_a_head_with_its_temperature_folded_in(clef, tmp_path, monkeypatch, mode):
    pipeline, _, item = clef
    source, record, out = _clef_checkpoint_tensors(pipeline.model), item("hello"), tmp_path / "out"
    mx.save_safetensors(str(tmp_path / "model.safetensors"), source)
    out.mkdir()
    stale = [out / "model-00001-of-00002.safetensors", out / "model.safetensors.index.json"]
    [path.write_text("{}") for path in stale]
    if mode == "qlora":
        with pytest.raises(ValueError, match = "load_in_4bit"):
            load_decision_model(tmp_path, family = "laya", load_in_4bit = True)
        pipeline = load_decision_model(tmp_path, load_in_4bit = True)
        assert "scales" in pipeline.model.language_model.model.layers[0].linear_attn.in_proj_qkv
        from unsloth_zoo.mlx.decision import _QwenModel

        asked, loaded = {}, [_decoder(), _decoder()]
        monkeypatch.setattr("unsloth_zoo.mlx.loader.FastMLXModel.from_pretrained", lambda *args, **kwargs: asked.update(kwargs) or (loaded[kwargs["load_in_4bit"]], None))
        for four in (False, True):
            _QwenModel._load(SimpleNamespace(), tmp_path, None, None, None, None, four)
        assert (asked["load_in_4bit"], asked["load_in_16bit"]) == (True, False)
        # What the loader leaves in 16-bit for other trainers, the embedding and the output head, is quantized too.
        assert [["scales" in module for module in (model.language_model.model.embed_tokens, model.language_model.lm_head)] for model in loaded] == [[False, False], [True, True]]
    network = clef_training_network(pipeline, full_finetuning = mode == "full", r = 4, lora_alpha = 8, modules_to_save = ["embed_tokens"] if mode == "embedding" else None)
    if mode == "qlora":
        # Adapters over quantized layers receive gradients.
        grads = dict(tree_flatten(_clef_record_loss_and_grad(network, record)[1]))
        assert any(name.endswith("lora_b") and mx.any(value).item() for name, value in grads.items())
    trainable = pipeline.model.trainable_parameters()
    pipeline.model.update(tree_map(lambda value: value + 0.05 * mx.random.normal(value.shape).astype(value.dtype), trainable))
    adapters = [module for _, module in pipeline.model.named_modules() if "lora_a" in module]
    whole = {"model.language_model." + key.split(".", 2)[2] for key, _ in tree_flatten(trainable) if "lora_" not in key}
    args = record["input_ids"], record["question_spans"], record["option_spans"], record["types"]
    before = np.array(pipeline.logits(*args))
    # Scoring runs in generation mode, which swaps a quantized output head's class.
    assert [len(row) for row in clef_logits(network, [record])[0]] == [3, 2]
    save_clef_model(pipeline, out, tmp_path, {"head_temperature": 2.0, "temperature": [1.0, 1.0, 40.0]})
    saved, name = mx.load(str(out / "model.safetensors")), "model.language_model.layers.0.linear_attn.{}.weight"
    moved, trained = {key for key in source if not mx.array_equal(saved[key], source[key])}, _clef_checkpoint_tensors(pipeline.model)
    assert not any(path.exists() for path in stale) and len(whole) == {"adapters": 0, "qlora": 0, "embedding": 1}.get(mode, len(source) - 2)
    assert saved.keys() == source.keys() and all(saved[key].shape == source[key].shape and saved[key].dtype == source[key].dtype for key in source)
    assert len(moved) == len(adapters) + len(whole) and whole <= moved
    for key in whole:
        assert mx.allclose(saved[key].astype(mx.float32), trained[key].astype(mx.float32), atol = 2e-2).item(), key
    if adapters:
        low = dict(pipeline.model.named_modules())["language_model.model.layers.0.linear_attn.in_proj_qkv"]
        delta = np.array(saved[name.format("in_proj_qkv")].astype(mx.float32) - source[name.format("in_proj_qkv")].astype(mx.float32))
        assert np.abs(delta - np.array(low.scale * low.lora_b.T @ low.lora_a.T)).max() < 2e-2
    head = mx.load(str(out / "joint_head.safetensors"))
    assert {str(value.dtype).rsplit(".", 1)[-1] for value in head.values() if value.ndim} == {"bfloat16"} and head["residual_gate"].dtype == mx.float32
    pipeline.head = _load_joint_head(out)
    np.testing.assert_allclose(np.array(pipeline.logits(*args)), before / 2, atol = 3e-2)
    config = json.loads((out / "unsloth_decision_config.json").read_text())
    assert config["folded_temperature"] == 2.0 and "head_temperature" not in config and config["fine_tuned"] is True
    assert load_decision_model(out).temperatures == {"choice": 1.0, "score": 1.0, "noul": 5.0}
    # A temperature the head's logit scales cannot absorb stays in the config and is applied when serving.
    save_clef_model(pipeline, out, tmp_path, {"head_temperature": 0.001, "temperature": [1.0, 1.0, 40.0]})
    assert load_decision_model(out).temperatures == pytest.approx({"choice": 0.001, "score": 0.001, "noul": 0.005})
    with pytest.raises(ValueError, match = "saved over"):
        save_clef_model(pipeline, tmp_path / "out" / ".." , tmp_path)


@pytest.mark.parametrize("four_bit", [False, True])
def test_a_clef_saves_as_adapters_that_load_over_their_base_and_go_on_training(clef, tmp_path, four_bit):
    from unsloth_zoo.mlx.decision import save_clef_adapter

    record, base, out = clef[2]("hello"), tmp_path / "base", tmp_path / "out"
    base.mkdir(), out.mkdir()
    pipeline = load_decision_model(tmp_path, load_in_4bit = four_bit)
    mx.save_safetensors(str(base / "model.safetensors"), _clef_checkpoint_tensors(_decoder()))
    (base / "tokenizer.json").write_text("{}"), (base / "config.json").write_text("{}"), (out / "model.safetensors").write_text("stale")
    with pytest.raises(ValueError, match = "through LoRA adapters"):
        save_clef_adapter(pipeline, out, base, "org/base")
    # Adapters on every layer, or on the last one only.
    clef_training_network(pipeline, r = 4, lora_alpha = 8, **({"finetune_last_n_layers": 1} if four_bit else {}))
    trained = pipeline.model.trainable_parameters()
    assert any("layers.0." in name for name, _ in tree_flatten(trained)) != four_bit
    pipeline.model.update(tree_map(lambda value: value + 0.05 * mx.random.normal(value.shape).astype(value.dtype), trained))
    args = record["input_ids"], record["question_spans"], record["option_spans"], record["types"]
    before = np.array(pipeline.logits(*args))
    with pytest.raises(ValueError, match = "saved over"):
        save_clef_adapter(pipeline, base, base, "org/base")
    save_clef_adapter(pipeline, out, base, "org/base", "abc", {"head_temperature": 2.0})
    adapter, tensors = json.loads((out / "adapter_config.json").read_text()), mx.load(str(out / "adapter_model.safetensors"))
    assert (adapter["base_model_name_or_path"], adapter["revision"], adapter["r"], adapter["lora_alpha"]) == ("org/base", "abc", 4, 8.0)
    assert sorted(item.name for item in out.iterdir()) == ["adapter_config.json", "adapter_model.safetensors", "joint_head.safetensors", "joint_head_config.json", "tokenizer.json", "unsloth_decision_config.json"]
    low = dict(pipeline.model.named_modules())["language_model.model.layers.3.mlp.down_proj"]
    name = "base_model.model.model.language_model.layers.3.mlp.down_proj.lora_{}.weight"
    assert mx.array_equal(tensors[name.format("A")], low.lora_a.T).item() and mx.array_equal(tensors[name.format("B")], low.lora_b.T).item()
    served = load_decision_model(out, base_model = base, load_in_4bit = four_bit)
    assert served.base_folder == base and not tree_flatten(served.model.trainable_parameters())
    np.testing.assert_allclose(np.array(served.logits(*args)), before / 2, atol = 3e-2)
    with pytest.raises(ValueError, match = "trains through"):
        clef_training_network(served, full_finetuning = True)
    network = clef_training_network(served, r = 64)
    again = dict(tree_flatten(served.model.trainable_parameters()))
    assert again.keys() == dict(tree_flatten(trained)).keys() and again["language_model.model.layers.3.mlp.down_proj.lora_a"].shape[1] == 4
    grads = dict(tree_flatten(_clef_record_loss_and_grad(network, record)[1]))
    assert mx.any(grads["encoder.language_model.model.layers.3.mlp.down_proj.lora_a"]).item()
    # A dropout the adapters trained with is saved and comes back with them, off while the model serves.
    for _, module in pipeline.model.named_modules():
        if "lora_a" in module:
            module.dropout = Dropout(0.25)
    save_clef_adapter(pipeline, out, base, "org/base")
    assert json.loads((out / "adapter_config.json").read_text())["lora_dropout"] == 0.25
    reloaded = [module.dropout for _, module in load_decision_model(out, base_model = base, load_in_4bit = four_bit).model.named_modules() if "lora_a" in module]
    assert reloaded and all(dropout._p_1 == pytest.approx(0.75) and not dropout.training for dropout in reloaded)
    # An adapter for a module the base does not have, or for the output embedding the joint head reads, is not put on it.
    for stem, refusal in (("layers.9.mlp.down_proj", "of which the base model has"), ("lm_head", "output embedding")):
        mx.save_safetensors(str(out / "adapter_model.safetensors"), {key.replace("model.language_model.layers.3.mlp.down_proj", stem): value for key, value in tensors.items()})
        with pytest.raises(ValueError, match = refusal):
            load_decision_model(out, base_model = base)
    # A save over that checkpoint which stops part way leaves a folder that no longer reads as a Clef.
    from unsloth_zoo.mlx import decision

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(decision.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
        with pytest.raises(OSError, match = "interrupted"):
            save_clef_adapter(pipeline, out, base, "org/base")
    assert not decision.ClefModel.matches(out)


def test_a_plain_language_model_gets_a_new_joint_head_and_saves_as_a_clef(clef, tmp_path, monkeypatch):
    from unsloth_zoo.mlx.decision import ClefModel, _decoder_tensor_name, clef_head_config, load_language_model_as_clef

    source, record, out = tmp_path / "lm", clef[2]("hello"), tmp_path / "out"
    source.mkdir()
    mx.save_safetensors(str(source / "model.safetensors"), _clef_checkpoint_tensors(clef[0].model))
    assert [clef_head_config(size)[key] for size in (3071, 3072) for key in ("width", "heads", "feedforward")] == [512, 8, 2048, 1024, 16, 4096] and _decoder_tensor_name({}, "model.layers.0.mlp.up_proj.weight") == "model.layers.0.mlp.up_proj.weight"
    first, again, other = (ClefModel.from_language_model(source, head_width = 128, seed = seed) for seed in (1, 1, 2))
    assert first.head_config == {"hidden_size": 64, "width": 128, "routing_layers": 2, "layers": 4, "heads": 2, "feedforward": 512}
    weights = [dict(tree_flatten(pipeline.head.parameters())) for pipeline in (first, again, other)]
    assert all(mx.array_equal(weights[0][name], weights[1][name]) for name in weights[0]) and not mx.array_equal(weights[0]["question_projection.weight"], weights[2]["question_projection.weight"])
    assert not weights[0]["residual_gate"].item() and 0.8 < weights[0]["type_embedding.weight"].std().item() < 1.2 and all(mx.any(value[:256]).item() and mx.any(value[256:]).item() for name, value in weights[0].items() if name.endswith("in_proj_weight"))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"lm_head.weight": mx.zeros((512, 64))})
    with pytest.raises(ValueError, match = "holds no tensor"):
        ClefModel.from_language_model(tmp_path)
    with pytest.raises(ValueError, match = "hidden size"):
        ClefModel.from_language_model(source, head_config = {**first.head_config, "hidden_size": 32})
    args = record["input_ids"], record["question_spans"], record["option_spans"], record["types"]
    (source / "joint_head_config.json").write_text(json.dumps({**first.head_config, "width": 64}))
    save_clef_model(first, out, source, {"base_model": "org/lm"})
    served = load_decision_model(out)
    assert served.head_config == first.head_config and json.loads((out / "unsloth_decision_config.json").read_text())["base_model"] == "org/lm"
    np.testing.assert_allclose(np.array(served.logits(*args)), np.array(first.logits(*args)), atol = 3e-2)
    packed = _decoder(quantized = True)
    monkeypatch.setattr(ClefModel, "_load", lambda self, *args: vars(self).update(model = packed, tokenizer = None, asked = args[-1]))
    # Quantized on load, a float source still takes the trained update.
    assert load_language_model_as_clef(source, load_in_4bit = True).asked is True
    floats = mx.load(str(source / "model.safetensors"))
    # A source that keeps the names but holds integers, or other rows, cannot take the update.
    for other in (lambda value: value.astype(mx.uint8), lambda value: mx.zeros((value.shape[0] + 1, value.shape[1]))):
        mx.save_safetensors(str(source / "model.safetensors"), {name: other(value) if value.ndim == 2 else value for name, value in floats.items()})
        with pytest.raises(ValueError, match = "holds no tensor"):
            ClefModel.from_language_model(source, load_in_4bit = True)
    # An MLX-quantized source, under the names it is loaded with, trains through adapters and saves requantized.
    stored, name = dict(tree_flatten(packed.parameters())), "language_model.model.layers.2.mlp.down_proj"
    # A weight's scales may be in another file than the weight.
    apart = {key: stored[key] for key in stored if key.startswith(name) and not key.endswith(".weight")}
    mx.save_safetensors(str(source / "model.safetensors"), {key: stored[key] for key in stored if key not in apart})
    mx.save_safetensors(str(source / "model-00002-of-00002.safetensors"), apart)
    # A module with a quantization entry of its own does not take the mode of the rest.
    own = {key[: -len(".scales")]: {"group_size": 64, "bits": 8} for key in stored if key.endswith(".scales")}
    config = json.loads((source / "config.json").read_text()) if (source / "config.json").exists() else {}
    (source / "config.json").write_text(json.dumps({**config, "quantization": {"mode": "mxfp4", **own}}))
    pipeline = ClefModel.from_language_model(source, head_width = 128)
    with pytest.raises(ValueError, match = "adapters only"):
        clef_training_network(pipeline, full_finetuning = True)
    clef_training_network(pipeline, r = 4, lora_alpha = 8)
    low = dict(packed.named_modules())[name]
    low.lora_b = mx.random.normal(low.lora_b.shape) * 0.05
    save_clef_model(pipeline, out, source)
    saved = {**mx.load(str(out / "model.safetensors")), **mx.load(str(out / "model-00002-of-00002.safetensors"))}
    weights = [mx.dequantize(*(tensors[f"{name}.{leaf}"] for leaf in ("weight", "scales", "biases")), group_size = 64, bits = 8) for tensors in (stored, saved)]
    assert saved.keys() == stored.keys() and all(saved[key].dtype == stored[key].dtype and saved[key].shape == stored[key].shape for key in stored)
    assert np.abs(np.array(weights[1] - weights[0] - low.scale * low.lora_b.T @ low.lora_a.T)).max() < 2e-2
    assert sum(not mx.array_equal(saved[key], stored[key]) for key in stored if key.endswith(".weight")) == len([module for _, module in packed.named_modules() if "lora_a" in module])


def test_clef_prompt_gives_up_the_end_of_the_state_only(clef):
    from unsloth_zoo.mlx.decision import DecisionRequestError

    state = "".join(chr(97 + index % 23) for index in range(240))
    whole, cut = clef[2](state), clef[2](state, 800)
    removed = len(whole["input_ids"]) - 800
    assert removed > 0 and len(cut["input_ids"]) == 800 and cut["input_ids"][-50:] == whole["input_ids"][-50:] and cut["input_ids"][:200] == whole["input_ids"][:200]
    assert cut["option_spans"][1][1] == tuple(edge - removed for edge in whole["option_spans"][1][1])
    with pytest.raises(DecisionRequestError, match = "before the state"):
        clef[2]("state", 100)


def _vision_decoder():
    from mlx_vlm.models import qwen3_5

    text = dict(model_type = "qwen3_5_text", hidden_size = 64, intermediate_size = 128, num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1, vocab_size = 512, head_dim = 32)
    text.update(linear_num_value_heads = 4, linear_num_key_heads = 2, linear_key_head_dim = 32, linear_value_head_dim = 16, linear_conv_kernel_dim = 4, rms_norm_eps = 1e-6, max_position_embeddings = 4096)
    vision = dict(model_type = "qwen3_5", depth = 1, hidden_size = 32, intermediate_size = 64, num_heads = 2, out_hidden_size = 64, num_position_embeddings = 64, patch_size = 16)
    tokens = dict(image_token_id = 500, vision_start_token_id = 501, vision_end_token_id = 502)
    mx.random.seed(7)
    return qwen3_5.Model(qwen3_5.ModelConfig.from_dict({"model_type": "qwen3_5", "text_config": text, "vision_config": vision, **tokens}))


def _patches(text, images, return_tensors):
    # As the Qwen processor answers: one placeholder per 2x2 patches, and the patches of every image in one array.
    if any(image.width < 32 for image in images):
        raise ValueError("too thin")
    grids = [(1, image.height // 16, image.width // 16) for image in images]
    ids = [token for _, rows, columns in grids for token in (501, *[500] * (rows * columns // 4), 502)] + [10] * text[0].endswith("\n")
    pixels = [np.full((rows * columns, 1536), np.asarray(image).mean() / 255, np.float32) for image, (_, rows, columns) in zip(images, grids)]
    return {"input_ids": np.array([ids]), "pixel_values": np.concatenate(pixels), "image_grid_thw": np.array(grids)}


def test_clef_reads_images_between_the_opening_and_the_state(clef, monkeypatch):
    from PIL import Image

    from unsloth_zoo.mlx.decision import ClefModel, DecisionRequestError, DecisionUnsupportedError

    text_only, questions = clef[0], {"ok": {"type": "noul", "instructions": "fine?"}}
    images = [Image.new("RGB", size, shade) for size, shade in (((96, 64), "black"), ((64, 64), "white"))]
    with pytest.raises(DecisionUnsupportedError):
        text_only.encode_images(images)
    model = _vision_decoder()
    model._processor = _patches
    reader = copy.copy(text_only)
    reader.model = model
    assert reader.reads_images and not text_only.reads_images
    monkeypatch.setattr(ClefModel, "_IMAGE_TOKENS", 15)
    with pytest.raises(DecisionRequestError, match = "the images take 15 tokens"):
        reader.encode_images(images)
    monkeypatch.setattr(ClefModel, "_IMAGE_TOKENS", 16)
    with pytest.raises(DecisionRequestError, match = "could not be read: too thin"):
        reader.encode_images([Image.new("RGB", (16, 16))])

    parsed = reader._parse_questions(questions)
    image_ids, media = reader.encode_images(images)
    plain, with_images = reader.encode("state", parsed), reader.encode("state", parsed, image_ids = image_ids)
    opening = len(reader._encode(next(reader._pieces("state", parsed))[0]))
    assert image_ids == [501, *[500] * 6, 502, 501, *[500] * 4, 502, 10] and media["image_grid_thw"].tolist() == [[1, 4, 6], [1, 4, 4]] and with_images[0] == plain[0][:opening] + image_ids + plain[0][opening:]
    assert with_images[1:] == ([(start + 15, end + 15) for start, end in plain[1]], [[(start + 15, end + 15) for start, end in spans] for spans in plain[2]])
    # Only the state gives way to a length limit, images included in what must fit.
    assert reader.encode("state", parsed, len(with_images[0]) - 2, image_ids)[0] == with_images[0][: opening + 15 + 3] + with_images[0][opening + 15 + 5 :]

    ids = mx.array(with_images[0])[None]
    with generation_mode(model):
        hidden = reader._hidden(with_images[0], media)
        # mlx-vlm's own forward places the image features and positions.
        expected = model(ids, media["pixel_values"], image_grid_thw = media["image_grid_thw"]).logits[0]
        np.testing.assert_allclose(np.array(model.language_model.lm_head(hidden).astype(mx.float32)), np.array(expected.astype(mx.float32)), atol = 1e-4)
        assert mx.abs(hidden - reader._hidden(with_images[0])).max().item() > 1e-2

    def url(image):
        import base64, io

        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    dark, light = (reader.answer([{"role": "user", "content": ["state", {"type": "image_url", "image_url": url(image.resize((96, 64)))}]}], questions) for image in images)
    assert dark["usage"] == light["usage"] and dark["answers"]["ok"]["noul"] != light["answers"]["ok"]["noul"]
    assert dark["usage"]["input_tokens"] == len(reader.encode([{"role": "user", "content": ["state"]}], parsed)[0]) + 9
    model._processor = None
    assert not reader.reads_images


def test_prompts_that_share_their_images_read_them_once(monkeypatch):
    from PIL import Image

    from unsloth_zoo.mlx import utils

    reader = _LabelModel.__new__(_LabelModel)
    reader.model, reader.takes_images = _vision_decoder(), True
    reader.model._processor = _patches
    image_ids, media = reader.encode_images([Image.new("RGB", (96, 64), "gray"), Image.new("RGB", (64, 64), "white")])
    shared = [3, 4, *image_ids, *range(40, 60)]
    prompts = [shared + [7, 8, 9], shared + [300], shared + [301, 302, 303, 304]]
    with generation_mode(reader.model):
        got = list(reader._hidden_states(prompts, media, 2 + len(image_ids)))
        want = [reader._hidden(ids, media) for ids in prompts]
        # Prompts that part inside the images cannot share them.
        calls, forward = [], utils._forward_text_hidden_states
        monkeypatch.setattr(utils, "_forward_text_hidden_states", lambda model, inputs, **kwargs: calls.append(inputs.shape[1]) or forward(model, inputs, **kwargs))
        apart = list(reader._hidden_states(prompts, media, len(shared) + 1))
    assert calls == [len(ids) for ids in prompts] and all(mx.array_equal(a, b) for a, b in zip(apart, want))
    for a, b in zip(got, want):
        assert a.shape == b.shape and mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item() <= 2**-8 * mx.abs(b.astype(mx.float32)).max().item()


def _clef_record_loss_and_grad(network, record):
    return mx.value_and_grad(lambda params: (network.update(params), _clef_record_loss(network, record))[1])(network.trainable_parameters())


class _Recorder:
    def __init__(self, stop_at = None):
        self.logs, self.events, self.stop_at = [], [], stop_at

    def on_log(self, args, state, control, logs = None, **kwargs):
        self.logs.append(logs)

    def on_step_end(self, args, state, control, **kwargs):
        control.should_training_stop = state.global_step == self.stop_at

    def on_train_end(self, args, state, control, **kwargs):
        self.events.append("end")


@pytest.mark.parametrize("encoder_lr, head_lr", [(1e-2, 0.0), (0.0, 1e-2)])
def test_trainer_steps_logs_and_separates_learning_rates(checkpoint, encoder_lr, head_lr):
    model = add_lora_adapters(load_trainable_decision_model(checkpoint[1]), r = 4)
    before, recorder = _parameters(model), _Recorder()
    args = _config(learning_rate = encoder_lr, lr_scheduler_type = "linear")
    trainer = MLXDecisionTrainer(model, args, _items(), _items()[:3], head_learning_rate = head_lr, callbacks = [recorder])
    trainer.train()
    # Six items in pairs is three micro-batches: an epoch is one full step and one step for the odd micro-batch.
    assert trainer.state.global_step == 4 and recorder.events == ["end"]
    steps = [log for log in recorder.logs if "loss" in log]
    np.testing.assert_allclose([log["learning_rate"] for log in steps], [encoder_lr * (1 - i / 4) for i in range(4)], rtol = 1e-5)
    evals = [log["eval_loss"] for log in recorder.logs if "eval_loss" in log]
    model.eval()
    assert len(evals) == 2 and abs(evals[-1] - _soft_cross_entropy(model, collate_decisions(_items()[:3], 0)).item()) < 1e-3
    changed = {name for name, value in _parameters(model).items() if not np.array_equal(value, before[name])}
    assert changed == {name for name, _ in tree_flatten(model.trainable_parameters()) if name.startswith("encoder.") == (encoder_lr > 0)}


def test_a_clef_run_resumed_from_a_checkpoint_ends_where_the_uninterrupted_run_does(clef, tmp_path):
    base, items = tmp_path / "base", [clef[2](text) for text in ("hello", "a longer state", "x", "yes or no")]
    base.mkdir()
    mx.save_safetensors(str(base / "model.safetensors"), _clef_checkpoint_tensors(_decoder()))
    (base / "tokenizer.json").write_text("{}"), (base / "config.json").write_text("{}")

    def run(output, resume = None):
        mx.random.seed(3)
        network = clef_training_network(load_decision_model(tmp_path), r = 4, lora_alpha = 4, origin = (base, "org/base", None, {"head_temperature": 2.0}))
        args = _config(output_dir = str(output), gradient_accumulation_steps = 1, save_steps = 1, save_total_limit = 2, learning_rate = 1e-2)
        trainer = MLXDecisionTrainer(network, args, items, items)
        trainer.train(resume_from_checkpoint = resume)
        return trainer, network

    (whole, network), out = run(tmp_path / "whole"), tmp_path / "part"
    assert sorted(item.name for item in (tmp_path / "whole").iterdir()) == ["checkpoint-3", "checkpoint-4"]
    # The third of four steps is the first of the second epoch, so the resumed run starts inside an epoch.
    shutil.copytree(tmp_path / "whole" / "checkpoint-3", out / "checkpoint-3")
    (out / "checkpoint-9").mkdir()
    resumed, again = run(out, True)
    assert resumed.state.global_step == 4 and sorted(item.name for item in out.iterdir()) == ["checkpoint-4", "checkpoint-9"] and [log["step"] for log in resumed.state.log_history] == [log["step"] for log in whole.state.log_history]
    assert [log for log in resumed.state.log_history if log["step"] < 4] == [log for log in whole.state.log_history if log["step"] < 4]
    for (name, value), (_, other) in zip(tree_flatten(network.trainable_parameters()), tree_flatten(again.trainable_parameters()), strict = True):
        # Not bitwise: Metal's results move in the fourth digit when the GPU is shared.
        np.testing.assert_allclose(np.array(other.astype(mx.float32)), np.array(value.astype(mx.float32)), atol = 2e-3, err_msg = name)
    # A checkpoint is the adapters and the head as they are, which also loads to serve.
    saved, record = out / "checkpoint-4", items[0]
    assert json.loads((saved / "unsloth_decision_config.json").read_text())["head_temperature"] == 2.0
    assert all(value.dtype == mx.float32 for value in mx.load(str(saved / "joint_head.safetensors")).values())
    args = record["input_ids"], record["question_spans"], record["option_spans"], record["types"]
    np.testing.assert_allclose(np.array(load_decision_model(saved, base_model = base).logits(*args)), np.array(again._pipeline.logits(*args)), atol = 3e-2)
    with pytest.raises(ValueError, match = "trainer_state.json"):
        run(tmp_path / "none", True)
    (tmp_path / "none" / "checkpoint-1").mkdir(parents = True)
    with pytest.raises(ValueError, match = "is not replaced"):
        run(tmp_path / "none")


def test_trainer_keeps_and_returns_to_its_best_checkpoint(checkpoint, tmp_path):
    model, scores = add_lora_adapters(load_trainable_decision_model(checkpoint[1]), r = 4), iter([3.0, 1.0, 2.0])
    args = _config(output_dir = str(tmp_path), num_train_epochs = 3, save_total_limit = 1, load_best_model_at_end = True, metric_for_best_model = "score", greater_is_better = True)
    args.eval_strategy = args.save_strategy = "epoch"
    build = lambda: MLXDecisionTrainer(model, args, _items(), _items(), compute_metrics = lambda prediction: {"score": next(scores)})
    trainer = build()
    trainer.train()
    # The first epoch scored best: its checkpoint outlives the limit beside the newest, and the model ends as it was then.
    assert sorted(item.name for item in tmp_path.iterdir()) == ["checkpoint-2", "checkpoint-6"] and trainer.state.best_model_checkpoint == str(tmp_path / "checkpoint-2")
    best = mx.load(str(tmp_path / "checkpoint-2" / "trainable.safetensors"))
    assert all(mx.array_equal(value, best[name]).item() for name, value in tree_flatten(model.trainable_parameters())) and len(best) > 4
    args.save_strategy = "best"
    with pytest.raises(NotImplementedError, match = "save_strategy"):
        MLXDecisionTrainer(model, args, _items(), _items()).train()
    # Saved every step but evaluated every epoch, the best evaluation could be of a state no checkpoint holds.
    args.save_strategy, args.save_steps = "steps", 1
    with pytest.raises(ValueError, match = "load_best_model_at_end needs"):
        MLXDecisionTrainer(model, args, _items(), _items()).train()
    # Evaluated every step and saved every other: the best step is saved although it is off the interval.
    args.eval_strategy, args.eval_steps, args.save_steps, args.output_dir, scores = "steps", 1, 2, str(tmp_path / "steps"), iter([3.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    build().train()
    assert sorted(item.name for item in (tmp_path / "steps").iterdir()) == ["checkpoint-1", "checkpoint-6"]


def test_trainer_takes_datasets_and_fractional_intervals(checkpoint):
    from datasets import Dataset

    recorder = _Recorder()
    args = _config(eval_steps = 0.5, logging_steps = 0.5)
    data = Dataset.from_list(_items())
    MLXDecisionTrainer(load_trainable_decision_model(checkpoint[1]), args, data, data.select(range(3)), callbacks = [recorder]).train()
    assert [("eval_loss" in log, round(log["epoch"], 2)) for log in recorder.logs[:4]] == [(False, 1.0), (True, 1.0), (False, 2.0), (True, 2.0)]


def _metal_limits():
    limits = mx.set_memory_limit(1 << 40), mx.set_wired_limit(0), mx.set_cache_limit(0)
    mx.set_memory_limit(limits[0]), mx.set_wired_limit(limits[1]), mx.set_cache_limit(limits[2])
    return limits


def test_trainer_applies_memory_limits_for_the_run_only(checkpoint):
    model = load_trainable_decision_model(checkpoint[1])
    before, cap = _metal_limits(), int(mx.device_info()["max_recommended_working_set_size"] / 1e9 * 0.85 * 1e9)
    cases = (({}, (cap, cap)), ({"wired_limit_gb": 1, "cache_limit_gb": 2}, (cap, 10**9, 2 * 10**9)), ({"disable_memory_limits": True}, before))
    for kwargs, during in cases:
        seen = []
        trainer = MLXDecisionTrainer(model, _config(max_steps = 1, **kwargs), _items())
        trainer._event = lambda *args, **kwargs: seen.append(_metal_limits()[: len(during)])
        trainer.train()
        assert {*seen} == {during} and _metal_limits() == before
        assert bool(trainer._memory_limits_applied) == ("disable_memory_limits" not in kwargs)
    with pytest.raises(ValueError):
        MLXDecisionTrainer(model, _config(wired_limit_gb = 1e6), _items()).train()
    assert _metal_limits() == before


def test_trainer_reports_its_logs_to_tensorboard(checkpoint, monkeypatch, tmp_path):
    written = []
    writer = SimpleNamespace(add_scalar = lambda *row: written.append(row), close = lambda: written.append("closed"))
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", SimpleNamespace(SummaryWriter = lambda log_dir: writer))
    config = _config(max_steps = 2, logging_steps = 1, eval_steps = 1, report_to = "tensorboard", output_dir = str(tmp_path))
    trainer = MLXDecisionTrainer(load_trainable_decision_model(checkpoint[1]), config, _items(), _items())
    trainer.train()
    history = trainer.state.log_history
    for tag, key, kind in (("train/loss", "loss", "learning_rate"), ("train/learning_rate", "learning_rate", "learning_rate"), ("eval/loss", "eval_loss", "eval_loss")):
        assert [row for row in written if row[0] == tag] == [(tag, log[key], log["step"]) for log in history if kind in log]
    assert written[-1] == "closed" and len(written) > 6 and trainer._report_to == (None, None)


def test_early_stopping_ends_the_run_when_its_metric_stops_improving(checkpoint):
    model = load_trainable_decision_model(checkpoint[1])

    def run(best = None, threshold = 10.0, **chosen):
        config = _config(max_steps = 6, eval_steps = 1, **chosen)
        accuracy = _Accuracy()
        accuracy.best = best
        # An improvement of 10 never happens, so the first evaluation that has a best to compare with stops the run.
        trainer = MLXDecisionTrainer(model, config, _items(), _items(), callbacks = [accuracy, transformers.EarlyStoppingCallback(1, threshold)])
        trainer.train()
        return trainer

    class _Accuracy(transformers.TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            state.best_metric = self.best

        def on_evaluate(self, args, state, control, metrics = None, **kwargs):
            metrics["eval_accuracy"] = 0.1 * state.global_step

    trainer = run(metric_for_best_model = "eval_loss")
    losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    assert trainer.state.global_step == 2 and trainer.state.best_metric == min(losses) and trainer.args.eval_strategy == "steps"
    # The stop request ends with its run: the same trainer trains again.
    assert trainer.train().global_step == 2
    # A best loss no evaluation reaches is kept, and the first evaluation already fails to improve on it.
    trainer = run(best = 0.0, metric_for_best_model = "eval_loss")
    assert trainer.state.global_step == 1 and trainer.state.best_metric == 0.0
    # The best metric is the one the callback watches, and both read it as higher-is-better: a rising accuracy never stops the run.
    trainer = run(threshold = 0.0, metric_for_best_model = "accuracy")
    assert trainer.state.global_step == 6 and trainer.state.best_metric == pytest.approx(0.6) and trainer.args.greater_is_better is True


def test_trainer_logs_evaluates_saves_and_ends_an_epoch_when_a_callback_asks(checkpoint, tmp_path):
    class _Ask(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_log = state.global_step == 1
            control.should_evaluate = control.should_save = control.should_epoch_stop = state.global_step == 2

    args = _config(gradient_accumulation_steps = 1, logging_steps = 100, output_dir = str(tmp_path))
    args.eval_strategy = args.save_strategy = "no"
    trainer = MLXDecisionTrainer(load_trainable_decision_model(checkpoint[1]), args, _items(), _items(), callbacks = [_Ask()])
    trainer.train()
    history = trainer.state.log_history
    assert [log["step"] for log in history if "learning_rate" in log] == [1] and [log["step"] for log in history if "eval_loss" in log] == [2]
    assert sorted(item.name for item in tmp_path.iterdir()) == ["checkpoint-2"]
    # Three steps an epoch: the first ends after two, so the sixth step is the first of a third epoch.
    assert trainer.state.global_step == 6 and trainer.state.epoch == pytest.approx(2 + 1 / 3)
    # Callbacks step by the run's own intervals, not TrainerState's defaults.
    args = _config(max_steps = 1, eval_steps = 3, save_steps = 4, output_dir = str(tmp_path / "flow"))
    trainer = MLXDecisionTrainer(load_trainable_decision_model(checkpoint[1]), args, _items(), _items(), callbacks = [transformers.DefaultFlowCallback()])
    trainer.train()
    assert (trainer.state.eval_steps, trainer.state.save_steps) == (3, 4)


def test_trainer_clips_gradients_as_mlx_trainer_does(checkpoint, monkeypatch):
    from unsloth_zoo.mlx import trainer as module

    seen = []
    for name in ("_clip_grad_by_value", "_clip_grad_by_leaf_norm", "_clip_grad_norm_fp32"):
        monkeypatch.setattr(module, name, lambda grads, cap, name = name, original = getattr(module, name): seen.append((name, cap)) or original(grads, cap))
    model = load_trainable_decision_model(checkpoint[1])
    # No clip knob is the per-leaf cap of 1.0; a value clamp wins over the others and a leaf cap over a global norm.
    cases = (
        ({"max_grad_norm": 0.0}, ("_clip_grad_by_leaf_norm", 1.0)),
        ({"max_grad_norm": 0.5}, ("_clip_grad_norm_fp32", 0.5)),
        ({"max_grad_norm": 0.5, "max_grad_leaf_norm": 2.0}, ("_clip_grad_by_leaf_norm", 2.0)),
        ({"max_grad_norm": 0.5, "max_grad_value": 0.1}, ("_clip_grad_by_value", 0.1)),
        ({"max_grad_norm": 0.0, "max_grad_leaf_norm": 0.0}, None),
    )
    for kwargs, want in cases:
        seen.clear()
        MLXDecisionTrainer(model, _config(max_steps = 1, **kwargs), _items()).train()
        assert seen == ([want] if want else [])


def test_trainer_stops_on_request_and_refuses_other_optimizers(checkpoint):
    model = load_trainable_decision_model(checkpoint[1])
    recorder = _Recorder(stop_at = 1)
    trainer = MLXDecisionTrainer(model, _config(), _items(), callbacks = [recorder])
    limit = mx.set_cache_limit(123 << 20)
    trainer.train()
    assert mx.set_cache_limit(limit) == 123 << 20
    assert trainer.state.global_step == 1 and recorder.events == ["end"]
    seen = []
    trainer = MLXDecisionTrainer(model, _config(cache_limit_gb = 0), _items())
    trainer._event = lambda *args, **kwargs: seen.append(mx.set_cache_limit(123 << 20))
    limit = mx.set_cache_limit(123 << 20)
    trainer.train()
    assert {*seen, mx.set_cache_limit(limit)} == {123 << 20}
    with pytest.raises(NotImplementedError, match = "sgd"):
        MLXDecisionTrainer(model, _config(optim = "sgd"), _items()).train()


def test_length_grouped_batches_cover_every_item_longest_first():
    lengths = [5, 40, 12, 33, 7, 21, 9, 18, 3]
    batches = _length_grouped_batches(lengths, 2, random.Random(0))
    assert sorted(i for batch in batches for i in batch) == list(range(9))
    assert 1 in batches[0] and [len(batch) for batch in batches] == [2, 2, 2, 2, 1]
