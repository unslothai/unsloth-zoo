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

import importlib.metadata
import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

from datasets import Dataset  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from unsloth_zoo.speculative.capture import run_frozen_target  # noqa: E402
from unsloth_zoo.speculative.eagle3 import (  # noqa: E402
    Eagle3TrainingArguments,
    UnslothEagle3Trainer,
    _aux_capture_ids,
    aux_hidden_states,
    build_eagle3_drafter,
    build_eagle3_vocab_map,
    eagle3_chunked_loss,
    eagle3_loss,
    eagle3_supported,
    load_eagle3_drafter,
    save_eagle3_drafter,
)

VOCAB = 300
LAYERS = 8
DRAFT_VOCAB = 64


def _target(seed = 0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size = VOCAB, hidden_size = 64, intermediate_size = 128, num_hidden_layers = LAYERS,
                      num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 256)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval()


def _drafter(target, seed = 1):
    gen = torch.Generator().manual_seed(seed)
    d2t, t2d = build_eagle3_vocab_map([torch.randint(0, VOCAB, (2000,), generator = gen).tolist()], VOCAB, DRAFT_VOCAB)
    drafter = build_eagle3_drafter(target, d2t, t2d)
    with torch.no_grad():
        for p in drafter.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator = gen))
    return drafter.eval()


def _aux(target, drafter, ids, mask = None):
    cap = run_frozen_target(target, ids, attention_mask = mask, layer_ids = _aux_capture_ids(drafter.aux_layer_ids, LAYERS))
    return aux_hidden_states(cap, drafter.aux_layer_ids), cap


def _speculators():
    root = os.path.join(os.environ.get("WORKSPACE", ""), "speculators")
    if not os.path.isdir(os.path.join(root, "src", "speculators")):
        pytest.skip("vllm-project/speculators checkout not found")
    version = importlib.metadata.version
    importlib.metadata.version = lambda n: "0.0.0" if n == "speculators" else version(n)
    for p in (os.path.join(root, "src"), os.path.join(root, "hs_connectors", "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from speculators.config import SpeculatorsConfig, VerifierConfig
        from speculators.losses import resolve_loss_config
        from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
        from speculators.proposals.greedy import GreedyTokenProposalConfig
    except Exception as e:
        pytest.skip(f"speculators import failed: {e}")
    finally:
        importlib.metadata.version = version
    return {
        "SpeculatorsConfig": SpeculatorsConfig, "VerifierConfig": VerifierConfig,
        "resolve_loss_config": resolve_loss_config, "Eagle3DraftModel": Eagle3DraftModel,
        "Eagle3SpeculatorConfig": Eagle3SpeculatorConfig, "GreedyTokenProposalConfig": GreedyTokenProposalConfig,
    }


def test_supported_targets():
    assert eagle3_supported(Qwen3Config())[0]
    from transformers import Gemma3TextConfig, LlamaConfig
    assert eagle3_supported(LlamaConfig())[0]
    ok, reason = eagle3_supported(Gemma3TextConfig())
    assert not ok and "gemma3" in reason


def test_vocab_map_matches_speculators_rule():
    d2t, t2d = build_eagle3_vocab_map([[5, 5, 5, 9, 9, 2, 7]], target_vocab_size = 20, draft_vocab_size = 6)
    # top by count (5, 9), ties by id (2, 7), then the lowest unused ids (0, 1); sorted.
    chosen = (d2t + torch.arange(6)).tolist()
    assert chosen == [0, 1, 2, 5, 7, 9]
    assert t2d.sum() == 6 and t2d[chosen].all()
    d2t, t2d = build_eagle3_vocab_map([[1]], target_vocab_size = 4, draft_vocab_size = 32000)
    assert d2t.tolist() == [0, 0, 0, 0] and t2d.all()


def test_aux_ids_are_layer_inputs():
    target = _target()
    ids = torch.randint(0, VOCAB, (1, 12))
    cap = run_frozen_target(target, ids, layer_ids = list(range(LAYERS)))
    with torch.no_grad():
        ref = target(ids, output_hidden_states = True).hidden_states
    for k in range(1, LAYERS):
        # hidden_states[k] is the residual stream entering layer k (vLLM / llama.cpp aux id k).
        assert torch.allclose(cap[k - 1], ref[k], atol = 1e-6)


def test_forward_and_loss_match_speculators():
    spec = _speculators()
    target = _target()
    drafter = _drafter(target)
    ids = torch.randint(0, VOCAB, (1, 20))
    aux, _ = _aux(target, drafter, ids)
    cap = run_frozen_target(target, ids, layer_ids = [LAYERS - 1])
    from transformers import LlamaConfig
    layer = drafter.config.to_dict()
    layer.pop("model_type", None)
    lcfg = LlamaConfig(**layer)
    lcfg._attn_implementation = "eager"
    cfg = spec["Eagle3SpeculatorConfig"](
        transformer_layer_config = lcfg, draft_vocab_size = DRAFT_VOCAB, norm_before_residual = False,
        eagle_aux_hidden_state_layer_ids = drafter.aux_layer_ids,
        speculators_config = spec["SpeculatorsConfig"](
            algorithm = "eagle3", default_proposal_method = "greedy",
            proposal_methods = [spec["GreedyTokenProposalConfig"](speculative_tokens = 3)],
            verifier = spec["VerifierConfig"](name_or_path = None, architectures = [])))
    ref = spec["Eagle3DraftModel"](cfg).eval()
    state = {k: v for k, v in drafter.state_dict().items() if not k.startswith("rotary_emb.")}
    state["embed_tokens.weight"] = target.model.embed_tokens.weight.detach()
    ref.load_state_dict(state, strict = False)
    with torch.no_grad():
        ref.verifier_norm.weight.copy_(target.model.norm.weight)
        ref.verifier_lm_head.weight.copy_(target.lm_head.weight[drafter.t2d])
    loss_mask = torch.zeros(1, 20, dtype = torch.bool)
    loss_mask[:, 6:] = True
    seen = []
    hook = ref.lm_head.register_forward_hook(lambda m, a, o: seen.append(o))
    with torch.no_grad():
        _, ref_loss, metrics = ref(
            hidden_states = aux[:, :-1], input_ids = ids[:, 1:], document_ids = torch.zeros(1, 19, dtype = torch.long),
            position_ids = torch.arange(1, 20).unsqueeze(0), ttt_steps = 3, loss_mask = loss_mask[:, 1:],
            verifier_last_hidden_states = cap[LAYERS - 1][:, 1:],
            loss_config = spec["resolve_loss_config"]("kl_div", "eager"))
        ours = drafter.forward_train(aux, ids, None, 3)
        teacher = drafter.target_logits(cap["final"][:, 1:], target.lm_head.weight)
        our_loss, stats = eagle3_loss(ours, teacher, loss_mask[:, 1:])
    hook.remove()
    for k in range(3):
        assert torch.allclose(ours[k], seen[k], atol = 1e-5)
    assert abs(float(our_loss) - float(ref_loss)) < 1e-5
    for k in range(3):
        assert abs(stats["acc"][k] - float(metrics[f"full_acc_{k}_sum"] / metrics[f"full_acc_{k}_total"])) < 1e-5


def test_training_rows_equal_incremental_drafting():
    """Teacher forcing on the drafter's own drafts reproduces draft_greedy step by step."""
    target = _target()
    drafter = _drafter(target)
    prompt = torch.randint(0, VOCAB, (1, 10))
    anchor = prompt.shape[1] - 1
    aux, _ = _aux(target, drafter, prompt)
    drafts = drafter.draft_greedy(aux, prompt, anchor = anchor, num_steps = 3)
    seq = torch.cat([prompt, drafts], dim = 1)
    aux_seq, _ = _aux(target, drafter, seq)
    with torch.no_grad():
        steps = drafter.forward_train(aux_seq, seq, None, 3)
    for k in range(3):
        draft_id = steps[k][0, anchor - 1].argmax(-1)
        assert drafter.draft_to_target(draft_id).item() == drafts[0, k].item()


def test_rows_never_see_the_future_or_padding():
    target = _target()
    drafter = _drafter(target)
    ids = torch.randint(0, VOCAB, (1, 16))
    aux, _ = _aux(target, drafter, ids)
    with torch.no_grad():
        base = drafter.forward_train(aux, ids, None, 3)
        changed = ids.clone()
        changed[0, 12:] = (changed[0, 12:] + 1) % VOCAB
        aux2, _ = _aux(target, drafter, changed)
        moved = drafter.forward_train(aux2, changed, None, 3)
    # Step k row t reads tokens up to t + 1 + k: rows with t + 1 + k < 12 are unchanged.
    for k in range(3):
        last = 12 - 1 - k
        assert torch.allclose(base[k][:, :last], moved[k][:, :last], atol = 1e-5)
    padded = torch.cat([ids, torch.zeros(1, 5, dtype = torch.long)], dim = 1)
    mask = torch.cat([torch.ones(1, 16), torch.zeros(1, 5)], dim = 1).long()
    aux3, _ = _aux(target, drafter, padded, mask)
    with torch.no_grad():
        pad_out = drafter.forward_train(aux3, padded, mask, 3)
    for k in range(3):
        assert torch.allclose(base[k][:, :15 - k], pad_out[k][:, :15 - k], atol = 1e-5)


def test_trainer_moves_only_the_drafter(tmp_path):
    target = _target()
    drafter = _drafter(target).train()
    before_target = {k: v.clone() for k, v in target.state_dict().items()}
    before = {k: v.clone() for k, v in drafter.state_dict().items()}
    rows = [{"input_ids": torch.randint(0, VOCAB, (24,)).tolist()} for _ in range(8)]
    for r in rows:
        r["labels"] = [-100] * 6 + r["input_ids"][6:]

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

    args = Eagle3TrainingArguments(output_dir = str(tmp_path / "out"), per_device_train_batch_size = 4, max_steps = 3,
                                   learning_rate = 1e-2, report_to = "none", save_strategy = "no", logging_steps = 1,
                                   use_cpu = True)
    trainer = UnslothEagle3Trainer(model = drafter, target_model = target, args = args,
                                   train_dataset = Dataset.from_list(rows), processing_class = Tok())
    trainer.train()
    after = drafter.state_dict()
    assert not torch.equal(before["fc.weight"], after["fc.weight"])
    assert not torch.equal(before["layers.0.self_attn.q_proj.weight"], after["layers.0.self_attn.q_proj.weight"])
    assert torch.equal(before["lm_head.weight"], after["lm_head.weight"])
    assert torch.equal(before["d2t"], after["d2t"])
    for k, v in target.state_dict().items():
        assert torch.equal(v, before_target[k]), k
    logged = [h for h in trainer.state.log_history if "eagle3_loss_0" in h]
    assert logged and all(f"eagle3_acc_{k}" in logged[0] for k in range(3))


def test_save_load_round_trip_and_config_contract(tmp_path):
    target = _target()
    drafter = _drafter(target)
    out = tmp_path / "eagle3"
    save_eagle3_drafter(str(out), drafter, "org/fine-tuned-model", speculative_tokens = 3, dtype = torch.float32)
    config = json.loads((out / "config.json").read_text())
    # Fields vLLM's SpeculatorsConfig.validate_speculators_config and update_eagle3 read.
    assert config["speculators_model_type"] == "eagle3"
    assert config["speculators_config"]["proposal_methods"][0]["speculative_tokens"] == 3
    assert config["speculators_config"]["verifier"]["name_or_path"] == "org/fine-tuned-model"
    assert config["transformer_layer_config"]["model_type"] == "llama"
    assert config["transformer_layer_config"]["num_hidden_layers"] == 1
    assert config["draft_vocab_size"] == DRAFT_VOCAB
    assert config["norm_before_residual"] is False and config["norm_before_fc"] is False
    assert config["fc_norm"] is False and config["norm_output"] is False
    assert config["eagle_aux_hidden_state_layer_ids"] == drafter.aux_layer_ids
    from safetensors.torch import load_file
    names = set(load_file(str(out / "model.safetensors")))
    assert "embed_tokens.weight" not in names and {"fc.weight", "lm_head.weight", "d2t", "t2d"} <= names
    loaded = load_eagle3_drafter(str(out), target).eval()
    ids = torch.randint(0, VOCAB, (1, 12))
    aux, _ = _aux(target, drafter, ids)
    with torch.no_grad():
        a, b = drafter.forward_train(aux, ids, None, 2), loaded.forward_train(aux, ids, None, 2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    try:
        from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig
    except Exception:
        return
    SpeculatorsConfig.validate_speculators_config(config)
    vllm_cfg = SpeculatorsConfig.extract_transformers_pre_trained_config(json.loads(json.dumps(config)))
    assert vllm_cfg["architectures"] == ["Eagle3LlamaForCausalLM"] and vllm_cfg["norm_before_residual"] is False


def test_chunked_loss_equals_full_loss_and_gradients():
    """The trainer's chunked, supervised-rows-only loss is the same objective as `eagle3_loss`."""
    target = _target()
    from unsloth_zoo.speculative.capture import resolve_text_parts
    lm = resolve_text_parts(target)["lm_head"].weight
    ids = torch.randint(0, VOCAB, (2, 24))
    mask = torch.ones(2, 24, dtype = torch.long)
    mask[1, 18:] = 0
    labels = ids.clone()
    labels[:, :6] = -100
    loss_mask = (labels[:, 1:] != -100) & mask[:, 1:].bool()
    import copy
    base = _drafter(target).train()
    results = []
    for chunked in (False, True):
        drafter = copy.deepcopy(base)
        aux, cap = _aux(target, drafter, ids, mask)
        if chunked:
            hidden = drafter.forward_train(aux, ids, mask, 3, return_hidden = True)
            loss, stats = eagle3_chunked_loss(drafter, hidden, cap["final"][:, 1:], lm, loss_mask, chunk_rows = 7)
        else:
            logits = drafter.forward_train(aux, ids, mask, 3)
            teacher = drafter.target_logits(cap["final"][:, 1:], lm)
            loss, stats = eagle3_loss(logits, teacher.to(logits[0].dtype), loss_mask)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in drafter.named_parameters() if p.grad is not None}
        results.append((loss.detach(), stats, grads))
    (l0, s0, g0), (l1, s1, g1) = results
    assert torch.allclose(l0, l1, rtol = 1e-5, atol = 1e-6)
    assert s0["count"] == s1["count"]
    assert all(abs(a - b) < 1e-6 for a, b in zip(s0["acc"], s1["acc"]))
    assert g0.keys() == g1.keys() and g0
    for name in g0:
        assert torch.allclose(g0[name], g1[name], rtol = 1e-4, atol = 1e-6), name
