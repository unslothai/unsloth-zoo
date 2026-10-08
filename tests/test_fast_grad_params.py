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

"""unsloth_zoo.fast_grad_params: Trainer's per-step clip / zero_grad over a cached parameter tuple.

Every training test runs the same model twice, UNSLOTH_FAST_GRAD_PARAMS=0 and on, and asserts
BITWISE equal per-microstep losses, logged grad norms and final weights, plus the engagement
counters (FAST_GRAD_CALLS) of the fast arm. Unit tests pin each guard: the cache invalidations,
the Accelerator / generator / flag gates, torch's zero_grad semantics.
"""
import contextlib
import functools
import json
import os
import socket
import subprocess
import sys
import types
import warnings

import pytest
import torch
import torch.nn as nn

from unsloth_zoo import fast_grad_params as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
from peft import LoraConfig, get_peft_model  # noqa: E402

TV = tuple(int(x) for x in transformers.__version__.split(".")[:2])


@contextlib.contextmanager
def _switch(on):
    old = os.environ.get("UNSLOTH_FAST_GRAD_PARAMS")
    os.environ["UNSLOTH_FAST_GRAD_PARAMS"] = "1" if on else "0"
    try:
        yield
    finally:
        os.environ.pop("UNSLOTH_FAST_GRAD_PARAMS", None)
        if old is not None:
            os.environ["UNSLOTH_FAST_GRAD_PARAMS"] = old


def _dataset(n = 96, t = 24, vocab = 256, labels = True):
    from datasets import Dataset
    g = torch.Generator().manual_seed(0)
    ids = [torch.randint(2, vocab, (t,), generator = g).tolist() for _ in range(n)]
    return Dataset.from_dict({"input_ids": ids, "labels": ids} if labels else {"input_ids": ids})


def _llama(lora = True, seed = 0):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype = torch.float32).to(DEV)
    if lora:
        model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "up_proj"]))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 0.05)
    assert F.enable_fast_grad_params(model)
    return model


def _norm_weight(model):
    return dict(model.named_parameters())[next(n for n, _ in model.named_parameters() if n.endswith("model.norm.weight"))]


def _first_lora(model):
    return next(p for n, p in model.named_parameters() if "lora_A" in n)


def _train(model, out, kind = "hf", ga = 1, max_grad_norm = 1.0, steps = 10, events = None, set_to_none = True,
           calls = 1, between = None, **extra):
    """(exact losses per microstep, logged grad norms, final weights). `events[k](model)` runs after
    microstep k's backward; `between(trainer)` runs between `calls` train() calls."""
    from transformers import TrainerCallback, TrainingArguments, Trainer
    if kind == "sft":
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoTokenizer
        base, Args = SFTTrainer, SFTConfig
        tok = AutoTokenizer.from_pretrained(LLAMA)
        tok.pad_token = tok.eos_token
        kw = dict(processing_class = tok, train_dataset = _dataset(labels = False))
        extra.setdefault("max_length", 64)
    else:
        base, Args = Trainer, TrainingArguments
        kw = dict(train_dataset = _dataset())
    events = events or {}

    class T(base):
        def training_step(self, *a, **k):
            loss = super().training_step(*a, **k)
            self.exact.append(loss.item())
            ev = events.get(len(self.exact))
            if ev is not None:
                ev(self.model)
            return loss

    norms = []

    class C(TrainerCallback):
        def on_log(self, args, state, control, logs = None, **k):
            if logs and "grad_norm" in logs:
                norms.append(logs["grad_norm"])

        def on_train_begin(self, args, state, control, model = None, **k):
            if not set_to_none:   # both arms: Trainer's model.zero_grad() with set_to_none=False
                for m in {id(x): x for x in (model, tr.model_wrapped)}.values():
                    zg = m.zero_grad
                    m.__dict__["zero_grad"] = functools.partial(zg, set_to_none = False)

    args = Args(output_dir = str(out), max_steps = steps, logging_steps = 1, per_device_train_batch_size = 2,
                gradient_accumulation_steps = ga, learning_rate = 1e-2, max_grad_norm = max_grad_norm,
                report_to = [], seed = 3, save_strategy = "no", dataloader_num_workers = 0, disable_tqdm = True,
                **extra)
    _ = args.device
    if not torch.distributed.is_initialized():
        args._n_gpu = 1   # no DataParallel when several GPUs are visible
    tr = T(model = model, args = args, callbacks = [C()], **kw)
    tr.exact = []
    for i in range(calls):
        if i:
            between(tr)
        tr.train()
    m = tr.model
    weights = {n: p.detach().clone() for n, p in m.named_parameters()}
    return tr.exact, norms, weights


def _both(build, out, expect_clip = True, **kw):
    """Run off / on arms; assert bitwise equality and engagement. Returns the fast arm's counter deltas."""
    res = {}
    for on in (False, True):
        model = build()
        before = dict(F.FAST_GRAD_CALLS)
        with _switch(on):
            res[on] = _train(model, out / f"arm{int(on)}", **kw)
        res[on] += ({k: F.FAST_GRAD_CALLS[k] - before[k] for k in before},)
    (l0, n0, w0, c0), (l1, n1, w1, c1) = res[False], res[True]
    assert c0["zero_grad"] == 0 and c0["clip"] == 0, c0
    assert c1["zero_grad"] > 0, c1
    assert 0 < c1["rebuild"] <= 2 * kw.get("calls", 1), c1   # once per train(), not per forward / step
    if expect_clip:
        assert c1["clip"] > 0, c1
    assert len(l0) > 0 and l0 == l1, (l0, l1)
    assert (len(n0) > 0) == expect_clip and n0 == n1, (n0, n1)   # no grad_norm logged without clipping
    assert w0.keys() == w1.keys() and all(torch.equal(w0[k], w1[k]) for k in w0)
    return c1


def _freeze_after_backward(model):
    _first_lora(model).requires_grad_(False)   # keeps this microstep's grad: clipped and zeroed


def _unfreeze(model):
    _norm_weight(model).requires_grad_(True)   # not in the optimizer: grads only reach the norm


# ----------------------------------------------------------------------------- Trainer parity
@pytest.mark.parametrize("set_to_none", [True, False])
@pytest.mark.parametrize("max_grad_norm", [1.0, 0.0])
@pytest.mark.parametrize("ga", [1, 3])
@pytest.mark.parametrize("kind", ["hf", "sft"])
def test_trainer_bitwise(tmp_path, kind, ga, max_grad_norm, set_to_none):
    if kind == "sft":
        pytest.importorskip("trl")
    events = {2 * ga: _freeze_after_backward, 4 * ga + 1: _unfreeze}
    c = _both(_llama, tmp_path, kind = kind, ga = ga, max_grad_norm = max_grad_norm, events = events,
              set_to_none = set_to_none, expect_clip = max_grad_norm > 0 or TV >= (5, 17))
    assert c["zero_grad"] >= 10


def test_full_finetune_bitwise(tmp_path):
    _both(lambda: _llama(lora = False), tmp_path, ga = 2, steps = 10)


@pytest.mark.skipif(DEV != "cuda", reason = "fp16 GradScaler needs CUDA")
def test_fp16_grad_scaler_bitwise(tmp_path):
    _both(_llama, tmp_path, ga = 2, fp16 = True)


def test_two_train_calls_new_optimizer_and_adapter(tmp_path):
    """Second train(): optimizer and scheduler recreated, a new adapter added and made active."""
    def between(tr):
        tr.optimizer, tr.lr_scheduler = None, None
        torch.manual_seed(11)
        tr.model.add_adapter("second", LoraConfig(r = 2, lora_alpha = 4, target_modules = ["k_proj", "down_proj"]))
        with torch.no_grad():
            for n, p in tr.model.named_parameters():
                if "lora_B.second" in n:
                    p.normal_(0, 0.05)
        tr.model.set_adapter("second")
    c = _both(_llama, tmp_path, calls = 2, between = between, steps = 6)
    assert c["rebuild"] >= 2


def _moe(stacked):
    """Tiny Qwen3-MoE + expert LoRA. transformers < 5: ModuleList experts, zoo-stacked or per-expert."""
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size = 256, hidden_size = 64, intermediate_size = 128, moe_intermediate_size = 128,
                         num_experts = 8, num_experts_per_tok = 2, num_hidden_layers = 2, num_attention_heads = 4,
                         num_key_value_heads = 2, head_dim = 16, max_position_embeddings = 128,
                         norm_topk_prob = True, mlp_only_layers = [], decoder_sparse_step = 1)
    torch.manual_seed(0)
    model = Qwen3MoeForCausalLM(cfg).to(DEV, torch.bfloat16)
    if not isinstance(model.model.layers[0].mlp.experts, nn.ModuleList):
        lc = LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q_proj", "v_proj"],
                        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
    else:
        lc = LoraConfig(r = 8, lora_alpha = 16, target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lc)
    with torch.no_grad():
        torch.manual_seed(1)
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    old = os.environ.get("UNSLOTH_MOE_STACKED_LORA")
    os.environ["UNSLOTH_MOE_STACKED_LORA"] = "1" if stacked else "0"
    try:
        n = ML.enable_grouped_moe(model, verbose = False, stack_lora = True)
    finally:
        os.environ.pop("UNSLOTH_MOE_STACKED_LORA", None)
        if old is not None:
            os.environ["UNSLOTH_MOE_STACKED_LORA"] = old
    n_stack = sum(ML._STACK_NAME in k for k, _ in model.named_parameters())
    if isinstance(model.base_model.model.model.layers[0].mlp.experts, nn.ModuleList) and DEV == "cuda" and n == 2:
        assert (n_stack > 0) == stacked
    assert F.enable_fast_grad_params(model)
    return model


@pytest.mark.parametrize("stacked", [True, False])
def test_moe_expert_lora_bitwise(tmp_path, stacked):
    if TV >= (5, 0) and not stacked:
        pytest.skip("transformers 5 fused experts: no zoo stacking to toggle")
    try:
        _moe(stacked)
    except TypeError as e:   # PEFT without target_parameters
        pytest.skip(str(e))
    _both(lambda: _moe(stacked), tmp_path, ga = 2, bf16 = DEV == "cuda", events = {5: _unfreeze})


def test_unflagged_model_unchanged(tmp_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(LLAMA).to(DEV)
    assert "zero_grad" not in model.__dict__
    before = dict(F.FAST_GRAD_CALLS)
    _train(model, tmp_path, steps = 2)
    assert F.FAST_GRAD_CALLS["zero_grad"] == before["zero_grad"] and F.FAST_GRAD_CALLS["clip"] == before["clip"]


# ----------------------------------------------------------------------------- DDP
DDP_SCRIPT = r"""
import os, sys, json, torch
sys.path.insert(0, os.environ["TESTS_DIR"])

def worker(rank, world, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = os.environ["PORT"], RANK = str(rank),
                      LOCAL_RANK = str(rank), WORLD_SIZE = str(world))
    torch.cuda.set_device(rank)
    import test_fast_grad_params as T
    from unsloth_zoo import fast_grad_params as F
    import pathlib
    res = {}
    for on in (False, True):
        model = T._llama()
        before = dict(F.FAST_GRAD_CALLS)
        with T._switch(on):
            l, n, w = T._train(model, pathlib.Path(out) / f"r{rank}_{int(on)}", ga = 2, steps = 8,
                               events = {3: T._freeze_after_backward}, ddp_find_unused_parameters = True)
        res[on] = (l, n, w, {k: F.FAST_GRAD_CALLS[k] - before[k] for k in before})
    torch.save(res, os.path.join(out, f"rank{rank}.pt"))

if __name__ == "__main__":
    torch.multiprocessing.spawn(worker, args = (2, sys.argv[1]), nprocs = 2)
"""


def test_ddp_two_gpus_bitwise(tmp_path):
    if DEV != "cuda" or torch.cuda.device_count() < 2:
        pytest.skip("needs 2 CUDA devices")
    script = tmp_path / "ddp.py"
    script.write_text(DDP_SCRIPT)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "TESTS_DIR": os.path.dirname(os.path.abspath(__file__)), "PORT": str(port),
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    p = subprocess.run([sys.executable, str(script), str(tmp_path)], env = env, capture_output = True, text = True)
    assert p.returncode == 0, p.stderr[-4000:]
    for rank in (0, 1):
        res = torch.load(tmp_path / f"rank{rank}.pt")
        (l0, n0, w0, c0), (l1, n1, w1, c1) = res[False], res[True]
        assert c0["zero_grad"] == 0 and c0["clip"] == 0
        assert c1["zero_grad"] >= 8 and c1["clip"] == 8, c1   # the DDP wrapper is flagged too
        assert c1["rebuild"] <= 2, c1
        assert l0 == l1 and n0 == n1 and len(n0) == 8
        assert all(torch.equal(w0[k], w1[k]) for k in w0)


# ----------------------------------------------------------------------------- unit: guards
class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.b.weight.requires_grad_(False)


def _grads(m):
    return {n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()}


def _ref_and_fast():
    torch.manual_seed(0)
    ref = _Net()
    fast = _Net()
    fast.load_state_dict(ref.state_dict())
    assert F.enable_fast_grad_params(fast) and not F._is_flagged(ref)
    return ref, fast


def _backward(m, create_graph = False):
    x = torch.randn(3, 4, generator = torch.Generator().manual_seed(1))
    m.b.weight.requires_grad_(True)
    m.b(m.a(x)).square().sum().backward(create_graph = create_graph)
    m.b.weight.requires_grad_(False)   # frozen after backward, still holds a grad


@pytest.mark.parametrize("set_to_none", [True, False])
@pytest.mark.parametrize("create_graph", [False, True])
def test_zero_grad_matches_torch(set_to_none, create_graph):
    ref, fast = _ref_and_fast()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # create_graph=True warns about the reference cycle
        for m in (ref, fast):
            _backward(m, create_graph)
    for m in (ref, fast):   # a leaf grad that requires grad
        m.a.bias.grad = torch.ones(4, requires_grad = True)
    before = F.FAST_GRAD_CALLS["zero_grad"]
    ref.zero_grad(set_to_none = set_to_none)
    fast.zero_grad(set_to_none = set_to_none)
    assert F.FAST_GRAD_CALLS["zero_grad"] == before + 1
    gr, gf = _grads(ref), _grads(fast)
    assert gr.keys() == gf.keys()
    for k in gr:
        assert (gr[k] is None) == (gf[k] is None), k
        if gr[k] is not None:
            assert torch.equal(gr[k], gf[k]) and gf[k].grad_fn is None and not gf[k].requires_grad
    if not set_to_none:
        assert gf["b.weight"] is not None   # the frozen param's grad was zeroed, not dropped


def test_zero_grad_replica_warning():
    _, fast = _ref_and_fast()
    fast._is_replica = True
    with pytest.warns(UserWarning, match = "nn.DataParallel"):
        fast.zero_grad()


def test_registration_hooks_invalidate():
    """A child module or parameter added outside train() is seen by the next fast call."""
    _, fast = _ref_and_fast()
    c = nn.Linear(4, 4)   # built (its own parameter registrations) before the cache
    fast.zero_grad()
    fast.c = c   # module registration hook
    fast.c.weight.grad = torch.ones(4, 4)
    fast.zero_grad()
    assert fast.c.weight.grad is None
    fast.c.register_parameter("extra", nn.Parameter(torch.ones(2)))   # parameter registration hook
    fast.c.extra.grad = torch.ones(2)
    fast.zero_grad()
    assert fast.c.extra.grad is None


def test_kill_switch_at_runtime():
    _, fast = _ref_and_fast()
    fast.a.weight.grad = torch.ones(4, 4)
    before = dict(F.FAST_GRAD_CALLS)
    with _switch(False):
        fast.zero_grad()
    assert fast.a.weight.grad is None and F.FAST_GRAD_CALLS == before
    with _switch(False):
        assert not F.enable_fast_grad_params(_Net())


def test_declines():
    class Custom(nn.Module):
        def zero_grad(self, set_to_none = True):
            pass
    assert not F.enable_fast_grad_params(Custom())
    off = _Net()
    off.hf_device_map = {"a": 0, "b": "cpu"}
    assert not F.enable_fast_grad_params(off)
    assert not F.enable_fast_grad_params("not a module")
    m = _Net()
    assert F.enable_fast_grad_params(m) and F.enable_fast_grad_params(m)   # idempotent
    assert m.zero_grad.__func__ is F._zero_grad

    class Walk(_Net):
        def named_parameters(self, *a, **k):
            return super().named_parameters(*a, **k)
    assert not F.enable_fast_grad_params(Walk())


def test_pickle_and_deepcopy():
    import copy
    import pickle
    _, fast = _ref_and_fast()
    for clone in (pickle.loads(pickle.dumps(fast)), copy.deepcopy(fast)):
        clone.a.weight.grad = torch.ones(4, 4)
        clone.zero_grad()
        assert clone.a.weight.grad is None
        assert not any(p is q for p in clone.parameters() for q in fast.parameters())
    # Unpickling binds torch's zero_grad; enabling again rebinds the fast one.
    clone = pickle.loads(pickle.dumps(fast))
    assert F.enable_fast_grad_params(clone)
    n = F.FAST_GRAD_CALLS["zero_grad"]
    clone.zero_grad()
    assert F.FAST_GRAD_CALLS["zero_grad"] == n + 1


def test_flat_params_is_parameters_order():
    """Shared / tied params, None entries and a module reached twice: same tuple as parameters()."""
    m = _llama()
    shared = nn.Linear(4, 4)
    m.extra_a, m.extra_b = shared, nn.Sequential(shared, nn.Linear(4, 4))
    m.extra_b[1].weight = shared.weight
    m.extra_b[1].register_parameter("bias", None)
    flat = F._flat_params(m)
    ref = tuple(m.parameters())
    assert len(flat) == len(ref) and all(a is b for a, b in zip(flat, ref))


def test_forward_does_not_invalidate():
    """transformers' `self.layers[:n]` builds a new ModuleList (module registrations) every forward:
    outside the cached tree, so no rebuild."""
    m = _llama()
    F._flat_params(m)
    gen = F._GEN[0]
    ids = torch.randint(2, 200, (1, 8), device = DEV)
    m(input_ids = ids, labels = ids).loss.backward()
    m.zero_grad()
    assert F._GEN[0] == gen
    sub = nn.ModuleList([nn.Linear(2, 2)])   # unrelated module: no bump
    sub.append(nn.Linear(2, 2))
    assert F._GEN[0] == gen


def test_train_call_rebuilds(tmp_path):
    """A direct _parameters edit (no registration hook) between train() calls is picked up by the
    Trainer.train bump alone (prepare_model's bump disabled here)."""
    from accelerate import Accelerator
    import unittest.mock as mock
    model = _llama()
    lin = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"]
    with mock.patch.object(Accelerator, "prepare_model", Accelerator.prepare_model.__wrapped__):
        def between(tr):
            new = nn.Parameter(lin.weight.detach().clone())
            lin.__dict__["_parameters"]["weight"] = new   # bypasses torch's hooks
            tr.optimizer, tr.lr_scheduler = None, None
        _train(model, tmp_path, steps = 2, calls = 2, between = between)
    flat = model.__dict__[F._CACHE][1]
    assert any(p is lin.weight for p in flat)


def test_prepare_model_rebuilds(tmp_path):
    """Same, with Trainer.train unwrapped (a subclass that captured it before the wrap): the
    Accelerator.prepare_model bump inside train() rebuilds."""
    from transformers import Trainer
    import unittest.mock as mock
    model = _llama()
    lin = model.base_model.model.model.layers[0].self_attn.q_proj.lora_A["default"]
    with mock.patch.object(Trainer, "train", Trainer.train.__wrapped__):
        def between(tr):
            lin.__dict__["_parameters"]["weight"] = nn.Parameter(lin.weight.detach().clone())
            tr.optimizer, tr.lr_scheduler = None, None
        _train(model, tmp_path, steps = 2, calls = 2, between = between)
    assert any(p is lin.weight for p in model.__dict__[F._CACHE][1])


def test_stacking_bumps(monkeypatch):
    from unsloth_zoo.temporary_patches import moe_grouped_modulelist as ML
    seen = []
    monkeypatch.setattr(F, "bump", lambda *a: seen.append(1))
    projs = []
    for _ in range(2):
        p = types.SimpleNamespace(lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias = False)}),
                                  lora_B = nn.ModuleDict({"default": nn.Linear(2, 4, bias = False)}))
        projs.append(p)
    ML._stack_projs_lora(projs, "default")
    assert seen


def _accelerator():
    from accelerate import Accelerator
    return Accelerator(cpu = DEV != "cuda")


def _clip_case(acc, params):
    before = F.FAST_GRAD_CALLS["clip"]
    norm = acc.clip_grad_norm_(params, 1.0)
    return F.FAST_GRAD_CALLS["clip"] - before, norm


def test_clip_matches_and_engages():
    acc = _accelerator()
    ref, fast = _ref_and_fast()
    for m in (ref, fast):
        _backward(m)
    n, nf = _clip_case(acc, fast.parameters())
    _, nr = _clip_case(acc, ref.parameters())
    assert n == 1 and torch.equal(nf, nr)
    assert all(torch.equal(a, b) for a, b in zip(_grads(ref).values(), _grads(fast).values()))


def test_clip_generator_gates():
    acc = _accelerator()
    ref, fast = _ref_and_fast()
    _backward(fast)
    _backward(ref)
    started = fast.parameters()
    first = next(started)
    cases = {
        "list": list(fast.parameters()),
        "named": (p for _, p in fast.named_parameters()),
        "no recurse": fast.parameters(recurse = False),
        "child": fast.a.parameters(),   # child module is not flagged
        "unflagged": ref.parameters(),
        "started": started,
        "buffers": fast.buffers(),   # same locals (self, recurse), another generator
    }
    for name, params in cases.items():
        n, _ = _clip_case(acc, params)
        assert n == 0, name
    assert first is not None


@pytest.mark.parametrize("dist_type", ["FSDP", "DEEPSPEED", "XLA", "MEGATRON_LM", "fsdp2", "tp", "tp_plugin"])
def test_clip_other_backends_take_the_original_path(monkeypatch, dist_type):
    from accelerate import Accelerator
    from accelerate.utils import DistributedType
    acc = _accelerator()
    _, fast = _ref_and_fast()
    _backward(fast)
    assert _clip_case(acc, fast.parameters())[0] == 1
    if dist_type == "fsdp2":
        monkeypatch.setattr(Accelerator, "is_fsdp2", property(lambda s: True))
    elif dist_type == "tp":
        monkeypatch.setattr(Accelerator, "parallelism_config",
                            property(lambda s: types.SimpleNamespace(tp_enabled = True)))
    elif dist_type == "tp_plugin":
        monkeypatch.setattr(acc.state, "torch_tp_plugin", object(), raising = False)
    else:
        monkeypatch.setattr(Accelerator, "distributed_type", property(lambda s: getattr(DistributedType, dist_type)))
    seen = []
    orig = F._accelerator_ok
    monkeypatch.setattr(F, "_accelerator_ok", lambda a: seen.append(orig(a)) or seen[-1])
    try:
        n, _ = _clip_case(acc, fast.parameters())
    except Exception:
        n = 0   # the backend's own path needs a real backend; only the routing matters here
    assert n == 0 and seen == [False]


def test_ddp_module_of(tmp_path):
    """A DDP wrapper whose .module is flagged: the generator over the wrapper is recognised."""
    import torch.distributed as dist
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method = f"tcp://127.0.0.1:{port}", rank = 0, world_size = 1)
    try:
        _, fast = _ref_and_fast()
        ddp = nn.parallel.DistributedDataParallel(fast)
        assert F._module_of(ddp.parameters()) is ddp
        assert F._module_of(ddp.module.parameters()) is fast
    finally:
        dist.destroy_process_group()


# ----------------------------------------------------------------------------- Unsloth loaders
UNSLOTH_SCRIPT = r"""
import os, sys, json, torch
import unsloth
from unsloth import FastLanguageModel
from unsloth_zoo import fast_grad_params as F
from transformers import Trainer, TrainingArguments
from datasets import Dataset
out, repo = sys.argv[1], sys.argv[2]
model, tok = FastLanguageModel.from_pretrained(repo, max_seq_length = 64, load_in_4bit = False, dtype = torch.bfloat16)
flag_base = F._is_flagged(model)
model = FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, random_state = 3407,
    target_modules = ["q_proj", "v_proj", "up_proj"], use_gradient_checkpointing = False)
g = torch.Generator().manual_seed(0)
ids = [torch.randint(2, 200, (24,), generator = g).tolist() for _ in range(32)]
ds = Dataset.from_dict({"input_ids": ids, "labels": ids, "attention_mask": [[1] * 24] * 32})
exact = []
class T(Trainer):
    def training_step(self, *a, **k):
        loss = super().training_step(*a, **k)
        exact.append(loss.item())
        return loss
args = TrainingArguments(output_dir = out + "/o", max_steps = 4, per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2, learning_rate = 1e-2, report_to = [], seed = 3, save_strategy = "no",
    logging_steps = 1, bf16 = True, disable_tqdm = True)
_ = args.device
args._n_gpu = 1
before = dict(F.FAST_GRAD_CALLS)
T(model = model, args = args, train_dataset = ds).train()
json.dump({"flag_base": flag_base, "flag_peft": F._is_flagged(model), "losses": exact,
           "calls": {k: F.FAST_GRAD_CALLS[k] - before[k] for k in before}}, open(out + "/info.json", "w"))
"""


def test_unsloth_fast_language_model_path(tmp_path):
    """FastLanguageModel.from_pretrained + get_peft_model flag the model (zoo's loader wrapper);
    the Trainer step runs the fast clip / zero_grad and equals UNSLOTH_FAST_GRAD_PARAMS=0."""
    if DEV != "cuda":
        pytest.skip("Unsloth needs CUDA")
    pytest.importorskip("unsloth")
    from huggingface_hub import snapshot_download
    repo = snapshot_download("trl-internal-testing/tiny-LlamaForCausalLM-3.2", local_dir = str(tmp_path / "tiny"))
    script = tmp_path / "u.py"
    script.write_text(UNSLOTH_SCRIPT)
    res = {}
    for arm in ("1", "0"):
        out = tmp_path / f"arm{arm}"
        out.mkdir()
        env = {**os.environ, "UNSLOTH_FAST_GRAD_PARAMS": arm, "UNSLOTH_COMPILE_LOCATION": str(tmp_path / f"cache{arm}"),
               "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
        p = subprocess.run([sys.executable, str(script), str(out), repo], env = env, cwd = str(tmp_path),
                           capture_output = True, text = True)
        assert p.returncode == 0, p.stderr[-4000:]
        res[arm] = json.load(open(out / "info.json"))
    on, off = res["1"], res["0"]
    assert on["flag_base"] and on["flag_peft"], on
    assert on["calls"]["zero_grad"] >= 4 and on["calls"]["clip"] == 4 and on["calls"]["rebuild"] <= 2, on
    assert off["calls"]["zero_grad"] == 0 and off["calls"]["clip"] == 0, off
    assert on["losses"] == off["losses"] and len(on["losses"]) == 8
