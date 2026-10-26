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

"""MoE routing replay against vLLM's output contracts (0.21 split routes, prompt start offset,
async scheduling), duplicate completions and checkpoint recompute identity. CPU only."""

import ast
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import unsloth_zoo.routing_replay as rr

PAD = 0
LAYERS = 2
TOP_K = 2


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setenv(rr.ROUTING_REPLAY_ENV, "1")
    rr.reset_routing_replay()
    yield rr
    rr.reset_routing_replay()


def _routes(n, seed):
    g = torch.Generator().manual_seed(seed)
    return np.stack([
        np.stack([torch.randperm(8, generator = g)[:TOP_K].numpy() for _ in range(LAYERS)]) for _ in range(n)
    ]).astype(np.int16)


@pytest.mark.parametrize("version, expected", [
    ("0.20.2", None), ("0.21.0", False), ("0.21.1rc0", False), ("0.22.0", None), ("0.32.0", None),
])
def test_vllm_021_turns_async_scheduling_off(replay, monkeypatch, version, expected):
    import importlib.metadata
    monkeypatch.setattr(rr, "_engine_arg_names", lambda: {"enable_return_routed_experts", "async_scheduling"})
    monkeypatch.setattr(importlib.metadata, "version", lambda name: version)
    args = rr.routing_replay_engine_args(SimpleNamespace(num_experts = 8))
    assert args.get("async_scheduling") is expected
    assert args["enable_return_routed_experts"] is True


def test_vllm_021_prompt_routes_on_the_request_are_placed_first(replay):
    # vLLM 0.21: prompt routes on RequestOutput, only generated tokens on the completion.
    prompt, completion = [11, 12, 13], [21, 22, 23]
    full = _routes(6, 0)
    request = SimpleNamespace(
        prompt_token_ids = prompt,
        prompt_routed_experts = full[:3],
        outputs = [SimpleNamespace(token_ids = completion, routed_experts = full[3:5])],
    )
    rr.record_routed_experts([request])
    rows = rr._lookup(tuple(prompt + completion), exact = True)
    assert torch.equal(rows[:5].long(), torch.as_tensor(full[:5]).long())
    assert bool((rows[5] == -1).all())

    rr.reset_routing_replay()
    request.prompt_routed_experts = None
    rr.record_routed_experts([request])
    rows = rr._lookup(tuple(prompt + completion), exact = True)
    assert bool((rows[:3] == -1).all())
    assert torch.equal(rows[3:5].long(), torch.as_tensor(full[3:5]).long())


def test_vllm_021_one_token_completion_keeps_the_prompt_routes(replay):
    # vLLM 0.21: empty generation part is None, prompt part stays (immediate EOS).
    prompt = [11, 12, 13]
    full = _routes(3, 2)
    request = SimpleNamespace(
        prompt_token_ids = prompt,
        prompt_routed_experts = full,
        outputs = [SimpleNamespace(token_ids = [2], routed_experts = None)],
    )
    assert rr.record_routed_experts([request]) == 1
    rows = rr._lookup(tuple(prompt + [2]), exact = True)
    assert torch.equal(rows[:3].long(), torch.as_tensor(full).long())
    assert bool((rows[3] == -1).all())

    rr.reset_routing_replay()
    request.prompt_routed_experts = None
    assert rr.record_routed_experts([request]) == 0


@pytest.mark.parametrize("per_prompt", [False, True])
def test_routed_experts_prompt_start_offsets_the_rows(replay, per_prompt):
    # vLLM >= 0.22 routed_experts_prompt_start = N: returned row 0 is position N.
    prompt, completion = [1, 2, 3], [4, 5]
    full = _routes(5, 1)
    params = SimpleNamespace(routed_experts_prompt_start = 2)
    llm = SimpleNamespace(generate = lambda *a, **k: [SimpleNamespace(
        prompt_token_ids = prompt, outputs = [SimpleNamespace(token_ids = completion, routed_experts = full[2:4])],
    )])
    rr.wrap_vllm_generate(llm)
    llm.generate(["x"], sampling_params = [params] if per_prompt else params)
    rows = rr._lookup(tuple(prompt + completion), exact = True)
    assert bool((rows[:2] == -1).all())
    assert torch.equal(rows[2:4].long(), torch.as_tensor(full[2:4]).long())
    assert bool((rows[4] == -1).all())


def test_duplicate_completions_replay_only_their_common_routes(replay):
    tokens = [1, 2, 3]
    a = _routes(3, 2)
    b = a.copy()
    b[1, 0] = (a[1, 0] + 1) % 8  # token 1, layer 0: another expert set in the second copy

    def req(routed):
        return SimpleNamespace(prompt_token_ids = tokens[:1], outputs = [SimpleNamespace(token_ids = tokens[1:], routed_experts = routed)])

    for split_calls in (False, True):
        rr.reset_routing_replay()
        if split_calls:
            rr.record_routed_experts([req(a)])
            rr.record_routed_experts([req(b)])
        else:
            rr.record_routed_experts([req(a), req(b)])
        rows = rr._lookup(tuple(tokens), exact = True)
        assert bool((rows[1, 0] == -1).all()), split_calls
        keep = torch.ones(3, LAYERS, dtype = torch.bool)
        keep[1, 0] = False
        assert torch.equal(rows[keep].long(), torch.as_tensor(a)[keep].long())
        for _ in range(3):
            rr._lookup(tuple(tokens), exact = True)
        assert rr.routing_replay_stats()["conflicting_routes"] == 1, split_calls


def test_invalid_capture_next_to_a_valid_one_is_not_a_conflict(replay):
    tokens = [1, 2, 3]
    a = _routes(3, 3)
    b = a.copy()
    b[1, 0] = -1  # vLLM gave no routing there (rows with repeated ids are marked -1 too)

    def req(routed):
        return SimpleNamespace(prompt_token_ids = tokens[:1], outputs = [SimpleNamespace(token_ids = tokens[1:], routed_experts = routed)])

    rr.record_routed_experts([req(a), req(b)])
    assert rr.routing_replay_stats().get("conflicting_routes", 0) == 0


def test_return_args_path_does_not_ask_vllm_serve_for_routed_experts():
    # `vllm serve` drops False values (no --no-async-scheduling), so replay args stay out.
    import unsloth_zoo.vllm_utils as vu
    path = pathlib.Path(vu.__file__)
    tree = ast.parse(path.read_text(encoding = "utf-8-sig"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    guard = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and "routing_replay_engine_args" in ast.unparse(n)
        and not isinstance(n.test, ast.Constant)
    )
    assert "not return_args" in ast.unparse(guard.test)


transformers = pytest.importorskip("transformers")


def _moe(seed = 0):
    try:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter  # noqa: F401
    except Exception:
        pytest.skip("needs transformers >= 5 router modules")
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size = 64, hidden_size = 32, intermediate_size = 64, moe_intermediate_size = 16,
        num_hidden_layers = LAYERS, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 8,
        num_experts = 8, num_experts_per_tok = TOP_K, norm_topk_prob = True, pad_token_id = PAD,
    )
    model = Qwen3MoeForCausalLM(config).float()
    for layer in model.model.layers:
        layer.mlp.gate.weight.data.normal_(0, 0.5)
    return model


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_recompute_of_identical_inputs_with_different_replays_keeps_gradients(replay, use_reentrant, compiled):
    # Two rollouts with identical tokens and routes: only the layer input tensor tells recomputes apart.
    ids = [5, 6, 7, 30, 31, 32]
    natural = None
    with torch.no_grad():
        m = _moe()
        captured = {}
        for i, layer in enumerate(m.model.layers):
            layer.mlp.gate.register_forward_hook(lambda mod, a, out, i = i: captured.__setitem__(i, out[2].clone()))
        m(input_ids = torch.tensor([ids]))
        natural = torch.stack([captured[i] for i in range(LAYERS)], dim = 1).numpy()
    shifted = [((natural + s) % 8).astype(np.int16) for s in (2, 5)]

    def run(checkpointing):
        m = _moe()
        m.train()
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant})
        m.model.embed_tokens.weight.requires_grad_(True)
        if compiled:
            torch._dynamo.reset()
            for layer in m.model.layers:
                layer.compile(backend = "eager")
        rr.reset_routing_replay()
        assert rr.install_routing_replay(m, None, SimpleNamespace(pad_token_id = PAD)) == LAYERS
        losses = []
        for routed in shifted:
            rr.record_routed_experts([SimpleNamespace(
                prompt_token_ids = ids[:3], outputs = [SimpleNamespace(token_ids = ids[3:], routed_experts = routed[:-1])],
            )])
            losses.append(m(input_ids = torch.tensor([ids])).logits.float().square().mean())
        sum(losses).backward()
        return torch.stack([l.detach() for l in losses]), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    losses_plain, grads_plain = run(False)
    losses_gc, grads_gc = run(True)
    assert not torch.allclose(losses_plain[0], losses_plain[1])
    torch.testing.assert_close(losses_gc, losses_plain, rtol = 1e-5, atol = 1e-6)
    assert grads_gc.keys() == grads_plain.keys()
    for name in grads_plain:
        torch.testing.assert_close(grads_gc[name], grads_plain[name], rtol = 1e-4, atol = 1e-6, msg = name)


def test_replayssm_yields_to_routing_replay_on_vllm_031():
    # Routed experts capture needs MRV2; the skip must come before use_replayssm is set.
    path = pathlib.Path(rr.__file__).with_name("vllm_utils.py")
    src = path.read_text(encoding = "utf-8-sig")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    body = ast.get_source_segment(src, fn)
    guard = body.find('engine_args.get("enable_return_routed_experts")')
    assert guard != -1 and "_vllm_at_least(0, 31)" in body[guard : guard + 600]
    assert guard < body.find('engine_args["use_replayssm"] = True')


class _NoStream:
    def wait_stream(self, other):
        pass

    def wait_event(self, event):
        pass

    def record_event(self, event = None):
        # A released CPU slot keeps the event as its fence; nothing here ever waits on it.
        return object() if event is None else event


def test_the_cpu_stream_stand_in_covers_every_stream_call():
    """The offload path calls these on whatever stream it holds; a method the stand-in lacks stops
    the recompute test below with an AttributeError instead of checking the recompute."""
    import unsloth_zoo.gradient_checkpointing as gc

    tree = ast.parse(pathlib.Path(gc.__file__).read_text(encoding = "utf-8"))
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
            if "stream" in name.lower():
                called.add(node.func.attr)
    assert called, "found no stream calls in gradient_checkpointing.py; the scan is stale"
    missing = sorted(m for m in called if not callable(getattr(_NoStream, m, None)))
    assert not missing, f"_NoStream needs {missing}: gradient_checkpointing.py calls them on a stream"


@pytest.fixture
def offloaded_gc(monkeypatch):
    """Unsloth's offloaded checkpoint (UnslothCheckpointFunction) with its offload branch on, CPU
    tensors standing in for the device: the first activation is copied to a host buffer in forward
    and rebuilt in the one reused staging buffer for every recompute, as on a GPU."""
    import contextlib

    from unsloth_zoo import gradient_checkpointing as gc
    staging = torch.empty(0, dtype = torch.float32)
    for name, value in (
        ("FIRST_PASS", True), ("LAST_GC_INDEX", 0), ("CURRENT_GC_INDEX", 0),
        ("MINIMUM_SIZE", 0),  # every activation is above the offload threshold
        ("CPU_INDEX", 0), ("CPU_BUFFERS", [torch.empty(1)]), ("BACKWARD_PASS", False),
        ("USE_DOUBLE_BUFFER", False), ("GPU_BUFFERS", {None: staging}), ("GPU_BUFFERS_B", None),
        ("EXTRA_STREAMS", {None: _NoStream()}), ("NEXT_BUFFER_SLOT", {None: 0}),
        ("FP32_OFFLOAD_EXACT", True), ("GC_PREFETCH", False), ("OFFLOAD_SLOT_META", []),
        ("USE_UNSLOTH_GC", False), ("PINNED_MEMORY_AVAILABLE", False),
    ):
        monkeypatch.setattr(gc, name, value, raising = False)
    monkeypatch.setattr(gc, "_compute_stream", lambda device_index: _NoStream())
    monkeypatch.setattr(gc, "torch_gpu_stream", lambda stream: contextlib.nullcontext(), raising = False)
    monkeypatch.setattr(gc, "_ensure_device_buffers", lambda device_index: None)
    return gc, staging


@pytest.mark.parametrize("prior", ["plain", "offloaded"])
def test_recompute_identity_survives_offloaded_checkpointing(replay, offloaded_gc, prior):
    # Same scenario on use_gradient_checkpointing = "unsloth": hidden states live in the staging buffer.
    gc, staging = offloaded_gc
    ids = [5, 6, 7, 30, 31, 32]
    with torch.no_grad():
        m = _moe()
        captured = {}
        for i, layer in enumerate(m.model.layers):
            layer.mlp.gate.register_forward_hook(lambda mod, a, out, i = i: captured.__setitem__(i, out[2].clone()))
        m(input_ids = torch.tensor([ids]))
        natural = torch.stack([captured[i] for i in range(LAYERS)], dim = 1).numpy()
    shifted = [((natural + s) % 8).astype(np.int16) for s in (2, 5)]

    def run(checkpointing):
        m = _moe()
        m.train()
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
            for layer in m.model.layers:
                layer._gradient_checkpointing_func = gc.unsloth_offloaded_gradient_checkpoint
        m.model.embed_tokens.weight.requires_grad_(True)
        rr.reset_routing_replay()
        assert rr.install_routing_replay(m, None, SimpleNamespace(pad_token_id = PAD)) == LAYERS
        recompute_inputs = []
        m.model.layers[0].register_forward_pre_hook(
            lambda mod, args: recompute_inputs.append(args[0].data_ptr()) if not rr._STATE.in_forward else None
        )
        losses = []
        for routed in shifted:
            rr.record_routed_experts([SimpleNamespace(
                prompt_token_ids = ids[:3], outputs = [SimpleNamespace(token_ids = ids[3:], routed_experts = routed[:-1])],
            )])
            losses.append(m(input_ids = torch.tensor([ids])).logits.float().square().mean())
        if checkpointing:
            # After the first step the last checkpoint stays on device (LAST_GC_INDEX).
            assert gc.CPU_INDEX >= 2 * LAYERS - 1
        sum(losses).backward()
        if checkpointing:
            assert len(recompute_inputs) == 2 and set(recompute_inputs) == {staging.data_ptr()}
        return torch.stack([l.detach() for l in losses]), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

    if prior == "offloaded":
        run(True)  # a finished step first: buffers are reused, the step after must still match
    losses_plain, grads_plain = run(False)
    losses_gc, grads_gc = run(True)
    torch.testing.assert_close(losses_gc, losses_plain, rtol = 1e-5, atol = 1e-6)
    assert grads_gc.keys() == grads_plain.keys()
    for name in grads_plain:
        torch.testing.assert_close(grads_gc[name], grads_plain[name], rtol = 1e-4, atol = 1e-6, msg = name)


@pytest.fixture
def runner_env(monkeypatch):
    import importlib.metadata

    import unsloth_zoo.vllm_utils as vu
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising = False)
    monkeypatch.setattr(vu, "_UNSLOTH_SET_V1_RUNNER", False)
    versions = {"vllm": "0.32.0"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: versions[name])
    # Hermetic: an installed vLLM's own constants and env cache would override the mocked version.
    # A test needing vllm.config.vllm plants its own.
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", None)
    monkeypatch.setattr(vu, "_vllm_disable_envs_cache", lambda: None)
    return vu, versions


@pytest.fixture
def installed_vllm_029(monkeypatch):
    # vLLM 0.29 installed: vllm.config.vllm exists but has no ROCM_DEFAULT_MRV1_ARCHITECTURES.
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", SimpleNamespace())


def test_mocked_versions_ignore_the_installed_vllm(installed_vllm_029, runner_env, monkeypatch):
    vu, versions = runner_env
    versions["vllm"] = "0.31.0"
    monkeypatch.setattr(vu, "is_hip", lambda: True)
    args = _replay_args()
    with pytest.warns(UserWarning, match = "on ROCm vLLM runs"):
        assert vu._routing_replay_model_runner(args, _rocm_v1_config()) == "replay_off"


def _replay_args():
    return {"enable_return_routed_experts": True, "moe_backend": "triton", "model": "m"}


def test_v1_runner_left_by_an_earlier_replayssm_load_is_taken_back(runner_env, monkeypatch):
    # Earlier ReplaySSM load left VLLM_USE_V2_MODEL_RUNNER=0; vLLM 0.31 refuses capture on V1.
    vu, _ = runner_env
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(vu, "_UNSLOTH_SET_V1_RUNNER", True)
    args = _replay_args()
    assert vu._routing_replay_model_runner(args) == "restored_v2"
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ and not vu._UNSLOTH_SET_V1_RUNNER
    assert args == _replay_args()


@pytest.mark.parametrize("value", ["0", " 0 "])
def test_user_chosen_v1_runner_turns_routing_replay_off_with_a_warning(runner_env, monkeypatch, value):
    vu, _ = runner_env
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", value)
    args = dict(_replay_args(), async_scheduling = False)
    with pytest.warns(UserWarning, match = "Model Runner V1"):
        assert vu._routing_replay_model_runner(args) == "replay_off"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == value  # the user's choice stays
    assert "enable_return_routed_experts" not in args and "async_scheduling" not in args


@pytest.mark.parametrize("version, env, owned", [
    ("0.30.0", "0", True),   # before 0.31 V1 captures routed experts: nothing to do
    ("0.32.0", None, False),
    ("0.32.0", "1", False),
])
def test_runner_is_left_alone_when_there_is_no_conflict(runner_env, monkeypatch, version, env, owned):
    vu, versions = runner_env
    versions["vllm"] = version
    if env is not None:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", env)
    monkeypatch.setattr(vu, "_UNSLOTH_SET_V1_RUNNER", owned)
    args = _replay_args()
    assert vu._routing_replay_model_runner(args) is None
    assert args == _replay_args() and os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == env
    assert vu._routing_replay_model_runner({"model": "m"}) is None


def test_load_vllm_checks_the_runner_even_when_replayssm_is_off():
    # Check sits before the ReplaySSM block, at its own level.
    import unsloth_zoo.vllm_utils as vu
    path = pathlib.Path(vu.__file__)
    tree = ast.parse(path.read_text(encoding = "utf-8-sig"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    for node in ast.walk(fn):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        ifs = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "replayssm_mode != 'off'"]
        calls = [
            i for i, s in enumerate(body)
            if isinstance(s, (ast.Expr, ast.Assign)) and isinstance(s.value, ast.Call)
            and getattr(s.value.func, "id", None) == "_routing_replay_model_runner"
        ]
        if ifs and calls:
            assert calls[0] < ifs[0]
            break
    else:
        raise AssertionError("_routing_replay_model_runner is not called beside the ReplaySSM block")
    src = ast.get_source_segment(path.read_text(encoding = "utf-8-sig"), fn)
    assert 'os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"\n                    replayssm_set_runner = True\n                    _UNSLOTH_SET_V1_RUNNER = True' in src


def _runner_and_replayssm_block(vu):
    # Compile the load_vllm statements on their own: load_vllm itself needs a GPU.
    path = pathlib.Path(vu.__file__)
    src = path.read_text(encoding = "utf-8-sig")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "load_vllm")
    for node in ast.walk(fn):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        ifs = [i for i, s in enumerate(body) if isinstance(s, ast.If) and ast.unparse(s.test) == "replayssm_mode != 'off'"]
        if not ifs:
            continue
        calls = [
            i for i, s in enumerate(body)
            if isinstance(s, (ast.Expr, ast.Assign)) and isinstance(s.value, ast.Call)
            and getattr(s.value.func, "id", None) == "_routing_replay_model_runner"
        ]
        start = next(i for i, s in enumerate(body) if ast.unparse(s) == "replayssm_set_runner = False")
        span = body[min([start, *calls]) : max([ifs[0], *calls]) + 1]
        module = ast.Module(body = [ast.Global(names = ["_UNSLOTH_SET_V1_RUNNER"]), *span], type_ignores = [])
        fn_src = "def _block():\n" + "\n".join("    " + line for line in ast.unparse(module).splitlines())
        fn_src += "\n    return locals()"
        return compile(fn_src, str(path), "exec")
    raise AssertionError("ReplaySSM block not found in load_vllm")


def _run_block(vu, monkeypatch, engine_args, **names):
    ns = dict(vars(vu))
    ns.update(
        config = SimpleNamespace(model_type = "nemotron_h"),
        good_keys = {"model", "use_replayssm", "enable_return_routed_experts"},
        DEVICE_TYPE = "cuda",
        engine_args = engine_args,
        replay_moe_pin = False,
        pre_speculation_sizes = None,
    )
    ns.update(names)
    exec(_runner_and_replayssm_block(vu), ns)
    ns["block_locals"] = ns["_block"]()
    return ns


@pytest.mark.parametrize("replay_pinned", [True, False])
def test_replay_off_releases_the_moe_backend_only_routing_replay_pinned(runner_env, monkeypatch, replay_pinned):
    vu, _ = runner_env
    monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    args = _replay_args()
    with pytest.warns(UserWarning, match = "Model Runner V1"):
        _run_block(vu, monkeypatch, args, replay_moe_pin = replay_pinned)
    assert "enable_return_routed_experts" not in args
    assert ("moe_backend" in args) is (not replay_pinned)


def test_moe_backend_pin_stays_while_routing_replay_is_on(runner_env, monkeypatch):
    vu, _ = runner_env
    monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", "0")
    args = _replay_args()
    _run_block(vu, monkeypatch, args, replay_moe_pin = True)
    assert args["moe_backend"] == "triton" and args["enable_return_routed_experts"] is True


@pytest.mark.parametrize("owned_v1", [False, True])
def test_dropped_suffix_decoding_gives_back_its_batch_limits(runner_env, monkeypatch, owned_v1):
    # Once routing replay drops suffix decoding, the pre-speculation limits come back.
    vu, _ = runner_env
    monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", "0")
    if owned_v1:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        monkeypatch.setattr(vu, "_UNSLOTH_SET_V1_RUNNER", True)
    args = dict(_replay_args(), speculative_config = {"method": "suffix"}, max_num_seqs = 12, max_num_batched_tokens = 512)
    ns = _run_block(
        vu, monkeypatch, args, approx_max_num_seqs = 12, max_num_batched_tokens = 512,
        pre_speculation_sizes = (64, 256, 64, 256),
    )
    assert "speculative_config" not in args
    assert args["max_num_seqs"] == 64 and args["max_num_batched_tokens"] == 256
    assert ns["block_locals"]["approx_max_num_seqs"] == 64
    assert ns["block_locals"]["max_num_batched_tokens"] == 256


def test_kept_speculation_keeps_its_batch_limits(runner_env, monkeypatch):
    vu, _ = runner_env
    monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", "0")
    args = dict(_replay_args(), speculative_config = {"method": "mtp"}, max_num_seqs = 12, max_num_batched_tokens = 512)
    _run_block(vu, monkeypatch, args, approx_max_num_seqs = 12, max_num_batched_tokens = 512,
               pre_speculation_sizes = (64, 256, 64, 256))
    assert args["max_num_seqs"] == 12 and args["max_num_batched_tokens"] == 512


def test_user_v1_runner_keeps_replayssm_when_it_turns_routing_replay_off(runner_env, monkeypatch, capsys):
    # User V1 runner + ReplaySSM: routing replay drops, ReplaySSM stays on.
    vu, _ = runner_env
    monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    args = _replay_args()
    with pytest.warns(UserWarning, match = "Model Runner V1"):
        _run_block(vu, monkeypatch, args)
    assert "enable_return_routed_experts" not in args
    assert args.get("use_replayssm") is True
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert "ignored" not in capsys.readouterr().out


def test_inherited_v1_runner_keeps_routing_replay_and_says_replayssm_was_dropped(runner_env, monkeypatch, capsys):
    vu, _ = runner_env
    monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(vu, "_UNSLOTH_SET_V1_RUNNER", True)
    args = _replay_args()
    _run_block(vu, monkeypatch, args)
    assert args.get("enable_return_routed_experts") is True and "use_replayssm" not in args
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ
    assert "UNSLOTH_VLLM_REPLAYSSM=1 ignored: routing replay" in capsys.readouterr().out


def test_unset_runner_keeps_routing_replay_on_v2(runner_env, monkeypatch, capsys):
    vu, _ = runner_env
    monkeypatch.setenv("UNSLOTH_VLLM_REPLAYSSM", "1")
    args = _replay_args()
    _run_block(vu, monkeypatch, args)
    assert args.get("enable_return_routed_experts") is True and "use_replayssm" not in args
    assert "VLLM_USE_V2_MODEL_RUNNER" not in os.environ
    assert "UNSLOTH_VLLM_REPLAYSSM=1 ignored: routing replay" in capsys.readouterr().out


def test_routing_replay_is_off_without_prefix_caching(runner_env, monkeypatch):
    # load_vllm turns prefix caching off on compute capability < 7.5 and on request.
    vu, _ = runner_env
    args = dict(_replay_args(), enable_prefix_caching = False, async_scheduling = False)
    with pytest.warns(UserWarning, match = "prefix caching"):
        assert vu._routing_replay_model_runner(args) == "replay_off"
    assert "enable_return_routed_experts" not in args and "async_scheduling" not in args
    assert args["enable_prefix_caching"] is False


@pytest.mark.parametrize("version", ["0.30.0", "0.32.0"])
def test_routing_replay_stays_on_with_prefix_caching(runner_env, version):
    vu, versions = runner_env
    versions["vllm"] = version
    args = dict(_replay_args(), enable_prefix_caching = True)
    assert vu._routing_replay_model_runner(args) is None
    assert args["enable_return_routed_experts"] is True
    if version == "0.30.0":
        args = dict(_replay_args(), enable_prefix_caching = False)
        assert vu._routing_replay_model_runner(args) is None
        assert args["enable_return_routed_experts"] is True


@pytest.mark.parametrize("owned_v1", [False, True])
def test_suffix_decoding_yields_to_routing_replay_on_model_runner_v2(runner_env, monkeypatch, capsys, owned_v1):
    vu, _ = runner_env
    if owned_v1:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        monkeypatch.setattr(vu, "_UNSLOTH_SET_V1_RUNNER", True)
    args = dict(_replay_args(), enable_prefix_caching = True, speculative_config = {"method": "suffix"})
    action = vu._routing_replay_model_runner(args)
    assert action == ("restored_v2" if owned_v1 else "speculation_off")
    assert "speculative_config" not in args and args["enable_return_routed_experts"] is True
    assert "suffix decoding is off" in capsys.readouterr().out


def test_v2_supported_speculation_is_kept(runner_env):
    vu, versions = runner_env
    args = dict(_replay_args(), speculative_config = {"method": "mtp"})
    assert vu._routing_replay_model_runner(args) is None
    assert args["speculative_config"] == {"method": "mtp"}
    versions["vllm"] = "0.30.0"
    args = dict(_replay_args(), speculative_config = {"method": "suffix"})
    assert vu._routing_replay_model_runner(args) is None and "speculative_config" in args


def _rocm_v1_config(arch = "DeepseekV32ForCausalLM"):
    return SimpleNamespace(architectures = [arch], model_type = "deepseek_v32")


@pytest.mark.parametrize("arch", ["DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM", "GlmMoeDsaForCausalLM"])
@pytest.mark.parametrize("owned_v1", [False, True])
def test_rocm_v1_default_architectures_turn_routing_replay_off(runner_env, monkeypatch, arch, owned_v1):
    # On ROCm an unset runner is V1 for ROCM_DEFAULT_MRV1_ARCHITECTURES on vLLM 0.31.
    vu, versions = runner_env
    versions["vllm"] = "0.31.0"
    monkeypatch.setattr(vu, "is_hip", lambda: True)
    if owned_v1:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        monkeypatch.setattr(vu, "_UNSLOTH_SET_V1_RUNNER", True)
    args = dict(_replay_args(), async_scheduling = False)
    with pytest.warns(UserWarning, match = "on ROCm vLLM runs"):
        assert vu._routing_replay_model_runner(args, _rocm_v1_config(arch)) == "replay_off"
    assert "enable_return_routed_experts" not in args and "async_scheduling" not in args


@pytest.mark.parametrize("hip, env, arch", [
    (True, "1", "DeepseekV32ForCausalLM"),   # the user forces V2: vLLM honours it
    (False, None, "DeepseekV32ForCausalLM"),  # CUDA: V2 by default
    (True, None, "Qwen3MoeForCausalLM"),     # not a ROCm V1 default architecture
])
def test_rocm_v1_default_leaves_routing_replay_on_otherwise(runner_env, monkeypatch, hip, env, arch):
    vu, _ = runner_env
    monkeypatch.setattr(vu, "is_hip", lambda: hip)
    if env is not None:
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", env)
    args = _replay_args()
    assert vu._routing_replay_model_runner(args, _rocm_v1_config(arch)) is None
    assert args == _replay_args()


@pytest.mark.parametrize("version", ["0.32.0", "0.33.1"])
def test_rocm_v1_default_is_gone_from_vllm_032(runner_env, monkeypatch, version):
    vu, versions = runner_env
    versions["vllm"] = version
    monkeypatch.setattr(vu, "is_hip", lambda: True)
    args = _replay_args()
    assert vu._routing_replay_model_runner(args, _rocm_v1_config()) is None
    assert args == _replay_args()


@pytest.mark.parametrize("defined, expected", [
    (frozenset({"Qwen3MoeForCausalLM"}), "replay_off"),
    (None, None),
])
def test_rocm_v1_default_follows_the_installed_vllm(runner_env, monkeypatch, defined, expected):
    vu, versions = runner_env
    versions["vllm"] = "0.31.0"
    monkeypatch.setattr(vu, "is_hip", lambda: True)
    module = SimpleNamespace() if defined is None else SimpleNamespace(ROCM_DEFAULT_MRV1_ARCHITECTURES = defined)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", module)
    args = _replay_args()
    if expected is None:
        assert vu._routing_replay_model_runner(args, _rocm_v1_config("DeepseekV32ForCausalLM")) is None
    else:
        with pytest.warns(UserWarning, match = "on ROCm vLLM runs"):
            assert vu._routing_replay_model_runner(args, _rocm_v1_config("Qwen3MoeForCausalLM")) == expected


def test_load_vllm_passes_the_config_to_the_runner_check():
    import unsloth_zoo.vllm_utils as vu
    src = pathlib.Path(vu.__file__).read_text(encoding = "utf-8-sig")
    assert "_routing_replay_model_runner(engine_args, config)" in src
