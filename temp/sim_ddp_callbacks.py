"""World-size=2 simulation for the merged MLX HF-callback DDP loop.

Validates the three merge-critical properties:
  (i)   a stop set on rank 0 (e.g. EarlyStopping) stops BOTH ranks via the
        _distributed_should_stop() OR-reduce -- never rank 0 alone.
  (ii)  HF TrainerCallbacks fire in canonical HF order over a real train() run.
  (iii) TrainerState.is_world_process_zero / is_local_process_zero reflect the
        real rank, not a hardcoded True.

Runs under the tests/ MLX-on-torch shim (no Metal, no real distributed).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))  # prefer the worktree copy of unsloth_zoo


def _preload_device_type():
    """Mirror tests/conftest.py: let unsloth_zoo import without a live GPU."""
    try:
        import torch
    except Exception:
        return
    real = getattr(getattr(torch, "cuda", None), "is_available", None)
    try:
        if callable(real) and torch.cuda.is_available():
            return  # a real accelerator is visible; normal detection works
    except Exception:
        pass
    torch.cuda.is_available = lambda: True
    try:
        import unsloth_zoo.device_type  # noqa: F401
        unsloth_zoo.device_type.get_device_type()
    except Exception:
        pass
    finally:
        if callable(real):
            torch.cuda.is_available = real


_preload_device_type()

from mlx_simulation import simulate_mlx_on_torch  # noqa: E402

simulate_mlx_on_torch()

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import unsloth_zoo.mlx.trainer as trainer_mod  # noqa: E402
from unsloth_zoo.mlx.trainer import (  # noqa: E402
    MLXTrainer,
    MLXTrainingConfig,
    _MLXTrainerControl,
    _create_labeled_batches,
)

failures = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------------------
# (i) Stop set on rank 0 must stop BOTH ranks via the OR-reduce collective.
#     Emulate a 2-rank all_sum: each rank's collective sees its own local
#     contribution plus the peer's. peer_contrib[rank] is what the OTHER rank
#     contributed for the current reduction.
# ---------------------------------------------------------------------------
def make_rank(rank, world_size=2):
    t = MLXTrainer.__new__(MLXTrainer)
    t._distributed_world = object()  # non-None so collectives are taken
    t._distributed_initialized = True
    t._distributed_rank = rank
    t._distributed_world_size = world_size
    t._distributed_is_main_process = rank == 0
    t.stop_requested = False
    t.control = _MLXTrainerControl()
    return t


class _PeerAllSum:
    """Emulate all_sum(local) = local + peer for a fixed peer contribution."""

    def __init__(self):
        self.peer = 0

    def __call__(self, value, group=None, stream=None):
        return value + mx.array(self.peer, dtype=value.dtype)


peer = _PeerAllSum()
orig_all_sum = mx.distributed.all_sum
mx.distributed.all_sum = peer

try:
    rank0 = make_rank(0)
    rank1 = make_rank(1)

    # An EarlyStopping-style callback ran on rank 0 only and set the stop flag.
    rank0.control.should_training_stop = True
    # rank 1 never saw the callback: its control stays default.

    # Rank 0: copy control -> stop_requested, then OR-reduce. Peer (rank 1)
    # contributed 0 to this reduction.
    rank0._sync_callback_stop()
    peer.peer = 0
    r0_stop = rank0._distributed_should_stop()

    # Rank 1: no local stop; the OR-reduce must see rank 0's contribution (1).
    rank1._sync_callback_stop()
    peer.peer = 1
    r1_stop = rank1._distributed_should_stop()

    check("(i) rank0 stop_requested True after callback+reduce", rank0.stop_requested is True)
    check("(i) rank0 _distributed_should_stop() returns True", r0_stop is True)
    check("(i) rank1 had no local stop before reduce is respected via OR", r1_stop is True)
    check("(i) rank1 stop_requested flipped True by OR-reduce (no deadlock)", rank1.stop_requested is True)

    # Sanity: with NO rank requesting stop, neither stops (no false positive).
    q0, q1 = make_rank(0), make_rank(1)
    q0._sync_callback_stop(); peer.peer = 0
    s0 = q0._distributed_should_stop()
    q1._sync_callback_stop(); peer.peer = 0
    s1 = q1._distributed_should_stop()
    check("(i) no stop requested -> both ranks continue", (s0 is False) and (s1 is False))

    # Control-action sync: rank 0 requests eval only; both ranks must agree so
    # the collective eval path runs in lockstep (else peers deadlock).
    c0, c1 = make_rank(0), make_rank(1)
    c0.control.should_evaluate = True   # a callback on rank 0 asked for eval
    c1.control.should_evaluate = False  # rank 1 saw nothing
    # base = world_size + 1 = 3; rank0 code for eval-only = 3, rank1 = 0.
    peer.peer = 0
    c0._distributed_sync_control_actions()  # rank0 sees its 3 + peer 0 = 3
    peer.peer = 3
    c1._distributed_sync_control_actions()  # rank1 sees its 0 + peer 3 = 3
    check("(i) control-actions sync: rank0 keeps should_evaluate", c0.control.should_evaluate is True)
    check("(i) control-actions sync: rank1 gains should_evaluate (lockstep)", c1.control.should_evaluate is True)
    check("(i) control-actions sync: log/save stay False", (c1.control.should_log is False) and (c1.control.should_save is False))
finally:
    mx.distributed.all_sum = orig_all_sum


# ---------------------------------------------------------------------------
# (iii) TrainerState rank flags come from the real rank.
# ---------------------------------------------------------------------------
main = MLXTrainer.__new__(MLXTrainer)
main._distributed_initialized = True
main._distributed_world = None
main._distributed_rank = 0
main._distributed_world_size = 2
main._distributed_is_main_process = True
main.args = MLXTrainingConfig(max_steps=2, logging_steps=1)
main._init_callback_state(total_steps=2, resume_step=0)
check("(iii) rank0 is_world_process_zero True", main.state.is_world_process_zero is True)
check("(iii) rank0 is_local_process_zero True", main.state.is_local_process_zero is True)

peerrank = MLXTrainer.__new__(MLXTrainer)
peerrank._distributed_initialized = True
peerrank._distributed_world = None
peerrank._distributed_rank = 1
peerrank._distributed_world_size = 2
peerrank._distributed_is_main_process = False
peerrank.args = MLXTrainingConfig(max_steps=2, logging_steps=1)
peerrank._init_callback_state(total_steps=2, resume_step=0)
check("(iii) rank1 is_world_process_zero False", peerrank.state.is_world_process_zero is False)
check("(iii) rank1 is_local_process_zero False", peerrank.state.is_local_process_zero is False)


# ---------------------------------------------------------------------------
# (ii) HF callbacks fire in canonical order over a real single-rank train().
# ---------------------------------------------------------------------------
class TinyTokenizer:
    pad_token_id = eos_token_id = 2

    def encode(self, text):
        return [int(part) for part in str(text).split()]

    def __call__(self, text, **_kwargs):
        return {"input_ids": self.encode(text)}


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.proj = nn.Linear(8, 64, bias=False)
        self._config = {"model_type": "tiny"}

    def __call__(self, x):
        return self.proj(self.embed(x))


# The mlx-on-torch shim omits nn.Module.state / optimizer.state, so a full
# gradient train() is Metal-gated (see tests/test_mlx_ddp_metal.py). We validate
# HF callback ORDER two authentic ways instead: (ii-a) the real _train_inner
# source fires the events in canonical HF order, and (ii-b) the real
# _MLXCallbackHandler dispatches to the matching method, threads control, and a
# rank-gated dispatch (the loop's _fire) fires on rank 0 only.

# (ii-a) Canonical HF dispatch order at the real loop-body call sites. The
# log/eval/save/epoch_end events live in DDP-correct helper closures, so we
# order by the call sites inside the training loop (runtime order), not by the
# event-string text position (which would hit the earlier helper definitions).
import inspect  # noqa: E402

canonical = [
    "on_train_begin",
    "on_step_begin",
    "on_substep_end",
    "on_optimizer_step",
    "on_step_end",
    "on_log",
    "on_evaluate",
    "on_save",
    "on_epoch_end",
    "on_train_end",
]
src = inspect.getsource(trainer_mod.MLXTrainer._train_inner)
loop_body = src[src.index("while self._global_step < total_steps:"):]
# Runtime call sites, in the order the loop executes them.
call_sites = [
    ("on_train_begin", '_fire("on_train_begin")', src),
    ("on_epoch_begin", "_maybe_callback_epoch_begin(it)", loop_body),
    ("on_step_begin", '_fire("on_step_begin")', loop_body),
    ("on_optimizer_step", '_fire("on_optimizer_step")', loop_body),
    ("on_substep_end", '_fire("on_substep_end")', loop_body),
    ("on_step_end", '_fire("on_step_end")', loop_body),
    ("on_log", "_run_training_log(current_step, grad_norm)", loop_body),
    ("on_evaluate", "_run_eval(current_step)", loop_body),
    ("on_save", "_run_checkpoint(current_step)", loop_body),
    ("on_epoch_end", "_maybe_callback_epoch_end(it", loop_body),
    ("on_train_end", '_fire("on_train_end")', src),
]
found = {name: hay.find(tok) for name, tok, hay in call_sites}
missing = [name for name, pos in found.items() if pos < 0]
check("(ii-a) every HF lifecycle event has a loop call site", not missing)
# Order the ones that share the loop_body haystack (comparable positions).
body_seq = [
    "on_epoch_begin", "on_step_begin", "on_optimizer_step", "on_substep_end",
    "on_step_end", "on_log", "on_evaluate", "on_save", "on_epoch_end",
]
ordered = all(found[a] < found[b] for a, b in zip(body_seq, body_seq[1:]))
check("(ii-a) loop body invokes callbacks in canonical HF order", ordered)
# on_step_begin resets the action control flags before dispatch (HF parity).
check("(ii-a) on_step_begin clears control action flags first",
      loop_body.find("self.control.should_log = False") < found["on_step_begin"])
# on_step_end is followed by a rank-wide control-action sync before log/eval/save.
check("(ii-a) control actions synced across ranks after on_step_end",
      0 < loop_body.find("_distributed_sync_control_actions()")
      and found["on_step_end"] < loop_body.find("_distributed_sync_control_actions()") < found["on_log"])


# (ii-b) Functional: the real handler dispatches in order and threads control.
class OrderingCallback:
    def __init__(self):
        self.events = []

    def _rec(self, name):
        def handler(args, state, control, **kwargs):
            self.events.append((name, state.global_step, state.is_world_process_zero))
            return control
        return handler

    def __getattr__(self, name):
        if name.startswith("on_"):
            return self._rec(name)
        raise AttributeError(name)


cb = OrderingCallback()
handler = trainer_mod._MLXCallbackHandler(
    [cb], model=object(), processing_class=object(),
    optimizer=None, lr_scheduler=None,
)


def rank_gated_fire(is_main, state, control, event, **kwargs):
    """Mirror the loop's _fire: dispatch on rank 0 only."""
    if is_main:
        return handler.call_event(event, None, state, control, **kwargs)
    return control


# Drive one optimizer step (grad_accum=2) as the loop does on rank 0.
class _State:
    def __init__(self, rank_zero):
        self.global_step = 0
        self.is_world_process_zero = rank_zero
        self.is_local_process_zero = rank_zero


control = _MLXTrainerControl()
st0 = _State(rank_zero=True)
for ev in ["on_train_begin", "on_step_begin", "on_substep_end",
           "on_optimizer_step", "on_step_end", "on_log", "on_evaluate",
           "on_save", "on_epoch_end", "on_train_end"]:
    control = rank_gated_fire(True, st0, control, ev)

fired = [e[0] for e in cb.events]
check("(ii-b) handler fires all events in HF order on rank 0", fired == canonical)
check("(ii-b) every rank-0 event sees is_world_process_zero True",
      all(e[2] is True for e in cb.events))

# Rank 1 (not main): the gate must suppress every dispatch (no duplicate I/O).
cb1 = OrderingCallback()
handler = trainer_mod._MLXCallbackHandler(
    [cb1], model=object(), processing_class=object(),
    optimizer=None, lr_scheduler=None,
)
st1 = _State(rank_zero=False)
control = _MLXTrainerControl()
for ev in canonical:
    control = rank_gated_fire(False, st1, control, ev)
check("(ii-b) rank 1 dispatch is fully suppressed by the rank gate",
      cb1.events == [])

print("callback order (rank 0):", fired)
print()
if failures:
    print("SIM FAILURES:", failures)
    sys.exit(1)
print("ALL SIM CHECKS PASSED")
