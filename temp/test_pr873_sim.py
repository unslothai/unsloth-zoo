"""Simulation of PR #873 HF-style TrainerCallback support for MLXTrainer.

Runs the PR's trainer.py (HEAD) under the torch-backed mlx_simulation shim on
Linux (no Metal). Registers real transformers.TrainerCallback spies and asserts
lifecycle ordering, TrainerControl.should_training_stop wiring, and that the
pre-existing function callbacks (add_step_callback/add_eval_callback) and
report_to path are preserved.
"""

from __future__ import annotations
import os, sys
import pytest

# install shim before importing trainer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from mlx_simulation import simulate_mlx_on_torch
simulate_mlx_on_torch()

import mlx.core as mx
import mlx.nn as nn
from transformers import TrainerCallback

# The norm-output-cast optimization patches real MLX norm classes; under the
# torch shim the norm classes are `_Noop` with no __dict__. It is irrelevant to
# callback behavior, so neutralize it (pure dtype/memory optimization).
import unsloth_zoo.mlx.trainer as _T
_T._set_norm_output_cast_to_input_dtype = lambda *a, **k: None

# Shim's nn.value_and_grad is "Phase 1": it returns element-0 loss only, but
# MLX (and mlx-lm's trainer) require value_and_grad(model, fn) where fn returns
# a tuple (loss, aux) to return the FULL tuple as the value. Patch to match real
# MLX so the trainer's `(lvalue, toks), grad = ...` unpack works.
import torch as _torch
def _mlx_faithful_value_and_grad(model, fn=None):
    def _make(fn_):
        def _wrapped(*args, **kwargs):
            params = model.parameters() if callable(getattr(model, "parameters", None)) else {}
            from tests.mlx_simulation.mlx_helpers.value_and_grad import (
                _flatten_params, _unflatten_params,
            )
            names, tensors = [], []
            for k, v in _flatten_params(params).items():
                if isinstance(v, _torch.Tensor):
                    v.requires_grad_(True)
                    names.append(k); tensors.append(v)
            out = fn_(*args, **kwargs)
            loss = out[0] if isinstance(out, tuple) else out
            if not tensors:
                return out, {}
            grads = _torch.autograd.grad(loss, tensors, allow_unused=True)
            tree = {n: (g if g is not None else _torch.zeros_like(t))
                    for n, g, t in zip(names, grads, tensors)}
            return out, _unflatten_params(tree)
        return _wrapped
    if fn is None:
        return _make
    return _make(fn)
nn.value_and_grad = _mlx_faithful_value_and_grad

# Neutralize the optimizer step: the shim's torch-backed optimizer wiring is
# fragile for arbitrary models, and gradient realism is irrelevant to callback
# lifecycle testing. Keep step counting so LR/state advance.
import mlx.optimizers as _opt
def _noop_update(self, model, grads):
    self._step = getattr(self, "_step", 0) + 1
_opt._OptimizerBase.update = _noop_update

# Shim lacks int/float in-place dtype promotion: eval accumulator inits
# ntokens=mx.array(0) (int64) then does `ntokens += ntoks` (float) -> torch
# raises. Real MLX promotes. Re-init as float; callback logic is unaffected.
_orig_eval_totals = _T.MLXTrainer._evaluate_batch_totals
def _eval_totals_float(self, eval_batches, loss_fn, is_vlm=False):
    all_losses = mx.array(0.0); ntokens = mx.array(0.0)
    for batch_data in eval_batches:
        batch, lengths, labels = batch_data
        loss, ntoks = loss_fn(self.model, batch, lengths, labels)
        all_losses = all_losses + loss * ntoks
        ntokens = ntokens + ntoks.float()
    return all_losses, ntokens
_T.MLXTrainer._evaluate_batch_totals = _eval_totals_float

from unsloth_zoo.mlx.trainer import (
    MLXTrainer, MLXTrainingConfig, _MLXTrainerState, _MLXTrainerControl,
    _MLXCallbackHandler,
)


def _callback_batch():
    tokens = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    return tokens, mx.array([[0, 4]], dtype=mx.int32), tokens


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def __call__(self, input_ids):
        return self.lm_head(self.embed(input_ids))

    @property
    def state(self):
        # real mlx.nn.Module exposes .state; shim omits it. Provide params dict.
        return self.parameters()

    def train(self, mode=True):
        return self

    def eval(self):
        return self


def _callback_trainer(tmp_path, callbacks, max_steps=3, eval_steps=1,
                      logging_steps=1, save_steps=0, with_eval=False):
    trainer = MLXTrainer(
        model=TinyLM(),
        tokenizer=None,
        train_dataset=[],
        eval_dataset=[{}] if (eval_steps or with_eval) else None,
        args=MLXTrainingConfig(
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=str(tmp_path),
            use_cce=False,
            compile=False,
            gradient_checkpointing=False,
            max_grad_norm=0.0,
            max_grad_leaf_norm=0.0,
            weight_decay=0.0,  # avoid shim Module.update nested-dict gap
            report_to="none",
        ),
        callbacks=callbacks,
    )
    trainer._batches = [_callback_batch()]
    if eval_steps or with_eval:
        trainer._eval_batches_labeled = [_callback_batch()]
    trainer._saved = []
    trainer.save_model = lambda output_dir=None: trainer._saved.append(
        output_dir or trainer.args.output_dir
    )
    return trainer


# ---- 1. Full lifecycle ordering + state -----------------------------------
def test_lifecycle_events_and_state(tmp_path):
    class Recorder(TrainerCallback):
        def __init__(self):
            self.events, self.eval_metrics = [], None
        def on_init_end(self, args, state, control, **k):
            self.events.append(("init", state.global_step))
        def on_train_begin(self, args, state, control, **k):
            self.events.append(("train_begin", state.global_step,
                                k["train_dataloader"] is not None))
        def on_step_begin(self, args, state, control, **k):
            self.events.append(("step_begin", state.global_step))
        def on_optimizer_step(self, args, state, control, **k):
            self.events.append(("optimizer", state.global_step))
        def on_step_end(self, args, state, control, **k):
            self.events.append(("step_end", state.global_step))
        def on_log(self, args, state, control, logs, **k):
            self.events.append(("log", state.global_step, "loss" in logs))
        def on_evaluate(self, args, state, control, metrics, **k):
            self.eval_metrics = dict(metrics)
            self.events.append(("eval", state.global_step))
        def on_epoch_begin(self, args, state, control, **k):
            self.events.append(("epoch_begin",))
        def on_epoch_end(self, args, state, control, **k):
            self.events.append(("epoch_end",))
        def on_train_end(self, args, state, control, **k):
            self.events.append(("train_end", state.global_step))

    class ClassCallback(TrainerCallback):
        calls = []
        def on_train_begin(self, args, state, control, **k):
            type(self).calls.append(state.global_step)

    rec = Recorder()
    ClassCallback.calls = []
    trainer = _callback_trainer(tmp_path, [rec, ClassCallback])
    output = trainer.train()

    names = [e[0] for e in rec.events]
    print("EVENTS:", rec.events)
    # required events present
    for req in ("init", "train_begin", "step_begin", "optimizer", "step_end",
                "log", "eval", "epoch_begin", "epoch_end", "train_end"):
        assert req in names, f"missing {req}"
    # init first, train_begin at step 0, train_end last
    assert rec.events[0] == ("init", 0)
    assert ("train_begin", 0, True) in rec.events
    assert names[-1] == "train_end"
    # per-step ordering: optimizer before step_end
    assert names.index("optimizer") < names.index("step_end")
    # step_begin before optimizer
    assert names.index("step_begin") < names.index("optimizer")
    # class callback instantiated + invoked
    assert ClassCallback.calls == [0]
    # eval metrics delivered
    assert rec.eval_metrics is not None and rec.eval_metrics.get("eval_loss") is not None
    # final state
    assert output.global_step == 3
    assert trainer.state.global_step == 3


# ---- 2. should_training_stop halts training -------------------------------
def test_control_stop(tmp_path):
    class StopAfterFirst(TrainerCallback):
        def __init__(self): self.events = []
        def on_step_end(self, args, state, control, **k):
            self.events.append(state.global_step)
            control.should_training_stop = (state.global_step == 1)
            return control
    cb = StopAfterFirst()
    out = _callback_trainer(tmp_path, [cb], max_steps=5, eval_steps=0).train()
    print("stop events:", cb.events, "final:", out.global_step)
    assert out.global_step == 1
    assert cb.events == [1]


# ---- 3. control.should_evaluate/should_log forced ------------------------
def test_control_force_log_eval(tmp_path):
    class Forcer(TrainerCallback):
        def __init__(self): self.logs, self.evals = [], []
        def on_step_end(self, args, state, control, **k):
            control.should_log = True
            control.should_evaluate = True
            return control
        def on_log(self, args, state, control, logs, **k):
            self.logs.append(state.global_step)
        def on_evaluate(self, args, state, control, metrics, **k):
            self.evals.append(state.global_step)
    cb = Forcer()
    _callback_trainer(tmp_path, [cb], max_steps=2, eval_steps=0,
                      logging_steps=0, with_eval=True).train()
    print("forced logs:", cb.logs, "forced evals:", cb.evals)
    assert cb.evals, "forced eval did not fire"
    assert cb.logs, "forced log did not fire"


# ---- 4. on_save fires only on checkpoints ---------------------------------
def test_on_save(tmp_path):
    import unsloth_zoo.mlx.trainer as T
    class SaveRec(TrainerCallback):
        def __init__(self): self.saves = []
        def on_save(self, args, state, control, **k):
            self.saves.append(state.global_step)
    # neutralize disk writes
    T.save_trainable_adapters = lambda *a, **k: os.makedirs(a[1], exist_ok=True)
    T.save_optimizer_state = lambda *a, **k: None
    T.save_trainer_state = lambda *a, **k: None
    cb = SaveRec()
    tr = _callback_trainer(tmp_path, [cb], max_steps=1, eval_steps=0, save_steps=1)
    tr.train()
    print("saves:", cb.saves)
    assert cb.saves == [1]


# ---- 5. existing function callbacks still work ----------------------------
def test_function_callbacks_preserved(tmp_path):
    steps, evals = [], []
    tr = _callback_trainer(tmp_path, [], max_steps=2, eval_steps=1, logging_steps=1)
    # function step cb: (step, total, loss, lr, tok_s, peak, elapsed, toks, gnorm)
    tr.add_step_callback(lambda step, *a: steps.append(step))
    # function eval cb: (step, val_loss, ppl)
    tr.add_eval_callback(lambda step, *a: evals.append(step))
    tr.train()
    print("fn step cb:", steps, "fn eval cb:", evals)
    assert steps, "function step callback never fired"
    assert evals, "function eval callback never fired"


# ---- 6. exception in a user callback ---------------------------------------
def test_callback_exception_propagates_or_not(tmp_path):
    class Boom(TrainerCallback):
        def on_step_end(self, args, state, control, **k):
            raise RuntimeError("boom in callback")
    tr = _callback_trainer(tmp_path, [Boom()], max_steps=1, eval_steps=0)
    raised = False
    try:
        tr.train()
    except RuntimeError as e:
        raised = True
        print("HF callback exception propagated:", e)
    print("raised:", raised)
    # record behavior; HF Trainer lets callback exceptions propagate
    assert raised, "HF-style callback exception was swallowed (differs from HF)"


# ---- 7. handler unit: control update semantics ----------------------------
def test_handler_control_return_semantics():
    class C(TrainerCallback):
        def on_log(self, args, state, control, **k):
            control.should_save = True
            return control
    class NoReturn(TrainerCallback):
        def on_log(self, args, state, control, **k):
            control.should_evaluate = True
            # returns None -> HF keeps prior control object; here should still see mutation
    h = _MLXCallbackHandler([C(), NoReturn()], None, None, None, None)
    ctrl = _MLXTrainerControl()
    st = _MLXTrainerState()
    args = MLXTrainingConfig()
    out = h.call_event("on_log", args, st, ctrl)
    print("control after:", out)
    assert out.should_save is True


# ---- 8. backwards compat: no callbacks param -------------------------------
def test_backwards_compat_no_callbacks(tmp_path):
    tr = _callback_trainer(tmp_path, [], max_steps=2, eval_steps=0)
    out = tr.train()
    assert out.global_step == 2
    assert isinstance(tr.callback_handler, _MLXCallbackHandler)
    assert tr.callback_handler.callbacks == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
