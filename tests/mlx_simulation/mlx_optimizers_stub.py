# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Unsloth Zoo - Utilities for Unsloth
# MLX optimizers stub — wraps torch.optim.* via the functional update API.
"""
mlx.optimizers — Adam/AdamW/SGD/Adafactor/Muon/Lion + schedulers.

MLX optimizers are functional: `optimizer.update(model, grads)` reads
the current params from `model.parameters()`, applies updates, and
writes back via `model.update(new_params)`.  Torch optimizers are
stateful: they hold parameter references and call `.step()`.

Phase 4 fleshes out the wrapping; for Phase 1 we provide skeletons
that capture lr/betas etc. and delegate to torch.optim.
"""

from __future__ import annotations

import math
import sys
import types

import torch


class _OptimizerBase:
    """Base for MLX-flavored optimizer adapters."""

    def __init__(self, learning_rate=1e-3, **kw):
        self.learning_rate = learning_rate
        self._kw = kw
        self._torch_opt = None  # constructed lazily on first update()
        self._step = 0

    @property
    def state(self):
        return {"step": self._step}

    @state.setter
    def state(self, s):
        self._step = s.get("step", 0) if isinstance(s, dict) else self._step

    def _build_torch_opt(self, params):
        raise NotImplementedError

    def update(self, model, grads):
        """Functional update — write new params back to the model.

        For Phase 1, this collects flat parameters and calls
        torch.optim.X.step() once.  Phase 4 extends to MLX's exact tree
        semantics.
        """
        # Gather parameters as a flat list with stable ordering.
        from .mlx_utils_stub import tree_flatten
        params = [p for _, p in tree_flatten(model.parameters() if callable(getattr(model, "parameters", None)) else model)]
        flat_grads = [g for _, g in tree_flatten(grads)]
        if self._torch_opt is None:
            self._torch_opt = self._build_torch_opt(params)
        # Assign grads
        for p, g in zip(params, flat_grads):
            if isinstance(p, torch.Tensor) and isinstance(g, torch.Tensor):
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.copy_(g.detach())
        self._torch_opt.step()
        self._torch_opt.zero_grad(set_to_none=False)
        self._step += 1


class SGD(_OptimizerBase):
    def _build_torch_opt(self, params):
        return torch.optim.SGD(params, lr=self.learning_rate,
                               momentum=self._kw.get("momentum", 0.0),
                               weight_decay=self._kw.get("weight_decay", 0.0),
                               dampening=self._kw.get("dampening", 0.0),
                               nesterov=self._kw.get("nesterov", False))


class Adam(_OptimizerBase):
    def _build_torch_opt(self, params):
        return torch.optim.Adam(params, lr=self.learning_rate,
                                betas=self._kw.get("betas", (0.9, 0.999)),
                                eps=self._kw.get("eps", 1e-8),
                                weight_decay=self._kw.get("weight_decay", 0.0))


class AdamW(_OptimizerBase):
    def _build_torch_opt(self, params):
        return torch.optim.AdamW(params, lr=self.learning_rate,
                                 betas=self._kw.get("betas", (0.9, 0.999)),
                                 eps=self._kw.get("eps", 1e-8),
                                 weight_decay=self._kw.get("weight_decay", 0.01))


class Adamax(_OptimizerBase):
    def _build_torch_opt(self, params):
        return torch.optim.Adamax(params, lr=self.learning_rate,
                                  betas=self._kw.get("betas", (0.9, 0.999)),
                                  eps=self._kw.get("eps", 1e-8))


class RMSprop(_OptimizerBase):
    def _build_torch_opt(self, params):
        return torch.optim.RMSprop(params, lr=self.learning_rate,
                                   alpha=self._kw.get("alpha", 0.99),
                                   eps=self._kw.get("eps", 1e-8))


class Adagrad(_OptimizerBase):
    def _build_torch_opt(self, params):
        return torch.optim.Adagrad(params, lr=self.learning_rate,
                                   eps=self._kw.get("eps", 1e-10))


class AdaDelta(_OptimizerBase):
    def _build_torch_opt(self, params):
        return torch.optim.Adadelta(params, lr=self.learning_rate,
                                    rho=self._kw.get("rho", 0.9),
                                    eps=self._kw.get("eps", 1e-6))


class Adafactor(_OptimizerBase):
    """MLX's Adafactor — wrap transformers.optimization.Adafactor if present."""
    def _build_torch_opt(self, params):
        try:
            from transformers.optimization import Adafactor as _Adafactor
        except ImportError:
            raise ImportError(
                "mlx-shim: optim.Adafactor needs `transformers` for the torch wrap."
            )
        return _Adafactor(params, lr=self.learning_rate,
                          beta1=self._kw.get("beta1", None),
                          weight_decay=self._kw.get("weight_decay", 0.0))


class Lion(_OptimizerBase):
    """Optional torch lion — uses lion-pytorch if available, else falls back."""
    def _build_torch_opt(self, params):
        try:
            from lion_pytorch import Lion as _Lion
        except ImportError:
            # Fallback to AdamW so the test suite continues — log loud.
            import warnings
            warnings.warn("mlx-shim: optim.Lion fell back to AdamW; "
                          "install lion-pytorch for the real thing.")
            return torch.optim.AdamW(params, lr=self.learning_rate,
                                     weight_decay=self._kw.get("weight_decay", 0.0))
        return _Lion(params, lr=self.learning_rate,
                     betas=self._kw.get("betas", (0.9, 0.99)),
                     weight_decay=self._kw.get("weight_decay", 0.0))


class Muon(_OptimizerBase):
    def _build_torch_opt(self, params):
        # Muon is a recent optimizer; if not installed, fall back to SGD.
        try:
            from muon import Muon as _Muon  # placeholder package name
            return _Muon(params, lr=self.learning_rate)
        except ImportError:
            import warnings
            warnings.warn("mlx-shim: optim.Muon fell back to SGD with momentum.")
            return torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9)


class MultiOptimizer:
    """MLX's MultiOptimizer routes different param groups to different optimizers."""
    def __init__(self, *optimizers, filters=None):
        self.optimizers = list(optimizers)
        self.filters = filters or [lambda *_: True] * len(optimizers)

    def update(self, model, grads):
        # Phase 4: route to the right optimizer per filter.  For now, just
        # apply the first.
        if self.optimizers:
            self.optimizers[0].update(model, grads)


# ---------------------------------------------------------------------------
# Schedulers — MLX returns lr-schedule callables (step -> lr) directly.
# ---------------------------------------------------------------------------
def linear_schedule(init, end, steps):
    def _sched(step):
        if step >= steps:
            return end
        return init + (end - init) * step / max(1, steps)
    return _sched


def cosine_decay(init, decay_steps, end=0.0):
    def _sched(step):
        if step >= decay_steps:
            return end
        return end + 0.5 * (init - end) * (1 + math.cos(math.pi * step / decay_steps))
    return _sched


def exponential_decay(init, decay_rate):
    def _sched(step):
        return init * (decay_rate ** step)
    return _sched


def step_decay(init, decay_rate, step_size):
    def _sched(step):
        return init * (decay_rate ** (step // step_size))
    return _sched


def join_schedules(schedules, boundaries):
    def _joined(step):
        for boundary, sched in zip(boundaries, schedules):
            if step < boundary:
                return sched(step)
        return schedules[-1](step - (boundaries[-1] if boundaries else 0))
    return _joined


def clip_grad_norm(grads, max_norm):
    """mlx.optimizers.clip_grad_norm — flat-tree version."""
    from .mlx_utils_stub import tree_flatten, tree_unflatten
    flat = tree_flatten(grads)
    tensors = [v for _, v in flat if isinstance(v, torch.Tensor)]
    if not tensors:
        return grads, 0.0
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(t.detach()) for t in tensors])
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for t in tensors:
            t.detach().mul_(clip_coef)
    return grads, total_norm.item()


def decay_weight(weights, decay):
    """Apply weight decay (returns updated tree)."""
    from .mlx_utils_stub import tree_map
    return tree_map(lambda w: w * (1.0 - decay) if isinstance(w, torch.Tensor) else w, weights)


# ---------------------------------------------------------------------------
schedulers_module = types.ModuleType("mlx.optimizers.schedulers")
schedulers_module.__path__ = []
schedulers_module.linear_schedule = linear_schedule
schedulers_module.cosine_decay = cosine_decay
schedulers_module.exponential_decay = exponential_decay
schedulers_module.step_decay = step_decay
schedulers_module.join_schedules = join_schedules


__path__ = []


def __getattr__(name):
    from .mlx_stub import _Noop
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _Noop(f"mlx.optimizers.{name}")


def inject_into_sys_modules():
    this = sys.modules[__name__]
    sys.modules.update({
        "mlx.optimizers": this,
        "mlx.optimizers.schedulers": schedulers_module,
    })
    if "mlx" in sys.modules:
        setattr(sys.modules["mlx"], "optimizers", this)
