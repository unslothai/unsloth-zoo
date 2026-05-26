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
# MLX nn stub — Module base + common layers + losses + value_and_grad
"""
mlx.nn — neural network primitives.

Skeleton for Phase 1.  Concrete classes (Module, Linear, Embedding,
QuantizedLinear, etc.) and value_and_grad are filled in later phases
via mlx_helpers/.  For now we expose:

  - nn.Module        (lightweight torch.nn.Module facade)
  - nn.Linear        (torch.nn.Linear wrapper)
  - nn.Embedding     (torch.nn.Embedding wrapper)
  - nn.AvgPool2d     (torch.nn.AvgPool2d wrapper)
  - nn.value_and_grad (functional autograd via torch.func.grad_and_value)
  - nn.losses.cross_entropy and friends
  - nn.QuantizedLinear / nn.QuantizedEmbedding (dequantize-on-forward)
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# nn.Module — lightweight wrapper around torch.nn.Module that mimics MLX's
# dict-derived parameter model.  MLX's Module IS a dict; here we keep a
# torch.nn.Module under the hood and expose MLX-flavored API on top.
# ---------------------------------------------------------------------------
class Module:
    """Lightweight stand-in for mlx.nn.Module.

    We subclass nothing here — concrete classes (Linear, Embedding,
    LoRALinear) wrap a torch.nn.Module via composition. Phase 4 fleshes
    out the freeze/unfreeze/state surface; this skeleton is enough for
    isinstance checks and basic attribute access.
    """

    def __init__(self):
        self._mlx_inference_mode = False

    def parameters(self):
        """Return a dict of {leaf_path: tensor} for tree_flatten consumption."""
        out = {}
        for name, child in self.named_modules():
            if child is self:
                continue
            for attr in ("weight", "bias", "scales", "biases"):
                v = getattr(child, attr, None)
                if isinstance(v, torch.Tensor):
                    out[f"{name}.{attr}" if name else attr] = v
        return out

    def trainable_parameters(self):
        return self.parameters()

    def named_modules(self, prefix=""):
        """Yield (path, module) pairs walking the full Module tree.

        Mirrors torch.nn.Module.named_modules: yields self first, then
        every Module-typed child attribute recursively.  Every Module
        attribute counts as a child — we don't track an explicit
        ModuleList, MLX uses plain attribute storage too.
        """
        yield prefix, self
        seen = {id(self)}
        for attr_name, attr_val in vars(self).items():
            # plain Module attribute
            if isinstance(attr_val, Module) and id(attr_val) not in seen:
                seen.add(id(attr_val))
                sub_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
                yield from attr_val.named_modules(sub_prefix)
            # list/tuple of Modules (MLX uses lists for stack of layers)
            elif isinstance(attr_val, (list, tuple)):
                for i, item in enumerate(attr_val):
                    if isinstance(item, Module) and id(item) not in seen:
                        seen.add(id(item))
                        sub_prefix = f"{prefix}.{attr_name}.{i}" if prefix else f"{attr_name}.{i}"
                        yield from item.named_modules(sub_prefix)
            # dict of Modules
            elif isinstance(attr_val, dict):
                for k, item in attr_val.items():
                    if isinstance(item, Module) and id(item) not in seen:
                        seen.add(id(item))
                        sub_prefix = f"{prefix}.{attr_name}.{k}" if prefix else f"{attr_name}.{k}"
                        yield from item.named_modules(sub_prefix)

    def freeze(self, *, recurse=True, keys=None):
        return self

    def unfreeze(self, *, recurse=True, keys=None):
        return self

    def update(self, params):
        """MLX semantics: in-place merge of params dict into module attributes."""
        if isinstance(params, dict):
            for k, v in params.items():
                if "." in k:
                    head, rest = k.split(".", 1)
                    sub = getattr(self, head, None)
                    if isinstance(sub, Module):
                        sub.update({rest: v})
                else:
                    setattr(self, k, v)
        return self

    def update_modules(self, tree):
        """Walk a nested dict/list of new modules and assign them in place.

        ``tree`` is the result of ``tree_unflatten([(path, new_module), ...])``,
        i.e. nested dicts with Module leaves.  Each leaf replaces the
        corresponding attribute on this Module's submodule tree.
        """
        self._apply_module_tree(tree)
        return self

    def _apply_module_tree(self, tree):
        if isinstance(tree, dict):
            for k, v in tree.items():
                if isinstance(v, Module):
                    if k.isdigit() and hasattr(self, "__getitem__"):
                        self[int(k)] = v
                    else:
                        setattr(self, k, v)
                elif isinstance(v, dict):
                    sub = getattr(self, k, None)
                    if isinstance(sub, Module):
                        sub._apply_module_tree(v)
                elif isinstance(v, list):
                    sub = getattr(self, k, None)
                    if isinstance(sub, list):
                        for i, leaf in enumerate(v):
                            if isinstance(leaf, Module):
                                sub[i] = leaf
                            elif isinstance(leaf, (dict, list)) and isinstance(sub[i], Module):
                                sub[i]._apply_module_tree(leaf)

    def apply(self, fn):
        return self

    def eval(self):
        """MLX semantics: force-realize lazy graph (no-op in eager torch)."""
        return None

    def set_inference_mode(self, mode: bool):
        """Torch semantics: switch between train and eval modes."""
        self._mlx_inference_mode = mode
        return self

    def load_weights(self, source, *, strict=True):
        """mlx.nn.Module.load_weights — accept dict or path.

        PR-A monkey-patches this method to allow loading without quant
        state; so we provide a permissive default that just stores
        the dict on `self` for inspection.
        """
        if isinstance(source, str):
            from safetensors.torch import load_file
            source = load_file(source)
        self._loaded_weights = source
        return self

    def save_weights(self, path):
        from safetensors.torch import save_file
        save_file(self.parameters(), path)
        return self


class Linear(Module):
    """mlx.nn.Linear -> torch.nn.Linear adapter.

    MLX stores weight as [out_features, in_features] (same as torch).
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    @property
    def weight(self):
        return self.linear.weight

    @weight.setter
    def weight(self, value):
        if isinstance(value, torch.Tensor):
            self.linear.weight = torch.nn.Parameter(
                value.detach().clone(), requires_grad=self.linear.weight.requires_grad
            )
        else:
            self.linear.weight = value

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        if isinstance(value, torch.Tensor):
            self.linear.bias = torch.nn.Parameter(
                value.detach().clone(), requires_grad=self.linear.bias.requires_grad
            )
        else:
            self.linear.bias = value

    def __call__(self, x):
        return self.linear(x)

    def __contains__(self, key):
        if key == "bias":
            return self.linear.bias is not None
        return False


class Embedding(Module):
    def __init__(self, num_embeddings, dims):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, dims)
        self.num_embeddings = num_embeddings
        self.dims = dims

    @property
    def weight(self):
        return self.embedding.weight

    @weight.setter
    def weight(self, value):
        if isinstance(value, torch.Tensor):
            self.embedding.weight = torch.nn.Parameter(
                value.detach().clone(),
                requires_grad=self.embedding.weight.requires_grad,
            )
        else:
            self.embedding.weight = value

    def __call__(self, x):
        return self.embedding(x)

    def as_linear(self, x):
        """MLX shortcut for tied lm_head: weight @ x."""
        return F.linear(x, self.embedding.weight)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def __call__(self, x):
        return self.pool(x)


# ---------------------------------------------------------------------------
# Quantized layers — dequantize on each forward.  Phase 4 wires these to
# mlx_helpers/quant.py for affine bit-layouts.
# ---------------------------------------------------------------------------
class QuantizedLinear(Module):
    def __init__(self, in_features, out_features, bias=True, group_size=64, bits=4, mode="affine"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        # Storage for packed weight + per-group scales/biases.  Tests
        # populate these directly; production code uses
        # mlx_loader._dequantize_and_replace which constructs them.
        self.weight = None
        self.scales = None
        self.biases = None
        self.bias = None if not bias else None  # placeholder

    def __call__(self, x):
        from .mlx_helpers.quant import dequantize_affine
        if self.mode != "affine":
            raise NotImplementedError(
                f"QuantizedLinear: mode={self.mode!r} not implemented."
            )
        w_fp = dequantize_affine(self.weight, self.scales, self.biases,
                                 self.group_size, self.bits, dtype=x.dtype)
        out = x @ w_fp.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __contains__(self, key):
        return key == "bias" and self.bias is not None


class QuantizedEmbedding(Module):
    def __init__(self, num_embeddings, dims, group_size=64, bits=4, mode="affine"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight = None
        self.scales = None
        self.biases = None

    def __call__(self, x):
        from .mlx_helpers.quant import dequantize_affine
        w_fp = dequantize_affine(self.weight, self.scales, self.biases,
                                 self.group_size, self.bits)
        return F.embedding(x, w_fp)


# ---------------------------------------------------------------------------
# value_and_grad — MLX's `nn.value_and_grad(model, fn)` returns a function
# that computes (loss, grads) where grads is a tree shaped like the model's
# trainable parameters.  See mlx_helpers/value_and_grad.py for the real impl.
# ---------------------------------------------------------------------------
def value_and_grad(model, fn=None):
    """mlx.nn.value_and_grad(model, fn) -> (model_aware_loss_and_grad).

    If fn is None, returns a decorator (the user passes the loss fn after).
    """
    from .mlx_helpers.value_and_grad import nn_value_and_grad
    if fn is None:
        def deco(fn_):
            return nn_value_and_grad(model, fn_)
        return deco
    return nn_value_and_grad(model, fn)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _ce_loss(logits, targets, axis=-1, weights=None, label_smoothing=0.0,
             reduction="mean", **kw):
    if axis != -1:
        # rotate target axis last
        logits = logits.movedim(axis, -1)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).long()
    out = F.cross_entropy(flat_logits, flat_targets, weight=weights,
                          label_smoothing=label_smoothing, reduction=reduction)
    return out


def _mse_loss(predictions, targets, reduction="mean", **kw):
    return F.mse_loss(predictions, targets, reduction=reduction)


def _l1_loss(predictions, targets, reduction="mean", **kw):
    return F.l1_loss(predictions, targets, reduction=reduction)


def _binary_ce_loss(logits, targets, reduction="mean", **kw):
    return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction=reduction)


losses_module = types.ModuleType("mlx.nn.losses")
losses_module.__path__ = []
losses_module.cross_entropy = _ce_loss
losses_module.mse_loss = _mse_loss
losses_module.l1_loss = _l1_loss
losses_module.binary_cross_entropy = _binary_ce_loss


# ---------------------------------------------------------------------------
# Initializers (returns callables that fill a tensor of given shape)
# ---------------------------------------------------------------------------
def _init_constant(value):
    def _init(shape, dtype=torch.float32):
        return torch.full(shape, value, dtype=dtype)
    return _init


def _init_normal(mean=0.0, std=1.0):
    def _init(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype).normal_(mean, std)
    return _init


def _init_uniform(low=0.0, high=1.0):
    def _init(shape, dtype=torch.float32):
        return torch.empty(shape, dtype=dtype).uniform_(low, high)
    return _init


init_module = types.ModuleType("mlx.nn.init")
init_module.__path__ = []
init_module.constant = _init_constant
init_module.normal = _init_normal
init_module.uniform = _init_uniform


# ---------------------------------------------------------------------------
# Module-level __getattr__: any unknown nn.X returns _Noop.
# ---------------------------------------------------------------------------
from . import mlx_stub  # for _Noop

__path__ = []


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return mlx_stub._Noop(f"mlx.nn.{name}")


# nn.utils.clip_grad_value_ — passthrough
class _NNUtils:
    @staticmethod
    def clip_grad_value_(parameters, clip_value):
        torch.nn.utils.clip_grad_value_(
            parameters if not callable(parameters) else parameters(),
            clip_value,
        )


utils_module = types.ModuleType("mlx.nn.utils")
utils_module.__path__ = []
utils_module.clip_grad_value_ = _NNUtils.clip_grad_value_


def inject_into_sys_modules():
    this = sys.modules[__name__]
    this.losses = losses_module
    this.init = init_module
    this.utils = utils_module
    sys.modules.update({
        "mlx.nn": this,
        "mlx.nn.losses": losses_module,
        "mlx.nn.init": init_module,
        "mlx.nn.utils": utils_module,
    })
    if "mlx" in sys.modules:
        setattr(sys.modules["mlx"], "nn", this)
