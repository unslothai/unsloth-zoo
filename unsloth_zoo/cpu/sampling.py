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

"""CPU sampling (generate's do_sample, GRPO rollouts): torch.multinomial with one draw, same values, faster.

torch's one-draw path is argmax(p / q) with q ~ Exp(1) from exponential_. Recent CPU builds fill q with a serial
per-element log: about 5 ms per token for two rows of a 152k vocabulary. q is -log1p(-u) with u a float64 uniform,
so a float64 uniform_ and a vectorized log1p give the same draws, and the generator ends in the same state.
The vectorized log1p can differ from libm's by one float64 ulp, which survives the float32 cast about once in 2^29
draws and changes a sample only if it also reorders the top two p / q; float32 probabilities (what generate passes)
only, float64 keeps torch's call. A one-time check against exponential_ decides whether the fast path is used on this
build; any mismatch keeps torch.multinomial. UNSLOTH_CPU_FAST_MULTINOMIAL=0 turns it off."""

import functools
import os

import torch

__all__ = [
    "fast_multinomial_ok",
    "install",
    "ENGAGED",
]

ENGAGED = [0]

# Below this many probabilities the stock call is as fast; the gate keeps tiny draws on torch's own path.
_MIN_NUMEL = 4096
_CHECKED = [None]


def _enabled():
    return os.environ.get("UNSLOTH_CPU_FAST_MULTINOMIAL", "1") != "0"


def _exponential(shape, dtype, generator):
    """Exp(1) as torch's CPU exponential_ forms it: -1.0 / lambda * log1p(-u), u a float64 uniform, cast."""
    # device = "cpu": under torch.set_default_device("cuda" / "mps") a bare rand would not be on the input's device.
    u = torch.rand(shape, dtype = torch.float64, device = "cpu", generator = generator)
    return torch.log1p(u.neg_()).neg_().to(dtype)


def fast_multinomial_ok():
    """Whether this build's float32 exponential_ equals _exponential on a 2 x 152k draw and consumes the generator alike."""
    if _CHECKED[0] is None:
        ok = True
        try:
            for shape in ((1, 37), (2, 151936)):
                a = torch.Generator(device = "cpu").manual_seed(3407)
                b = torch.Generator(device = "cpu").manual_seed(3407)
                q = torch.empty(shape, dtype = torch.float32, device = "cpu").exponential_(generator = a)
                r = _exponential(shape, torch.float32, b)
                if not torch.equal(q, r) or not torch.equal(a.get_state(), b.get_state()):
                    ok = False
                    break
        except Exception:
            ok = False
        _CHECKED[0] = ok
    return _CHECKED[0]


def _wrap(original):
    @functools.wraps(original)
    def multinomial(input, num_samples, replacement = False, *, generator = None, out = None):
        # Compiling first: a shape test ahead of it would add a guard (and break a marked dynamic range), and both
        # branches call torch's multinomial there anyway. Exact int / bool arguments: anything else gets torch's error.
        if (
            not torch.compiler.is_compiling()
            and out is None
            and type(num_samples) is int
            and num_samples == 1
            and type(replacement) is bool
            and type(input) is torch.Tensor
            and input.device.type == "cpu"
            and input.dtype == torch.float32
            and input.dim() in (1, 2)
            and input.numel() >= _MIN_NUMEL
            and input.is_contiguous()
            and not input.requires_grad
            and _enabled()
            and fast_multinomial_ok()
        ):
            # torch's own checks, before any draw; a failing input goes to torch so the error is its own.
            low, high = torch.aminmax(input)
            sums = input.sum(-1)
            if bool(low >= 0) and bool(high < float("inf")) and not bool((sums == 0).any()):
                q = _exponential(input.shape, input.dtype, generator)
                ENGAGED[0] += 1
                return torch.div(input, q, out = q).argmax(-1, keepdim = True)
        return original(input, num_samples, replacement, generator = generator, out = out)

    multinomial._unsloth_cpu_fast_multinomial = True
    multinomial._unsloth_original = original
    return multinomial


def install():
    """Wraps torch.multinomial once (CPU, one draw, float probabilities; everything else is the stock call)."""
    current = torch.multinomial
    if getattr(current, "_unsloth_cpu_fast_multinomial", False):
        return True
    if not _enabled() or not fast_multinomial_ok():
        return False
    torch.multinomial = _wrap(current)
    return True
