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

import torch
import torch.nn.functional as F

H = 16
I = 48


class GatedMLP(torch.nn.Module):
    def __init__(self, hidden = H, inter = I, out = None, bias = False, dropout = 0.0):
        super().__init__()
        out = hidden if out is None else out
        self.gate_proj = torch.nn.Linear(hidden, inter, bias = bias)
        self.up_proj = torch.nn.Linear(hidden, inter, bias = bias)
        self.down_proj = torch.nn.Linear(inter, out, bias = bias)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return self.down_proj(h)


class TopKMoE(torch.nn.Module):
    def __init__(self, hidden = H, inter = I, n_experts = 4, top_k = 2):
        super().__init__()
        self.top_k = top_k
        self.gate = torch.nn.Linear(hidden, n_experts, bias = False)
        self.experts = torch.nn.ModuleList([GatedMLP(hidden, inter) for _ in range(n_experts)])

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.reshape(-1, h)
        router_logits = self.gate(x)
        w = F.softmax(router_logits, dim = -1, dtype = torch.float)
        w, sel = torch.topk(w, self.top_k, dim = -1)
        w = (w / w.sum(dim = -1, keepdim = True)).to(x.dtype)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = (sel == e).to(x.dtype)
            coef = (w * mask).sum(-1, keepdim = True)
            out = out + coef * expert(x)
        return out.reshape(b, s, h), router_logits


def load_balancing_loss(router_logits, n_experts, top_k):
    p = F.softmax(router_logits.float(), dim = -1)
    _, sel = torch.topk(p, top_k, dim = -1)
    frac = F.one_hot(sel, n_experts).float().sum(1).mean(0)
    return n_experts * (frac * p.mean(0)).sum()


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r = 4):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None: self.base.bias.requires_grad_(False)
        self.lora_A = torch.nn.Linear(base.in_features, r, bias = False)
        self.lora_B = torch.nn.Linear(r, base.out_features, bias = False)
        torch.nn.init.normal_(self.lora_B.weight, std = 0.02)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


def lora_mlp(seed = 0, **kw):
    torch.manual_seed(seed)
    m = GatedMLP(**kw)
    for n in ("gate_proj", "up_proj", "down_proj"):
        setattr(m, n, LoRALinear(getattr(m, n)))
    return m


def grads(module):
    return {n: (None if p.grad is None else p.grad.detach().clone()) for n, p in module.named_parameters()}
