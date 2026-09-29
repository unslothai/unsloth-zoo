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

"""LoRALinear class + linear_to_lora_layers + load_adapters.

Minimal-viable Phase 1 implementations.  Phase 5 expands to:
- DoRA support (use_dora=True branch)
- Quantized base layer support (LoRALinear wrapping QuantizedLinear)
- adapter_config.json parsing for additional knobs
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class LoRALinear:
    """LoRALinear: base nn.Linear + low-rank adapter (lora_A @ lora_B).

    Forward: y = base(x) + (x @ lora_A.T @ lora_B.T) * scale
    """

    def __init__(self, base, r, alpha, dropout=0.0, scale=None, *, use_dora=False):
        self.linear = base
        in_features = getattr(base, "in_features", None) or base.linear.in_features
        out_features = getattr(base, "out_features", None) or base.linear.out_features
        self.r = r
        self.alpha = alpha
        self.scale = scale if scale is not None else (alpha / r if r else 1.0)
        # MLX init: lora_A kaiming-uniform, lora_B zeros
        self.lora_A = torch.empty(r, in_features)
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = torch.zeros(out_features, r)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.use_dora = use_dora

    def __call__(self, x):
        base_out = self.linear(x)
        x_d = self.dropout(x)
        delta = (x_d @ self.lora_A.T) @ self.lora_B.T
        return base_out + delta * self.scale


def linear_to_lora_layers(model, num_layers, config, use_dora=False):
    """Walk model, replace nn.Linear at the right depths with LoRALinear.

    Phase 1: lightweight impl that scans `model.layers[:num_layers]` if
    that path exists.  PR-A's mlx_loader uses this to wrap fine-tuning
    targets — full implementation is Phase 5.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        return model
    targets = config.get("target_modules", []) if isinstance(config, dict) else []
    rank = config.get("r", 8) if isinstance(config, dict) else 8
    alpha = config.get("alpha", 16.0) if isinstance(config, dict) else 16.0
    for i, layer in enumerate(layers[:num_layers]):
        for tname in targets:
            base = _find_module(layer, tname)
            if base is not None and not isinstance(base, LoRALinear):
                wrapped = LoRALinear(base, rank, alpha, use_dora=use_dora)
                _set_module(layer, tname, wrapped)
    return model


def load_adapters(model, adapter_path):
    """Load LoRA adapter weights from a path."""
    import json
    import os
    from safetensors.torch import load_file

    config_path = os.path.join(adapter_path, "adapter_config.json")
    weights_path = os.path.join(adapter_path, "adapters.safetensors")
    if not os.path.exists(weights_path):
        # Fallback: maybe path itself is the safetensors file.
        if adapter_path.endswith(".safetensors"):
            weights_path = adapter_path
            config_path = None
        else:
            raise FileNotFoundError(
                f"mlx-shim: no adapters.safetensors at {weights_path!r}"
            )

    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    weights = load_file(weights_path)
    # Phase 1: store raw weights on model for tests; Phase 5 walks the
    # adapter keys and applies them to matching LoRALinear modules.
    setattr(model, "_loaded_adapter_weights", weights)
    setattr(model, "_loaded_adapter_config", config)
    return model


def _find_module(parent, dotted):
    cur = parent
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _set_module(parent, dotted, value):
    parts = dotted.split(".")
    cur = parent
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)
