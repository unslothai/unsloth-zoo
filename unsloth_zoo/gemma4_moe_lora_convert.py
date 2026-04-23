# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Convert an Unsloth-trained Gemma 4 MoE LoRA adapter to the per-expert layout
that vLLM's PEFTHelper validates against.

Unsloth trains MoE LoRA on the stacked ``Gemma4TextExperts.gate_up_proj`` /
``down_proj`` 3D tensors via ParamWrapper, so the saved adapter has
block-diagonal stacked LoRA matrices (one per layer):

    experts.base_layer.lora_A   (E*r, H)          # gate_up
    experts.base_layer.lora_B   (2*I, E*r)        # gate_up
    experts.lora_A              (E*r, I)          # down
    experts.lora_B              (H, E*r)          # down

vLLM's Gemma 4 `FusedMoE` LoRA support expects per-expert entries:

    moe.experts.{e}.gate_proj.lora_A   (r, H)
    moe.experts.{e}.gate_proj.lora_B   (I, r)
    moe.experts.{e}.up_proj.lora_A     (r, H)
    moe.experts.{e}.up_proj.lora_B     (I, r)
    moe.experts.{e}.down_proj.lora_A   (r, I)
    moe.experts.{e}.down_proj.lora_B   (H, r)

gate and up share the same lora_A since the training-side LoRA was on the
merged gate_up projection; only lora_B splits into ``[:I, :]`` and ``[I:, :]``.

Usage:

    python -m unsloth_zoo.gemma4_moe_lora_convert \\
        --src outputs/gemma4_26b_a4b_lora_r16 \\
        --dst outputs/gemma4_26b_a4b_lora_r16_per_expert
"""
import argparse
import json
import os
import re
import shutil
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_STACKED_GATE_UP_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.\d+)\.experts\.base_layer\.lora_(?P<ab>A|B)\.weight$"
)
_STACKED_DOWN_RE = re.compile(
    r"^(?P<prefix>.*\.layers\.\d+)\.experts\.lora_(?P<ab>A|B)\.weight$"
)


def _split_expert_slice(tensor: torch.Tensor, dim: int, e: int, r: int) -> torch.Tensor:
    start, end = e * r, (e + 1) * r
    return tensor.narrow(dim, start, r).clone()


def convert(
    src_dir: str,
    dst_dir: str,
    num_experts: int,
    intermediate_size: int,
    rank: int,
) -> None:
    """Convert a stacked-MoE adapter directory to per-expert layout."""
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname in ("adapter_model.safetensors", "adapter_config.json"):
            continue
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    with open(os.path.join(src_dir, "adapter_config.json"), "r") as f:
        cfg = json.load(f)

    # Strip the Unsloth stacked-MoE target_parameters — per-expert keys are
    # picked up through the standard target_modules match (gate_proj/up_proj/
    # down_proj) once the adapter is exploded.
    cfg.pop("target_parameters", None)
    # Drop "gate_up_proj" from target_modules — split into gate+up per expert.
    cfg["target_modules"] = [
        m for m in cfg.get("target_modules", []) if m != "gate_up_proj"
    ]
    # Add per-expert gate/up/down_proj if missing, so PEFT/vLLM match them
    # against the expanded expert modules.
    for m in ("gate_proj", "up_proj", "down_proj"):
        if m not in cfg["target_modules"]:
            cfg["target_modules"].append(m)
    cfg["target_modules"] = sorted(set(cfg["target_modules"]))
    cfg.setdefault("auto_mapping", {})
    cfg["auto_mapping"]["unsloth_moe_lora_expanded"] = True

    with open(os.path.join(dst_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    new_state: Dict[str, torch.Tensor] = {}

    with safe_open(os.path.join(src_dir, "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():
            m_gu = _STACKED_GATE_UP_RE.match(key)
            m_dn = _STACKED_DOWN_RE.match(key)
            tensor = f.get_tensor(key)

            if m_gu is not None:
                prefix = m_gu.group("prefix")  # .../layers.N
                ab = m_gu.group("ab")
                if ab == "A":
                    assert tensor.shape[0] == num_experts * rank, (
                        f"expected A shape[0]={num_experts * rank}, got {tensor.shape}"
                    )
                    for e in range(num_experts):
                        sub = _split_expert_slice(tensor, dim=0, e=e, r=rank)  # (r, H)
                        # gate and up share lora_A of the merged gate_up_proj.
                        # safetensors refuses tensors that share storage, so
                        # clone into independent allocations for each proj.
                        for proj in ("gate_proj", "up_proj"):
                            k = f"{prefix}.experts.{e}.{proj}.lora_A.weight"
                            new_state[k] = sub.contiguous().clone()
                else:  # B
                    assert tensor.shape[1] == num_experts * rank, (
                        f"expected B shape[1]={num_experts * rank}, got {tensor.shape}"
                    )
                    I = intermediate_size
                    assert tensor.shape[0] == 2 * I, (
                        f"expected B shape[0]=2*I={2 * I}, got {tensor.shape}"
                    )
                    for e in range(num_experts):
                        sub = _split_expert_slice(tensor, dim=1, e=e, r=rank)  # (2I, r)
                        gate_B = sub[:I, :].contiguous()
                        up_B = sub[I:, :].contiguous()
                        new_state[f"{prefix}.experts.{e}.gate_proj.lora_B.weight"] = gate_B
                        new_state[f"{prefix}.experts.{e}.up_proj.lora_B.weight"] = up_B
                continue

            if m_dn is not None:
                prefix = m_dn.group("prefix")
                ab = m_dn.group("ab")
                if ab == "A":
                    assert tensor.shape[0] == num_experts * rank, (
                        f"expected down A shape[0]={num_experts * rank}, got {tensor.shape}"
                    )
                    for e in range(num_experts):
                        sub = _split_expert_slice(tensor, dim=0, e=e, r=rank)  # (r, I)
                        new_state[f"{prefix}.experts.{e}.down_proj.lora_A.weight"] = sub.contiguous()
                else:  # B
                    assert tensor.shape[1] == num_experts * rank, (
                        f"expected down B shape[1]={num_experts * rank}, got {tensor.shape}"
                    )
                    for e in range(num_experts):
                        sub = _split_expert_slice(tensor, dim=1, e=e, r=rank)  # (H, r)
                        new_state[f"{prefix}.experts.{e}.down_proj.lora_B.weight"] = sub.contiguous()
                continue

            # vLLM's Gemma 4 vision tower has no PunicaWrapper-compatible
            # LoRA path (warns "no matching PunicaWrapper" for every encoder
            # layer). Adapter tensors under vision_tower.* also carry a
            # ".linear" suffix from the bnb Linear4bit wrapper path, which
            # vLLM's validator rejects because "linear" isn't a target
            # module. Drop them — they can't be applied anyway.
            if ".vision_tower." in key or ".audio_tower." in key:
                continue

            # Non-MoE entries (attention projections, dense MLP) pass through.
            new_state[key] = tensor.clone()

    save_file(new_state, os.path.join(dst_dir, "adapter_model.safetensors"))
    print(
        f"[gemma4-moe-lora-convert] wrote {len(new_state)} tensors to "
        f"{dst_dir}/adapter_model.safetensors"
    )


def _infer_shapes_from_adapter(src_dir: str) -> Dict[str, int]:
    """Read one stacked A/B tensor to infer num_experts, rank, intermediate_size."""
    path = os.path.join(src_dir, "adapter_model.safetensors")
    with safe_open(path, "pt") as f:
        rank = None
        num_experts = None
        intermediate_size = None
        for key in f.keys():
            m = _STACKED_GATE_UP_RE.match(key)
            if m is None:
                continue
            t = f.get_tensor(key)
            if m.group("ab") == "B":
                two_i, er = t.shape  # (2I, E*r)
                # Need rank from A below.
                intermediate_size = two_i // 2
            else:
                er, _h = t.shape  # (E*r, H)
                # can't split E and r yet
                er_from_a = er
        # Second pass with adapter_config for rank
        with open(os.path.join(src_dir, "adapter_config.json"), "r") as g:
            cfg = json.load(g)
        rank = int(cfg["r"])
        num_experts = er_from_a // rank
    return {
        "rank": rank,
        "num_experts": num_experts,
        "intermediate_size": intermediate_size,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="source adapter directory")
    ap.add_argument("--dst", required=True, help="destination adapter directory")
    ap.add_argument("--num_experts", type=int, default=None)
    ap.add_argument("--intermediate_size", type=int, default=None)
    ap.add_argument("--rank", type=int, default=None)
    args = ap.parse_args()

    auto = _infer_shapes_from_adapter(args.src)
    num_experts = args.num_experts if args.num_experts is not None else auto["num_experts"]
    intermediate_size = (
        args.intermediate_size
        if args.intermediate_size is not None
        else auto["intermediate_size"]
    )
    rank = args.rank if args.rank is not None else auto["rank"]

    print(
        f"[gemma4-moe-lora-convert] src={args.src}  dst={args.dst}  "
        f"num_experts={num_experts}  intermediate_size={intermediate_size}  "
        f"rank={rank}"
    )
    convert(args.src, args.dst, num_experts, intermediate_size, rank)


if __name__ == "__main__":
    main()
