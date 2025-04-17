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

__all__ = [
    "get_peft_regex",
    "merge_and_overwrite_lora",
    "merge_and_dequantize_lora",
    "SKIP_QUANTIZATION_MODULES",
    "get_lora_layer_modules",
    "requires_grad_for_gradient_checkpointing",
]

import inspect
import os
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import torch

# Skip some modules sensitive to quantization
SKIP_QUANTIZATION_MODULES = [
    "lm_head",
    "multi_modal_projector",  # Llama 3.2 Vision, Pixtral, Llava
    "merger",  # Qwen2 VL
    "modality_projection",  # Idefics, SmolVLM
    "router",  # MoE Router
    "gate",  # MoE Router
    "mamba",
]


def get_peft_regex(
    model,
    finetune_vision_layers: bool = True,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    target_modules: List[str] = None,
    vision_tags: List[str] = [
        "vision",
        "image",
        "visual",
        "patch",
    ],
    language_tags: List[str] = [
        "language",
        "text",
    ],
    attention_tags: List[str] = [
        "self_attn",
        "attention",
        "attn",
    ],
    mlp_tags: List[str] = [
        "mlp",
        "feed_forward",
        "ffn",
        "dense",
    ],
) -> str:
    """
    Build a **safe** regular‑expression that matches ONLY the *leaf*
    `torch.nn.Linear` layers we want to adapt with LoRA.

    The previous implementation matched any module path that merely
    *contained* one of the projection names; after fused‑projection
    rewrites this included helpers such as
    `model.layers.3.mlp.gate_up_proj → ModuleDict`, which PEFT cannot
    patch.  We now anchor the name to the **last dot‑separated field**
    so only genuine linear layers match.
    """
    # — sanity checks --------------------------------------------------
    if not (finetune_vision_layers or finetune_language_layers):
        raise RuntimeError(
            "Select at least one of vision / language layers to finetune."
        )
    if not (finetune_attention_modules or finetune_mlp_modules):
        raise RuntimeError(
            "Select at least one of attention / MLP modules to finetune."
        )

    # — collect all leaf‑names of *linear* layers ----------------------
    linear_modules = [
        name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)
    ]
    leaf_names = [path.rsplit(".", 1)[-1] for path in linear_modules]
    leaf_counts = Counter(leaf_names)

    if target_modules is None:
        # keep names that appear in *more* than one place
        # (single‑occurrence heads are usually lm_head / projectors)
        candidate_leafs = [n for n, c in leaf_counts.items() if c > 1]
    else:
        if not isinstance(target_modules, list):
            raise TypeError("`target_modules` must be a list of strings.")
        candidate_leafs = list(target_modules)

    # — assemble regex parts ------------------------------------------
    def _join(xs):
        return "|".join(map(re.escape, xs)) or "$^"  # empty → no match

    # which *part* of the model path (vision/language)
    model_part_pat = (
        _join(vision_tags if finetune_vision_layers else [])
        + "|"
        + _join(language_tags if finetune_language_layers else "")
    )
    # which *sub‑module* inside the block (attn/mlp)
    component_pat = (
        _join(attention_tags if finetune_attention_modules else [])
        + "|"
        + _join(mlp_tags if finetune_mlp_modules else "")
    )

    # exact leaf names – anchor to “preceded by dot or start” AND “end of string”
    leaf_pat = r"(?:(?<=\.)|^)(?:" + _join(candidate_leafs) + r")$"

    # full matcher
    regex_matcher = (
        r".*?(?:" + model_part_pat + r")"  # vision / language part
        r".*?(?:" + component_pat + r")"  # attn / mlp component
        r".*?" + leaf_pat  # leaf linear layer
    )

    # also allow Qwen‑style `model.layers.0.self_attn.q_proj` paths
    if finetune_language_layers:
        regex_matcher = (
            regex_matcher
            + "|"
            + r"(?:\bmodel\.layers\.\d+\.(?:"
            + component_pat
            + r")\."
            + leaf_pat
            + ")"
        )

    # — verify we actually hit something ------------------------------
    if not any(re.search(regex_matcher, n) for n in linear_modules):
        raise RuntimeError(
            f"Unsloth: the generated regex matched **no** linear layers "
            f"in {model.__class__.__name__}.  "
            f"Check your *tags* / *target_modules* settings."
        )
    return regex_matcher


pass


def get_lora_layer_modules():
    # All Unsloth Zoo code licensed under LGPLv3
    import peft.tuners.lora

    path = os.path.split(peft.tuners.lora.__file__)[0]
    files = os.listdir(path)

    Linear_LoRA_Layers = []
    for file in files:
        if file == "__init__.py" or not file.endswith(".py"):
            continue
        item = f"peft.tuners.lora.{file[:-len('.py')]}"
        exec(f"import {item}", locals(), globals())
        modules = dir(eval(item))
        modules = [x for x in modules if x.startswith("Linear") or x.endswith("Linear")]
        if len(modules) == 0:
            continue
        exec(f"from {item} import ({', '.join(modules)})", locals(), globals())
        Linear_LoRA_Layers += [
            (
                eval(x),
                item,
                x,
            )
            for x in modules
        ]
    pass
    return tuple(Linear_LoRA_Layers)


pass


def requires_grad_for_gradient_checkpointing(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Enables requires_grad to make gradient checkpointing work on
    # non language models that don't just use .embed_tokens
    def register_other_hooks(name1, name2, module, _hooks):
        old_hooks = eval(f"module.{_hooks}")
        other_hooks = []
        for value in old_hooks.values():
            qualname = getattr(value, "__qualname__", "")
            name = getattr(value, "__name__", "")
            if name1 in qualname or name2 in qualname:
                pass
            elif name2 in name or name2 in name:
                pass
            else:
                other_hooks.append(value)
        pass
        # Keep none input requires grad hooks
        exec(f"module.{_hooks} = OrderedDict()")
        for hook in other_hooks:
            exec(f"module.register{_hooks[:-1]}(hook)")
        pass

    pass

    # Remove all previous forward hooks for gradient checkpointing
    for name, module in model.named_modules():
        if len(module._forward_hooks) != 0:
            register_other_hooks(
                "enable_input_require_grads",
                "make_inputs_require_grad",
                module,
                "_forward_hooks",
            )
        pass
    pass

    # Add post forward hook
    def requires_grad_post_hook(module, input, output):
        type_output = type(output)
        if type_output is torch.Tensor:
            output.requires_grad_(True)
        else:
            try:  # For dataclass from HF, try on loss or logits
                if hasattr(output, "loss") and output.loss is not None:
                    output.loss.requires_grad_(True)
                elif (
                    hasattr(output, "logits") and output.logits is not None
                ):  # with RL like GRPO there are no loss as you don't provide labels
                    output.logits.requires_grad_(True)
                else:
                    raise ValueError(
                        "Neither loss nor logits are available for grad post hook."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Unsloth: Failed to make output require gradients: {e}"
                )

    pass

    def requires_grad_pre_hook(module, input):
        type_input = type(input)
        if type_input is torch.Tensor:
            input.requires_grad_(True)
        elif type_input is tuple or type_input is list:
            if len(input) == 0:
                raise RuntimeError("Unsloth: Failed to make input require gradients!")
                # print(f"  WARNING: Empty list input to {module.__class__.__name__}!") #
                # return
            if torch.is_floating_point(input[0]):
                input[0].requires_grad_(True)
        else:
            raise RuntimeError("Unsloth: Failed to make input require gradients!")

    pass

    # Find 1st ever item which requires grad
    param = None
    for name, param in model.named_parameters():
        if param.requires_grad:
            break
    if param is None:
        return

    name = re.sub(r"\.([\d]{1,})\.", r"[\1].", name)
    name_components = name.split(".")

    if len(name_components) == 0:
        raise RuntimeError("Unsloth: Model has 0 layers?")

    final_where = None
    # Try getting previous parent module
    for j in range(len(name_components) - 1, 0, -1):
        name_curr = name_components[j]
        name_pre = "model." + ".".join(name_components[:j])
        # Disable [\d] since it fails in gradient checkpointing
        if re.search(r"\[[\d]{1,}\]", name_pre):
            continue
        module = eval(name_pre)
        if hasattr(module, "forward"):
            try:
                forward = inspect.getsource(module.forward)
            except:
                continue

            # Normal self.language_model(...)
            if f"self.{name_curr}(" in forward:
                final_where = j + 1
                break

            # Fix self.blocks[0] like in Qwen
            module_list = re.sub(r"\[[\d]{1,}\]", "", name_curr)
            if f"in self.{module_list}:" in forward:
                final_where = j
                break
            elif (
                re.search(r"for [^\s]{3,} in self\." + module_list, forward) is not None
            ):
                # Might have failed finding self.layers: like self.layers[...]:
                final_where = j
                break
            pass
        pass
    pass

    if final_where is None:
        # Find all input embeddings and just set them all as a fallback!
        # Add other hooks first
        register_other_hooks(
            "requires_grad_post_hook",
            "requires_grad_post_hook",
            module,
            "_forward_hooks",
        )
        module.register_forward_hook(requires_grad_post_hook)
        return
    pass

    module_name = "model." + ".".join(name_components[:final_where])
    module = eval(module_name)

    if hasattr(module, "config") and (
        module.config.__class__.__name__
        in (
            "CLIPVisionConfig",
            "SiglipVisionConfig",
        )
    ):
        # CLIP - backtrack to get_input_embeddings since requires_grad fails!
        old_module = model
        for module_name, module in model.named_modules():
            if not hasattr(module, "get_input_embeddings"):
                break
            old_module = module
        module = old_module
    pass
    print(f"Unsloth: Making `{module_name}` require gradients")

    still_need_patching = True
    # Check if input_embeddings exists
    if hasattr(module, "get_input_embeddings"):
        # Use forward hook after Embedding() is called
        try:
            module = module.get_input_embeddings()
            # Add other hooks first
            register_other_hooks(
                "requires_grad_post_hook",
                "requires_grad_post_hook",
                module,
                "_forward_hooks",
            )
            module.register_forward_hook(requires_grad_post_hook)
            still_need_patching = False
        except:
            # Not Implemented probably?
            still_need_patching = True
    pass

    if still_need_patching:
        # Use forward pre hook before module is called
        register_other_hooks(
            "requires_grad_pre_hook",
            "requires_grad_pre_hook",
            module,
            "_forward_pre_hooks",
        )
        module.register_forward_pre_hook(requires_grad_pre_hook)
    pass


pass

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
