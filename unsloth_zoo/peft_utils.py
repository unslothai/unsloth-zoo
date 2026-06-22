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
import torch
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
from collections import OrderedDict
import re
from .log import logger

# Skip some modules sensitive to quantization
SKIP_QUANTIZATION_MODULES = [
    "lm_head",
    "multi_modal_projector",    # Llama 3.2 Vision, Pixtral, Llava
    "merger",                   # Qwen2 VL
    "modality_projection",      # Idefics, SmolVLM
    "router",                   # MoE Router
    "mlp.gate",                 # MoE Router
    "block_sparse_moe.gate",    # MoE Router
    'mamba',
    "audio_tower",              # Gemma3N audio encoder conformer
    "vision_tower",             # Gemma3 vision encoder (SigLIP)
    "vision_embedder",          # multimodal embedders kept in full precision
    "embed_vision",
    "embed_audio",
    "score",                    # *ForSequenceClassification head
    "classifier",               # *ForTokenClassification, *ForImageClassification, BERT-family head
    "qa_outputs",               # *ForQuestionAnswering head
]

def get_peft_regex(
    model,
    finetune_vision_layers     : bool = True,
    finetune_language_layers   : bool = True,
    finetune_attention_modules : bool = True,
    finetune_mlp_modules       : bool = True,
    finetune_audio_layers      : bool = False,
    target_modules             : List[str] = None,
    vision_tags                : List[str] = ["vision", "image", "visual", "patch",],
    language_tags              : List[str] = ["language", "text",],
    attention_tags             : List[str] = ["self_attn", "attention", "attn", "mixer",],
    mlp_tags                   : List[str] = ["mlp", "feed_forward", "ffn", "dense", "mixer",],
) -> str:
    """
    Create a regex pattern to apply LoRA to only select layers of a model.
    """
    # All Unsloth Zoo code licensed under LGPLv3
    if not finetune_vision_layers and not finetune_language_layers and not finetune_audio_layers:
        raise RuntimeError(
            "Unsloth: No layers to finetune - please select to finetune the vision, language and/or audio layers!"
        )
    if not finetune_attention_modules and not finetune_mlp_modules:
        raise RuntimeError(
            "Unsloth: No modules to finetune - please select to finetune the attention and/or the mlp modules!"
        )
    pass

    from collections import Counter
    modules = model.named_modules()
    linear_modules = [name for name, module in modules if isinstance(module, torch.nn.Linear)]

    # Gemma4 ClippableLinear wraps nn.Linear as .linear child -- detect and add those
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        for name, module in model.named_modules():
            if isinstance(module, Gemma4ClippableLinear):
                linear_modules.append(name + ".linear")
    except ImportError:
        pass

    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)

    # Isolate lm_head / projection matrices (count == 1)
    if target_modules is None:
        only_linear_modules = []
        projection_modules  = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if count != 1:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
        pass
    else:
        assert(type(target_modules) is list)
        only_linear_modules = list(target_modules)
    pass

    regex_model_parts = []
    if finetune_vision_layers:     regex_model_parts += vision_tags
    if finetune_language_layers:   regex_model_parts += language_tags
    regex_components  = []
    if finetune_attention_modules: regex_components  += attention_tags
    if finetune_mlp_modules:       regex_components  += mlp_tags

    regex_model_parts = "|".join(regex_model_parts)
    regex_components  = "|".join(regex_components)

    match_linear_modules = r"(?:" + "|".join(re.escape(x) for x in only_linear_modules) + r")"
    # No trailing ".*?" after the linear-module group: PEFT uses re.fullmatch, so ".*?" would let
    # "...attn.proj_drop" (a Dropout) match ("proj" + ".*?" eating "_drop") -> "Target module
    # Dropout is not supported". LoRA targets are leaf Linears whose names ARE the group entries,
    # so ending at the group keeps every real target and drops same-prefix non-linear modules.
    if regex_model_parts == "":
        # No vision/language model-part selected (e.g. audio-only finetuning):
        # the standard matcher would degenerate into matching every attention/mlp
        # leaf in the model, so make the base inert and rely solely on the
        # dedicated audio branches added below.
        regex_matcher = r"(?!x)x"  # never matches
    else:
        regex_matcher = \
            r".*?(?:"  + regex_model_parts + \
            r").*?(?:" + regex_components + \
            r").*?"    + match_linear_modules

        # Also account for model.layers.0.self_attn/mlp type modules like Qwen
        if finetune_language_layers:
            regex_matcher = r"(?:" + regex_matcher + \
            r")|(?:\bmodel\.layers\.[\d]{1,}\.(?:" + regex_components + \
            r")\.(?:" + match_linear_modules + r"))"
        pass

        # Check if regex is wrong since model does not have vision parts
        check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
        if not check:
            regex_matcher = \
                r".*?(?:" + regex_components + \
                r").*?"   + match_linear_modules
        pass
    pass

    # Gemma 4 / Gemma 3N keep the visual/audio path in flat embedder Linears
    # (embed_vision.embedding_projection, embed_audio.embedding_projection,
    # vision_embedder.patch_dense, vision_tower.patch_embedder.input_proj) and a
    # conformer audio_tower whose ffw_layer_* / lconv1d.linear_* / attention.post
    # leaves carry no attn/mlp component token. The standard matcher above can
    # never select them, so add dedicated name-anchored branches. Each branch is
    # only appended if it actually matches a Linear in THIS model, so for every
    # other architecture the regex is byte-identical (the branches are anchored to
    # audio_tower / embed_vision / embed_audio / vision_embedder name segments that
    # exist only on Gemma 4 / Gemma 3N). Leading ".*?" for PEFT's re.fullmatch; no
    # trailing ".*?" (a real Linear leaf must terminate the match); "(?:\.linear)?"
    # also covers Gemma4ClippableLinear ".linear" children.
    candidate_branches = []
    if finetune_audio_layers:
        # Exact Linear leaf names of the Gemma 4 / Gemma 3N audio conformer
        # (verified from the checkpoint tensor names). conv / *norm leaves are not
        # nn.Linear so they are never candidates and need no exclusion here.
        audio_leaf_names = [
            "q_proj", "k_proj", "v_proj", "relative_k_proj", "pos_proj",
            "post", "output_proj", "ffw_layer_1", "ffw_layer_2",
            "linear_start", "linear_end", "input_proj_linear",
        ]
        audio_leaf_re = r"(?:" + "|".join(re.escape(x) for x in audio_leaf_names) + r")"
        candidate_branches.append(r"(?:.*?\baudio_tower\..*?" + audio_leaf_re + r"(?:\.linear)?)")
        candidate_branches.append(r"(?:.*?\bembed_audio\.embedding_projection(?:\.linear)?)")
    if finetune_vision_layers:
        candidate_branches.append(
            r"(?:.*?\b(?:embed_vision\.embedding_projection"
            r"|vision_embedder\.patch_dense"
            r"|vision_tower\.patch_embedder\.input_proj)(?:\.linear)?)"
        )
    extra_branches = [
        branch for branch in candidate_branches
        if any(re.search(branch, name, flags = re.DOTALL) for name in linear_modules)
    ]
    if extra_branches:
        regex_matcher = r"(?:" + regex_matcher + r")|" + "|".join(extra_branches)
    pass

    # Final check to confirm if matches exist
    check = any(re.search(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
    if not check and target_modules is not None:
        raise RuntimeError(
            f"Unsloth: No layers to finetune? You most likely specified target_modules = {target_modules} incorrectly!"
        )
    elif not check:
        raise RuntimeError(
            f"Unsloth: No layers to finetune for {model.config._name_or_path}. Please file a bug report!"
        )
    pass
    return regex_matcher
pass


def get_lora_layer_modules():
    # All Unsloth Zoo code licensed under LGPLv3
    import peft.tuners.lora
    path = os.path.split(peft.tuners.lora.__file__)[0]
    files = os.listdir(path)

    Linear_LoRA_Layers = []
    for file in files:
        if file == "__init__.py" or not file.endswith(".py"): continue
        item = f"peft.tuners.lora.{file[:-len('.py')]}"
        exec(f"import {item}", locals(), globals())
        modules = dir(eval(item))
        modules = [x for x in modules if x.startswith("Linear") or x.endswith("Linear")]
        if len(modules) == 0: continue
        exec(f"from {item} import ({', '.join(modules)})", locals(), globals())
        Linear_LoRA_Layers += [(eval(x), item, x,) for x in modules]
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
            name     = getattr(value, "__name__", "")
            if name1 in qualname or name2 in qualname: pass
            elif name2 in name or name2 in name: pass
            else: other_hooks.append(value)
        pass
        # Keep none input requires grad hooks
        exec(f"module.{_hooks} = OrderedDict()")
        if _hooks == "_forward_pre_hooks":
            if hasattr(module, "_forward_pre_hooks_with_kwargs"):
                module._forward_pre_hooks_with_kwargs.clear()
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
            try: # For HF dataclass, try loss or logits
                if hasattr(output, "loss") and output.loss is not None:
                    output.loss.requires_grad_(True)
                elif hasattr(output, "logits") and output.logits is not None: # RL like GRPO has no loss (no labels)
                    output.logits.requires_grad_(True)
                elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    # Encoder / decoder-style embedding backbones (e.g. Qwen3-Embedding) return a
                    # BaseModelOutputWithPast with only last_hidden_state (no loss/logits) when called
                    # for sentence embeddings. Make it require grad so gradient checkpointing works.
                    # See https://github.com/unslothai/unsloth/issues/5360
                    output.last_hidden_state.requires_grad_(True)
                else:
                    raise ValueError("Neither loss, logits, nor last_hidden_state are available for grad post hook.")
            except Exception as e:
                raise RuntimeError(f"Unsloth: Failed to make output require gradients: {e}")
    pass

    def requires_grad_pre_hook(module, args, kwargs):
        # Try positional args first (normal text models)
        if args:
            first = args[0]
            if type(first) is torch.Tensor:
                if torch.is_floating_point(first):
                    first.requires_grad_(True)
                return
            pass
        pass
        # Kwargs-only path (VLMs like Idefics3, SmolVLM2, Llava, Qwen2VL, etc.):
        # inputs_embeds is universal; hidden_states covers vision encoders;
        # pixel_values covers image inputs.
        for key in ("inputs_embeds", "hidden_states", "pixel_values"):
            tensor = kwargs.get(key)
            if tensor is not None and type(tensor) is torch.Tensor:
                if torch.is_floating_point(tensor):
                    tensor.requires_grad_(True)
                return
            pass
        pass
        # Fallback: scan kwargs for any float tensor
        for key, val in kwargs.items():
            if type(val) is torch.Tensor and torch.is_floating_point(val):
                val.requires_grad_(True)
                return
            pass
        pass
    pass

    def collect_hook_targets():
        hook_targets = OrderedDict()
        fallback_targets = OrderedDict()

        for name, param in model.named_parameters():
            if not param.requires_grad: continue

            name = re.sub(r"\.([\d]{1,})\.", r"[\1].", name)
            name_components = name.split(".")

            if len(name_components) == 0:
                raise RuntimeError("Unsloth: Model has 0 layers?")

            final_where = None
            fallback_name = None
            fallback_module = None

            # Try getting previous parent module
            for j in range(len(name_components)-1, 0, -1):
                name_curr = name_components[j]
                name_pre  = "model." + ".".join(name_components[:j])
                # Disable [\d] since it fails in gradient checkpointing
                if re.search(r"\[[\d]{1,}\]", name_pre): continue
                module = eval(name_pre, globals(), {"model" : model})
                fallback_name   = name_pre
                fallback_module = module
                if hasattr(module, "forward"):
                    try: forward = inspect.getsource(module.forward)
                    except: continue

                    # Normal self.language_model(...)
                    if f"self.{name_curr}(" in forward:
                        final_where = j + 1
                        break

                    # Fix self.blocks[0] like in Qwen
                    module_list = re.sub(r"\[[\d]{1,}\]", "", name_curr)
                    if f"in self.{module_list}:" in forward:
                        final_where = j
                        break
                    elif re.search(r"for [^\s]{3,} in self\." + module_list, forward) is not None:
                        # Might have failed finding self.layers: like self.layers[...]:
                        final_where = j
                        break
                    pass
                pass
            pass

            if final_where is None:
                if fallback_module is not None:
                    fallback_targets[fallback_name] = fallback_module
                continue
            pass

            module_name = "model." + ".".join(name_components[:final_where])
            module = eval(module_name, globals(), {"model" : model})
            hook_targets[module_name] = module
        pass
        return hook_targets, fallback_targets
    pass

    hook_targets, fallback_targets = collect_hook_targets()
    if len(hook_targets) == 0 and len(fallback_targets) == 0: return

    hook_target_ids = {id(module) for module in hook_targets.values()}
    for fallback_name, fallback_target in fallback_targets.items():
        if id(fallback_target) in hook_target_ids: continue
        logger.info(
            f"Unsloth: Falling back to output gradient hook for `{fallback_name}` "
            f"during gradient checkpointing."
        )
        register_other_hooks(
            "requires_grad_post_hook",
            "requires_grad_post_hook",
            fallback_target,
            "_forward_hooks",
        )
        fallback_target.register_forward_hook(requires_grad_post_hook)
    pass
    if len(hook_targets) == 0: return

    for module_name, module in hook_targets.items():
        logger.info(f"Unsloth: Making `{module_name}` require gradients")

        still_need_patching = True
        # Check if input_embeddings exists
        if hasattr(module, "get_input_embeddings"):
            # Use forward hook after Embedding() is called
            try:
                module = module.get_input_embeddings()
                register_other_hooks(
                    "requires_grad_post_hook",
                    "requires_grad_post_hook",
                    module,
                    "_forward_hooks",
                )
                module.register_forward_hook(requires_grad_post_hook)
                logger.info(f"Unsloth: Registered output gradient hook on `{module_name}` input embeddings.")
                still_need_patching = False
            except Exception as exception:
                logger.warning(
                    f"Unsloth: Failed to register input-embedding hook for `{module_name}`: {exception}. "
                    "Falling back to pre-forward hook."
                )
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
            module.register_forward_pre_hook(requires_grad_pre_hook, with_kwargs=True)
            logger.info(f"Unsloth: Registered pre-forward gradient hook on `{module_name}`.")
        pass
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
