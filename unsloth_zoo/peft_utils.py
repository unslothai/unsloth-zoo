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

# merge_and_overwrite_lora and merge_and_dequantize_lora were listed here after they had
# already moved to saving_utils, which made `from unsloth_zoo.peft_utils import *` raise
# AttributeError rather than import anything.
__all__ = [
    "get_peft_regex",
    "SKIP_QUANTIZATION_MODULES",
    "MOE_ROUTER_MODULES",
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
from .empty_model import _get_module_attribute

# Leaf names of MoE routers, kept out of automatic LoRA targets; not gate, since D-FINE's gateway.gate is a plain
# projection (_moe_router_gate_names decides by position).
MOE_ROUTER_MODULES = frozenset((
    "router",
))


def _moe_router_gate_names(model):
    """`gate` Linears beside `experts` or inside a router module (AfMoE's mlp.router.gate)."""
    modules = dict(model.named_modules())
    names = []
    for name, module in modules.items():
        parent_name, _, leaf = name.rpartition(".")
        if leaf != "gate" or not isinstance(module, torch.nn.Linear):
            continue
        parent = modules.get(parent_name)
        if parent is None:
            continue
        if hasattr(parent, "experts") or "router" in type(parent).__name__.lower():
            names.append(name)
    return names


# PEFT's _check_lora_target_modules_mamba set, plus nemotron_h_puzzle (Nemotron 3.5 Bolt): its fused Mamba kernel reads
# out_proj.weight directly, bypassing a LoRA adapter.
MAMBA_MODEL_TYPES = frozenset(("falcon_h1", "mamba", "mamba2", "falcon_mamba", "nemotron_h", "nemotron_h_puzzle"))
MAMBA_ONLY_LEAVES = frozenset(("out_proj", "conv1d"))
# Mamba, Mamba2, Nemotron-H name the mixer `mixer`; Falcon-H1, Bamba, Zamba2 `mamba`.
MAMBA_MIXER_NAMES = ("mixer", "mamba")


def _mamba_scope(model):
    """Return "root" (model is a Mamba family), "nested" (in a sub-config) or None."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    if str(getattr(config, "model_type", "") or "").lower() in MAMBA_MODEL_TYPES:
        return "root"
    seen, stack = set(), [config]
    while stack:
        cfg = stack.pop()
        if id(cfg) in seen or cfg is None:
            continue
        seen.add(id(cfg))
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        if model_type in MAMBA_MODEL_TYPES:
            return "nested"
        for value in list(vars(cfg).values()) if hasattr(cfg, "__dict__") else []:
            if hasattr(value, "model_type") and not isinstance(value, (str, int, float, bool)):
                stack.append(value)
    return None


def _mamba_only_leaves(model):
    return MAMBA_ONLY_LEAVES if _mamba_scope(model) else frozenset()


def _mamba_subtree_exclusion(leaves):
    return (
        r"(?!.*\.(?:" + "|".join(MAMBA_MIXER_NAMES) + r")\.(?:"
        + "|".join(re.escape(x) for x in sorted(leaves)) + r")$)"
    )


# Skip some modules sensitive to quantization
SKIP_QUANTIZATION_MODULES = [
    "lm_head",
    "multi_modal_projector",    # Llama 3.2 Vision, Pixtral, Llava
    "merger",                   # Qwen2 VL
    "modality_projection",      # Idefics, SmolVLM
    "mm_projector",             # Kimi K2.5 / K2.7, LLaVA-style remote code
    "router",                   # MoE Router
    "mlp.gate",                 # MoE Router
    "block_sparse_moe.gate",    # MoE Router
    "moe.gate",                 # MoE Router (Step-3.7 reads its weight directly)
    'mamba',
    "audio_tower",              # Gemma3N audio encoder conformer
    "vision_tower",             # Gemma3 vision encoder (SigLIP)
    "visual",                   # Qwen2/2.5/3-VL, Qwen3.5, GLM-4V vision encoder
    "vision_embedder",          # multimodal embedders kept in full precision
    "embed_vision",
    "embed_audio",
    "score",                    # *ForSequenceClassification head
    "classifier",               # *ForTokenClassification, *ForImageClassification, BERT-family head
    "qa_outputs",               # *ForQuestionAnswering head
]

_LINEAR_FORWARD_RETURNS_TENSOR_CACHE = {}

def _linear_forward_returns_tensor(cls) -> bool:
    """False for an nn.Linear subclass whose forward returns a tuple, which LoRA cannot add to."""
    cached = _LINEAR_FORWARD_RETURNS_TENSOR_CACHE.get(cls)
    if cached is not None:
        return cached
    result = True
    forward = getattr(cls, "forward", None)
    if forward is not None and forward is not torch.nn.Linear.forward:
        try:
            import ast, textwrap
            tree = ast.parse(textwrap.dedent(inspect.getsource(forward)))
            for node in ast.walk(tree):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                    result = False
                    break
        except Exception:
            result = True
    _LINEAR_FORWARD_RETURNS_TENSOR_CACHE[cls] = result
    return result
pass


def get_peft_regex(
    model,
    finetune_vision_layers     : bool = True,
    finetune_language_layers   : bool = True,
    finetune_attention_modules : bool = True,
    finetune_mlp_modules       : bool = True,
    target_modules             : List[str] = None,
    vision_tags                : List[str] = ["vision", "image", "visual", "patch",],
    language_tags              : List[str] = ["language", "text",],
    attention_tags             : List[str] = ["self_attn", "attention", "attn", "mixer",],
    mlp_tags                   : List[str] = ["mlp", "feed_forward", "ffn", "dense", "mixer",],
    finetune_audio_layers      : bool = False,
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
    linear_modules = [
        name for name, module in modules
        if isinstance(module, torch.nn.Linear) and _linear_forward_returns_tensor(type(module))
    ]

    # Gemma4 ClippableLinear wraps nn.Linear as .linear child -- detect and add those
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        for name, module in model.named_modules():
            if isinstance(module, Gemma4ClippableLinear):
                linear_modules.append(name + ".linear")
    except ImportError:
        pass

    all_linear_modules = Counter(x.rsplit(".")[-1] for x in linear_modules)
    _in_layer_stack = frozenset(
        x.rsplit(".")[-1] for x in linear_modules if re.search(r"\.\d+\.", x)
    )

    # Isolate lm_head / projection matrices (count == 1)
    if target_modules is None:
        only_linear_modules = []
        projection_modules  = {}
        for j, (proj, count) in enumerate(all_linear_modules.items()):
            if proj in MOE_ROUTER_MODULES:
                # Llama 4 and PhiMoE routers return a tuple, which breaks PEFT's LoRA forward (it reads result.dtype), so these routers are skipped.
                continue
            # Seen once = a head (lm_head), unless inside a numbered layer stack (one layer of a kind).
            if count != 1 or proj in _in_layer_stack:
                only_linear_modules.append(proj)
            else:
                projection_modules[proj] = j
        pass
    else:
        assert(type(target_modules) is list)
        only_linear_modules = list(target_modules)
    pass

    # Mamba fused kernels read out_proj / conv1d weights directly (LoRA never runs) and PEFT >= 0.17 refuses them.
    mamba_nested_exclusion = None
    if target_modules is None:
        mamba_scope = _mamba_scope(model)
        mamba_leaves = MAMBA_ONLY_LEAVES if mamba_scope else frozenset()
        if mamba_leaves:
            dropped = [x for x in only_linear_modules if x in mamba_leaves]
            if mamba_scope == "root":
                only_linear_modules = [x for x in only_linear_modules if x not in mamba_leaves]
            elif dropped:
                # Only the Mamba subtree: a vision/audio tower's own out_proj stays a target.
                mamba_nested_exclusion = _mamba_subtree_exclusion(mamba_leaves)
            if dropped:
                logger.info(
                    f"Unsloth: leaving {', '.join(dropped)} out of the LoRA targets"
                    + (" of the Mamba mixers" if mamba_scope == "nested" else "")
                    + ": a Mamba mixer feeds them to its fused kernels, so an adapter on them would not train."
                )
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
    # No trailing .*? after the linear group: PEFT re.fullmatch would let attn.proj_drop (a Dropout) match; ending at
    # the group keeps real Linears and drops same-prefix non-Linears; the dot before it anchors leaf names (bare proj
    # matched Nemotron-H's Identity fc1_latent_proj, which PEFT rejects).
    if regex_model_parts == "":
        # With no vision/language model part selected (e.g. audio-only finetuning), the standard matcher would match
        # every attention/mlp leaf, so the base is made inert and only the dedicated audio branches below apply.
        regex_matcher = r"(?!x)x"  # never matches
    else:
        regex_matcher = \
            r".*?(?:"  + regex_model_parts + \
            r").*?(?:" + regex_components + \
            r").*?\."  + match_linear_modules

        # Also account for model.layers.0.self_attn/mlp type modules like Qwen
        if finetune_language_layers:
            # Like the composite branch, so text-only loads reach nested leaves (mlp.shared_expert.up_proj) and
            # linear_attn.* components; routed experts (mlp.experts.<N>.up_proj) stay out, since unsloth widens remote
            # expert blocks itself and keeps native Qwen3-MoE experts (transformers 4.x) off LoRA.
            regex_matcher = r"(?:" + regex_matcher + \
            r")|(?:\bmodel\.layers\.[\d]{1,}(?!.*\.experts\.\d)\..*?(?:" + regex_components + \
            r").*?\.(?:" + match_linear_modules + r"))"
        pass
    pass

    # Gemma 4 / Gemma 3N keep the visual/audio path in flat embedder Linears
    # (embed_vision.embedding_projection, embed_audio.embedding_projection,
    # vision_embedder.patch_dense, vision_tower.patch_embedder.input_proj) and a
    # conformer audio_tower whose ffw_layer_* / lconv1d.linear_* / attention.post
    # leaves carry no attn/mlp component token, so the standard matcher above can
    # never select them. Add dedicated name-anchored branches, gated on the
    # vision/audio flags, scoped to the attention/mlp flags, and -- when an explicit
    # target_modules list is given -- intersected with it. Each branch is appended
    # only if it re.fullmatch-es a Linear in THIS model (matching PEFT's own matcher).
    # Some of these leaf names are NOT Gemma-exclusive: empty_model.py also lists
    # embed_vision.embedding_projection / vision_tower.patch_embedder.input_proj as
    # Mistral3 components, so a name-only match could change a non-Gemma target set.
    # Gate the whole block on the Gemma 4 / Gemma 3N model families (via model_type /
    # architectures) so every other architecture's regex stays byte-identical.
    # Leading ".*?"; no trailing ".*?" (a real Linear leaf terminates the match);
    # "(?:\.linear)?" also covers Gemma4ClippableLinear ".linear" children.
    def _scoped(leaves):
        if target_modules is not None:
            return [x for x in leaves if x in only_linear_modules]
        return leaves

    _config        = getattr(model, "config", None)
    _model_type    = str(getattr(_config, "model_type", "") or "").lower()
    _architectures = " ".join(getattr(_config, "architectures", None) or []).lower()
    _is_gemma_mm   = (
        "gemma4"  in _model_type or "gemma4"  in _architectures or
        "gemma3n" in _model_type or "gemma3n" in _architectures or
        # EmbeddingGemma 2 carries Gemma 4 vision / audio towers under its own model_type.
        "embedding_gemma2" in _model_type or "embeddinggemma2" in _architectures
    )

    def _linear_aware_branches(cores):
        # Branch only on the real nn.Linear per core (bare module or .linear child of a Gemma4ClippableLinear wrapper);
        # a blanket optional .linear would also match the wrapper and double-process the inner Linear; checking each
        # core against linear_modules keeps Gemma 3N and Gemma 4 correct.
        out = []
        for core in cores:
            if any(re.fullmatch(r".*?" + core + r"\.linear", name, flags = re.DOTALL) for name in linear_modules):
                out.append(r"(?:.*?" + core + r"\.linear)")
            if any(re.fullmatch(r".*?" + core, name, flags = re.DOTALL) for name in linear_modules):
                out.append(r"(?:.*?" + core + r")")
        return out

    candidate_branches = []
    if finetune_audio_layers and _is_gemma_mm:
        # conv/norm leaves are not nn.Linear so never candidates; conformer attention leaves follow
        # finetune_attention_modules; feed-forward, lightweight-conv, subsample and output projections follow
        # finetune_mlp_modules; a dot before each leaf keeps k_proj from matching relative_k_proj.
        audio_leaves = []
        if finetune_attention_modules:
            audio_leaves += ["q_proj", "k_proj", "v_proj", "relative_k_proj", "pos_proj", "post"]
        if finetune_mlp_modules:
            audio_leaves += ["ffw_layer_1", "ffw_layer_2", "linear_start", "linear_end",
                             "input_proj_linear", "output_proj"]
        audio_leaves = _scoped(audio_leaves)
        audio_cores = [r"\baudio_tower\.(?:.*\.)?" + re.escape(x) for x in audio_leaves]
        # The audio projector is a feed-forward projection -> gate under the mlp flag.
        if finetune_mlp_modules and _scoped(["embedding_projection"]):
            audio_cores.append(r"\bembed_audio\.embedding_projection")
        candidate_branches += _linear_aware_branches(audio_cores)
    # Voxtral (Whisper-style) / Voxtral Realtime (Mistral-style) audio_tower + projector carry no language tag.
    _is_voxtral = _model_type.startswith("voxtral") or "voxtral" in _architectures
    if finetune_audio_layers and _is_voxtral:
        audio_leaves = []
        if finetune_attention_modules:
            audio_leaves += ["q_proj", "k_proj", "v_proj", "out_proj", "o_proj"]
        if finetune_mlp_modules:
            audio_leaves += ["fc1", "fc2", "gate_proj", "up_proj", "down_proj"]
        audio_leaves = _scoped(audio_leaves)
        audio_cores = [r"\baudio_tower\.(?:.*\.)?" + re.escape(x) for x in audio_leaves]
        if finetune_mlp_modules:
            audio_cores += [r"\bmulti_modal_projector\." + re.escape(x)
                            for x in _scoped(["linear_1", "linear_2"])]
        candidate_branches += _linear_aware_branches(audio_cores)
    if finetune_vision_layers and finetune_mlp_modules and _is_gemma_mm:
        # Gemma vision embedders are flat projection and dense Linears, so gate them under the mlp flag like the audio projector; attention-only stays clean.
        vision_cores = []
        if _scoped(["embedding_projection"]): vision_cores.append(r"\bembed_vision\.embedding_projection")
        if _scoped(["patch_dense"]):          vision_cores.append(r"\bvision_embedder\.patch_dense")
        if _scoped(["input_proj"]):           vision_cores.append(r"\bvision_tower\.patch_embedder\.input_proj")
        candidate_branches += _linear_aware_branches(vision_cores)
    # _linear_aware_branches already emits only real Linear matches; this re-filter is for safety and mirrors PEFT's own re.fullmatch matcher exactly.
    extra_branches = [
        branch for branch in candidate_branches
        if any(re.fullmatch(branch, name, flags = re.DOTALL) for name in linear_modules)
    ]
    if extra_branches:
        regex_matcher = r"(?:" + regex_matcher + r")|" + "|".join(extra_branches)
    pass

    # Fallback, only for a requested vision/language part with no match (e.g. untagged VLM vision modules), drops the
    # model-part requirement; never for audio-only, which would pull in the whole language stack; fullmatch since
    # re.search would match CSM's depth_decoder.model.layers., which PEFT rejects.
    if regex_model_parts != "":
        check = any(re.fullmatch(regex_matcher, name, flags = re.DOTALL) for name in linear_modules)
        if not check:
            # Frozen audio codecs (CSM / Kyutai Mimi) run under no_grad, so LoRA there never trains.
            regex_matcher = \
                r"(?!(?:.*\.)?(?:codec_model|audio_tokenizer|acoustic_tokenizer|semantic_tokenizer)\.)" + \
                r".*?(?:" + regex_components + \
                r").*?\." + match_linear_modules
    pass

    # Exclude parameter-free leaves by exact name only (Nemotron-H's Identity fc1_latent_proj is a Linear elsewhere).
    placeholders = [
        name for name, module in model.named_modules()
        if name
        and next(module.parameters(recurse = False), None) is None
        and next(module.children(), None) is None
        and re.fullmatch(regex_matcher, name, flags = re.DOTALL)
    ]
    if placeholders:
        logger.info(
            f"Unsloth: leaving {len(placeholders)} parameter-free placeholder module(s) out of "
            f"the LoRA targets, e.g. {placeholders[0]}"
        )
    # A trained router can overflow (unsloth#3690); only an explicit target_modules trains it.
    router_gates = [] if target_modules is not None else [
        name for name in _moe_router_gate_names(model)
        if re.fullmatch(regex_matcher, name, flags = re.DOTALL)
    ]
    if router_gates:
        logger.info(
            f"Unsloth: leaving {len(router_gates)} MoE router gate(s) out of the LoRA targets, "
            f"e.g. {router_gates[0]}. Add \"gate\" to target_modules to train them."
        )
    excluded = placeholders + router_gates
    if excluded:
        regex_matcher = (
            r"(?!(?:" + "|".join(re.escape(x) for x in excluded) + r")$)(?:" + regex_matcher + r")"
        )
    pass

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
    if mamba_nested_exclusion is not None:
        regex_matcher = mamba_nested_exclusion + r"(?:" + regex_matcher + r")"
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
        # Skip submodules whose optional backend (bnb/eetq/awq/...) is missing instead of crashing LoRA-layer discovery.
        try:
            exec(f"import {item}", locals(), globals())
        except (ImportError, OSError, ValueError, AttributeError, TypeError) as exception:
            logger.debug(
                f"Unsloth: Skipping LoRA layers from `{item}` "
                f"(optional backend unavailable): {exception}"
            )
            continue
        modules = dir(eval(item))
        modules = [x for x in modules if x.startswith("Linear") or x.endswith("Linear")]
        if len(modules) == 0: continue
        exec(f"from {item} import ({', '.join(modules)})", locals(), globals())
        Linear_LoRA_Layers += [(eval(x), item, x,) for x in modules]
    pass
    return tuple(Linear_LoRA_Layers)
pass


def _outside_compiled_region(root, embeddings):
    """The input embedding's outermost ancestor below `root` whose class Unsloth's compiler swapped in
    (AST's ASTEmbeddings): a hook there runs outside torch.compile, where requires_grad_() is allowed."""
    path = next((name for name, sub in root.named_modules() if sub is embeddings), None)
    if not path:
        return embeddings
    current = root
    for part in path.split(".")[:-1]:
        current = getattr(current, part)
        if type(current).__module__.startswith("unsloth_compiled_module"):
            return current
    return embeddings
pass


def _frozen_frontend(module):
    """The frozen child whose output starts a tower's checkpointed layer loop, else None.

    Matches `h = self.<child>(<first input>, ...)` ahead of `for layer in self.<ModuleList>:` with
    `layer(h, ...)`, every later assignment to `h` reading `h`. Flipping that output, not the raw
    audio input, keeps autograd out of the frozen front end (Parakeet / Canary conv subsampling): the
    same adapter gradients without its activations and input gradients."""
    import ast, textwrap
    forward = getattr(type(module), "forward", None)
    if forward is None or hasattr(forward, "_torchdynamo_orig_callable"):
        return None
    if type(module).__module__.startswith("unsloth_compiled_module"):
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(forward)))
    except Exception:
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if fn is None or len(fn.args.args) < 2:
        return None
    first_input = fn.args.args[1].arg
    # Audio front ends only: a vision patch embedding is one cheap conv, not worth widening the change.
    if first_input not in ("input_features", "input_values", "audio_mel", "audio_values"):
        return None

    def self_attr(node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return node.attr
        return None

    loop_index = None
    for i, stmt in enumerate(fn.body):
        if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            it = stmt.iter
            name = self_attr(it)
            if isinstance(getattr(module, name or "", None), torch.nn.ModuleList):
                loop_index = i
                break
    if loop_index is None:
        return None
    loop = fn.body[loop_index]
    hidden = child = None
    for stmt in fn.body[:loop_index]:
        if (child is None and isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Call)
                and self_attr(stmt.value.func) and stmt.value.args
                and isinstance(stmt.value.args[0], ast.Name) and stmt.value.args[0].id == first_input):
            hidden, child = stmt.targets[0].id, self_attr(stmt.value.func)
            continue
        for node in ast.walk(stmt):
            targets = node.targets if isinstance(node, ast.Assign) else \
                [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
            for target in targets:
                stores = {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
                if hidden is None or hidden not in stores:
                    continue
                value = getattr(node, "value", None)
                metadata = {
                    id(n.value) for n in ast.walk(value) if isinstance(n, ast.Attribute)
                    and n.attr in ("shape", "size", "dtype", "device", "ndim")
                } if value is not None else set()
                reads = value is not None and any(
                    isinstance(n, ast.Name) and n.id == hidden and id(n) not in metadata for n in ast.walk(value)
                )
                detaches = value is not None and any(
                    isinstance(n, ast.Attribute) and n.attr in ("detach", "data", "detach_") for n in ast.walk(value)
                )
                if not isinstance(target, ast.Name) or not (reads or isinstance(node, ast.AugAssign)) or detaches:
                    return None
    if child is None:
        return None
    layer = loop.target.id
    feeds = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == layer
        and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == hidden
        for n in ast.walk(loop)
    )
    frontend = getattr(module, child, None)
    if not feeds or not isinstance(frontend, torch.nn.Module):
        return None
    params = list(frontend.parameters())
    if not params or any(p.requires_grad for p in params):
        return None
    return frontend
pass


def _freeze_unused_feature_encoders(model):
    """wav2vec2-style feature encoders (`_requires_grad`, `_freeze_parameters`) flag their input as
    requiring grad unless frozen, so a frozen conv stack still runs its whole backward. Mark the ones
    LoRA left frozen as frozen, as `freeze_feature_encoder()` does."""
    for module in model.modules():
        if getattr(module, "_requires_grad", None) is not True or not callable(getattr(module, "_freeze_parameters", None)):
            continue
        params = list(module.parameters())
        if params and not any(p.requires_grad for p in params):
            module._requires_grad = False
pass


def _subsampling_input_to_16bit(module, args, kwargs):
    """Pre-hook for the FastConformer subsampling (Parakeet, Canary): float32 features to 16 bit.

    Unsloth's conv dtype wrapper casts a conv's input to its weight dtype and the result back to the
    input's dtype, so float32 input_features kept the frozen conv stack in float32: twice autocast's
    activation memory. Casting up front is bit identical: every conv already ran in the 16-bit weight
    dtype, the ReLUs and 0/1 masks between them are exact, and autocast feeds the closing 16-bit Linear
    the same values. A quantized closing Linear (QLoRA) computes in its input's dtype, so it is skipped.
    A hook, not a forward patch, so the compiler's view of the forward is unchanged."""
    layers = getattr(module, "layers", None)
    weight = getattr(layers[0], "weight", None) if layers is not None and len(layers) else None
    linear_weight = getattr(getattr(module, "linear", None), "weight", None)
    if not (
        isinstance(weight, torch.Tensor) and weight.dtype in (torch.float16, torch.bfloat16)
        and isinstance(linear_weight, torch.Tensor) and linear_weight.dtype in (torch.float16, torch.bfloat16)
        and not hasattr(linear_weight, "quant_state")
    ):
        return None
    if args and type(args[0]) is torch.Tensor and args[0].dtype == torch.float32:
        return (args[0].to(weight.dtype),) + tuple(args[1:]), kwargs
    features = kwargs.get("input_features")
    if type(features) is torch.Tensor and features.dtype == torch.float32:
        return args, {**kwargs, "input_features": features.to(weight.dtype)}
    return None
pass


def _cast_subsampling_inputs(model):
    for module in model.modules():
        if type(module).__name__ != "ParakeetEncoderSubsamplingConv2D":
            continue
        for key, hook in list(module._forward_pre_hooks.items()):
            if getattr(hook, "__name__", "") == "_subsampling_input_to_16bit":
                del module._forward_pre_hooks[key]
                getattr(module, "_forward_pre_hooks_with_kwargs", {}).pop(key, None)
        module.register_forward_pre_hook(_subsampling_input_to_16bit, with_kwargs = True)
pass


def requires_grad_for_gradient_checkpointing(model):
    # All Unsloth Zoo code licensed under LGPLv3
    # Enables requires_grad to make gradient checkpointing work on
    # non language models that don't just use .embed_tokens
    def register_other_hooks(name1, name2, module, _hooks):
        old_hooks = eval(f"module.{_hooks}")
        # Hooks with with_kwargs = True (Unsloth's inputs_embeds pre-hook) take (module, args, kwargs); re-register them
        # the same way, or the next forward calls them with two arguments.
        with_kwargs = set(getattr(module, f"{_hooks}_with_kwargs", None) or ())
        other_hooks = []
        for key, value in old_hooks.items():
            qualname = getattr(value, "__qualname__", "")
            name     = getattr(value, "__name__", "")
            if name1 in qualname or name2 in qualname: pass
            elif name2 in name or name2 in name: pass
            else: other_hooks.append((value, key in with_kwargs))
        pass
        # Keep none input requires grad hooks
        exec(f"module.{_hooks} = OrderedDict()")
        if hasattr(module, f"{_hooks}_with_kwargs"):
            getattr(module, f"{_hooks}_with_kwargs").clear()
        register = getattr(module, f"register{_hooks[:-1]}")
        for hook, kwargs_hook in other_hooks:
            if kwargs_hook:
                register(hook, with_kwargs = True)
            else:
                register(hook)
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

    def requires_grad_post_hook(module, input, output):
        type_output = type(output)
        if type_output is torch.Tensor:
            target = output
        else:
            try: # For HF dataclass, try loss or logits
                if hasattr(output, "loss") and output.loss is not None:
                    target = output.loss
                elif hasattr(output, "logits") and output.logits is not None: # RL like GRPO has no loss (no labels)
                    target = output.logits
                elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    # Encoder / decoder-style embedding backbones (e.g. Qwen3-Embedding) return a
                    # BaseModelOutputWithPast with only last_hidden_state (no loss/logits) when called
                    # for sentence embeddings. Make it require grad so gradient checkpointing works.
                    # See https://github.com/unslothai/unsloth/issues/5360
                    target = output.last_hidden_state
                else:
                    # Raise while tracing too: is_compiling() is constant folded, so a skip
                    # here is permanent and the region trains with no adapter gradients.
                    raise ValueError("Neither loss, logits, nor last_hidden_state are available for grad post hook.")
            except Exception as e:
                raise RuntimeError(f"Unsloth: Failed to make output require gradients: {e}")
        # Dynamo rejects requires_grad_() only when it would flip the flag, so skipping the
        # no-op keeps fullgraph = True working. Skipping a real flip would break the frozen
        # input embedding this also lands on, losing every checkpointing gradient. Without
        # grad (a compiled decode step) nothing is recorded, so the flip is moot there.
        if torch.compiler.is_compiling() and (target.requires_grad or not torch.is_grad_enabled()): return
        target.requires_grad_(True)
    pass

    def requires_grad_post_hook_eager(module, input, output):
        # Skipped while compiling: the requires_grad flip is a needless graph break in Gemma 3N's compiled multimodal
        # embedder (its LoRA projection follows); Dia's encoder runs eager.
        if torch.compiler.is_compiling(): return
        requires_grad_post_hook(module, input, output)
    pass

    def requires_grad_pre_hook(module, args, kwargs):
        # Dynamo cannot trace requires_grad_(), and Gemma 3N compiles a LoRA target
        # (embed_audio.embedding_projection) with fullgraph = True, so it is a hard error.
        # Safe to skip: anything traced here is already downstream of trainable LoRA weights.
        if torch.compiler.is_compiling(): return
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

    def requires_grad_frontend_hook(module, input, output):
        if torch.compiler.is_compiling(): return
        target = output
        if type(target) in (tuple, list):
            target = next((x for x in target if type(x) is torch.Tensor), None)
        if type(target) is torch.Tensor and torch.is_floating_point(target) and torch.is_grad_enabled():
            target.requires_grad_(True)
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
                module = _get_module_attribute(model, ".".join(name_components[:j]))
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
                    elif re.search(r"for [^\n:]+ in enumerate\(self\." + module_list + r"\b", forward) is not None:
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
            module = _get_module_attribute(model, ".".join(name_components[:final_where]))
            hook_targets[module_name] = module
        pass
        return hook_targets, fallback_targets
    pass

    _freeze_unused_feature_encoders(model)
    _cast_subsampling_inputs(model)
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
        if hasattr(module, "get_input_embeddings"):
            # Use forward hook after Embedding() is called
            embeddings = None
            try:
                embeddings = module.get_input_embeddings()
            except NotImplementedError:
                logger.info(
                    f"Unsloth: `{module_name}` has no input embeddings; using a pre-forward hook."
                )
            except Exception as exception:
                logger.warning(
                    f"Unsloth: Failed to register input-embedding hook for `{module_name}`: {exception}. "
                    "Falling back to pre-forward hook."
                )
            else:
                # Keep `module` on the tower: the fallback below hooks it.
                if embeddings is not None:
                    embeddings = _outside_compiled_region(module, embeddings)
                try:
                    register_other_hooks(
                        "requires_grad_post_hook",
                        "requires_grad_post_hook",
                        embeddings,
                        "_forward_hooks",
                    )
                    embeddings.register_forward_hook(requires_grad_post_hook)
                    logger.info(f"Unsloth: Registered output gradient hook on `{module_name}` input embeddings.")
                    still_need_patching = False
                except Exception as exception:
                    logger.warning(
                        f"Unsloth: Failed to register input-embedding hook for `{module_name}`: {exception}. "
                        "Falling back to pre-forward hook."
                    )
        pass

        frontend = _frozen_frontend(module) if still_need_patching else None
        if frontend is not None:
            register_other_hooks(
                "requires_grad_frontend_hook",
                "requires_grad_frontend_hook",
                frontend,
                "_forward_hooks",
            )
            frontend.register_forward_hook(requires_grad_frontend_hook)
            logger.info(f"Unsloth: Registered output gradient hook on `{module_name}`'s frozen front end.")
            still_need_patching = False
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
            # Tower fed only token ids with no get_input_embeddings (Dia's encoder): the pre-hook gets no float input,
            # so its checkpointed layers would get no gradients; hook its own Embedding.
            embedding = next((c for c in module.children() if isinstance(c, torch.nn.Embedding)), None)
            if embedding is not None:
                register_other_hooks(
                    "requires_grad_post_hook",
                    "requires_grad_post_hook",
                    embedding,
                    "_forward_hooks",
                )
                embedding.register_forward_hook(requires_grad_post_hook_eager)
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
