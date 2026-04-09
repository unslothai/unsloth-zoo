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

"""
Lightweight FastLanguageModel for Apple Silicon / MLX.

No GPU dependencies — uses mlx-lm for model loading and LoRA.
Supports both text-only models (mlx-lm) and VLMs (mlx-vlm).
This avoids importing unsloth.models (which pulls in CUDA kernels).
"""

import json
import types
import warnings


_vlm_model_types_cache = None


def _is_vlm(config: dict) -> bool:
    """Detect whether a model config describes a VLM.

    Checks:
    1. "vision_config" key in config → True
    2. model_type is in mlx_vlm's supported model set (discovered dynamically)
    """
    if "vision_config" in config:
        return True

    model_type = config.get("model_type", "")
    if not model_type:
        return False

    global _vlm_model_types_cache
    if _vlm_model_types_cache is None:
        _vlm_model_types_cache = _build_vlm_model_types()

    return model_type in _vlm_model_types_cache


def _build_vlm_model_types():
    """Build the set of model_type strings that mlx_vlm supports.

    Uses dynamic discovery via pkgutil + MODEL_REMAPPING keys/values.
    Returns frozenset; cached at module level by _is_vlm().
    """
    types_set = set()
    try:
        import mlx_vlm.models as vlm_models_pkg
        import pkgutil
        for importer, modname, ispkg in pkgutil.iter_modules(vlm_models_pkg.__path__):
            if ispkg:
                types_set.add(modname)
    except (ImportError, Exception):
        pass

    try:
        from mlx_vlm.utils import MODEL_REMAPPING
        # Add both source and target keys
        for src, tgt in MODEL_REMAPPING.items():
            types_set.add(src)
            types_set.add(tgt)
    except (ImportError, Exception):
        pass

    return frozenset(types_set)


def _fix_missing_no_grad(model):
    """Ensure every nn.Module submodule has _no_grad / _training.

    Works around upstream model definitions that use __new__ without __init__
    (e.g. gemma4 AudioRelativePositionEmbedding).
    """
    import mlx.nn as nn
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Module):
            if not hasattr(mod, "_no_grad"):
                object.__setattr__(mod, "_no_grad", set())
            if not hasattr(mod, "_training"):
                object.__setattr__(mod, "_training", True)


class _TrainingKVStore:
    """Lightweight KV store for Gemma4 KV-sharing during training.

    Gemma4 E2B/E4B have shared attention layers that borrow K/V from earlier
    "store" layers via the KV cache. During training (cache=None), the shared
    layers silently fall through to computing their own K/V from wrong hidden
    states. This class provides a minimal interface so store layers can write
    K/V and shared layers can read them, without autoregressive offset tracking.

    Implements the subset of KVCache that Attention.__call__ needs:
    - offset (always 0 for training — no prior tokens)
    - state property (returns stored K/V)
    - update_and_fetch (stores K/V from store layer)
    """
    __slots__ = ("keys", "values")

    def __init__(self):
        self.keys = None
        self.values = None

    @property
    def offset(self):
        return 0

    @property
    def state(self):
        return (self.keys, self.values)

    def update_and_fetch(self, keys, values):
        self.keys = keys
        self.values = values
        return keys, values


def _fix_gemma4_kv_sharing(model):
    """Fix Gemma4 KV-shared layers producing wrong K/V during training.

    Gemma4 E2B/E4B have num_kv_shared_layers shared attention layers that
    borrow K/V from earlier "store" layers via the KV cache. When cache=None
    (training), shared layers fall through to computing their own K/V from
    the wrong hidden state — silently producing incorrect attention.

    Fix: monkey-patch the text backbone's __call__ to create _TrainingKVStore
    objects when cache=None, so store layers populate them and shared layers
    read correct K/V.
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    backbone = getattr(lm, "model", None)
    if backbone is None:
        return

    first_shared = getattr(backbone, "first_kv_shared_layer_idx", None)
    num_layers = getattr(backbone, "num_hidden_layers", None)
    if first_shared is None or num_layers is None or first_shared >= num_layers:
        return  # No shared layers

    cls = backbone.__class__
    if getattr(cls, "_kv_sharing_patched", False):
        return  # Already patched

    original_call = cls.__call__
    n_stores = first_shared  # number of store-layer cache slots

    def patched_call(self, inputs=None, inputs_embeds=None, mask=None,
                     cache=None, per_layer_inputs=None, **kwargs):
        if cache is None:
            # Objects created once under mx.compile tracing; data flow through
            # update_and_fetch/state is captured in the computation graph.
            cache = [_TrainingKVStore() for _ in range(n_stores)]
        return original_call(
            self, inputs=inputs, inputs_embeds=inputs_embeds, mask=mask,
            cache=cache, per_layer_inputs=per_layer_inputs, **kwargs,
        )

    cls.__call__ = patched_call
    cls._kv_sharing_patched = True
    n_shared = num_layers - first_shared
    print(f"Unsloth: Fixed Gemma4 KV-sharing for training "
          f"({n_shared} shared layers now read correct K/V).")


# ---------------------------------------------------------------------------
# Runtime VLM quantization via monkey-patching mlx_vlm.utils.load_model
# ---------------------------------------------------------------------------
_vlm_load_model_patched = False
_original_vlm_load_model = None

# Fragments that identify multimodal sub-networks we must *never* quantize.
_MULTIMODAL_SKIP_FRAGMENTS = (
    "lm_head", "embed_tokens",
    "multi_modal_projector", "mm_projector", "connector", "aligner",
    "vision_encoder", "audio_encoder", "audio_projection",
)


def _vlm_config_is_already_quantized(config_data: dict) -> bool:
    """Return True when the HF config indicates the model is pre-quantized."""
    if "quantization" in config_data:
        return True
    qc = config_data.get("quantization_config", {})
    if qc.get("quant_method"):
        return True
    return False


def _build_vlm_quant_predicate(model):
    """Build a quant_predicate for mlx_lm.utils.quantize_model.

    Two layers of filtering:
    1. Hard-skip multimodal modules (vision tower, projectors, embeddings, …)
    2. Delegate to model.quant_predicate for model-specific rules (MoE gates, …)
    """
    try:
        from mlx_vlm.utils import skip_multimodal_module
    except ImportError:
        skip_multimodal_module = None

    # Trigger model-specific setup (e.g. phi4mm LoRA merge side-effect)
    model_predicate = getattr(model, "quant_predicate", None)

    def predicate(path: str, module):
        # 1. Hard skip — mlx_vlm's own multimodal check
        if skip_multimodal_module is not None and skip_multimodal_module(path):
            return False
        # 2. Hard skip — extra fragments (embeddings, projectors, …)
        path_parts = path.split(".")
        for frag in _MULTIMODAL_SKIP_FRAGMENTS:
            if frag in path_parts:
                return False
        # 3. Model-specific predicate (MoE gates → 8-bit, phi4mm exclusions, …)
        if model_predicate is not None:
            return model_predicate(path, module)
        return True

    return predicate


def _patched_vlm_load_model(model_path, lazy=False, **kwargs):
    """Drop-in replacement for mlx_vlm.utils.load_model with runtime quantization."""
    import mlx.core as mx

    q_bits = kwargs.pop("q_bits", None)
    q_group_size = kwargs.pop("q_group_size", 64)

    # Load float weights (always lazy so quantization runs on lazy arrays)
    model = _original_vlm_load_model(model_path, lazy=True, **kwargs)

    if q_bits is not None:
        from mlx_lm.utils import quantize_model
        from mlx_vlm.utils import load_config

        config = load_config(model_path)
        predicate = _build_vlm_quant_predicate(model)
        model, updated_config = quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=predicate,
        )
        model._config = updated_config

    if not lazy:
        mx.eval(model.parameters())

    return model


def _ensure_vlm_load_model_patched():
    """Idempotent installer — patches mlx_vlm.utils.load_model on first call."""
    global _vlm_load_model_patched, _original_vlm_load_model

    if _vlm_load_model_patched:
        return

    import mlx_vlm.utils as _vlm_utils

    _original_vlm_load_model = _vlm_utils.load_model
    _vlm_utils.load_model = _patched_vlm_load_model
    _vlm_load_model_patched = True


def _mlx_save_pretrained_merged(self, save_directory, tokenizer=None, **kwargs):
    from .mlx_utils import save_merged_model
    tokenizer = tokenizer or self._tokenizer
    save_merged_model(self, tokenizer, save_directory)


def _mlx_save_pretrained_gguf(self, save_directory, tokenizer=None,
                               quantization_method="fast_quantized", **kwargs):
    from .mlx_utils import save_pretrained_gguf
    tokenizer = tokenizer or self._tokenizer
    save_pretrained_gguf(self, tokenizer, save_directory,
                         quantization_method=quantization_method)


def _mlx_push_to_hub_merged(self, repo_id, tokenizer=None, **kwargs):
    from .mlx_utils import push_to_hub_merged
    tokenizer = tokenizer or self._tokenizer
    push_to_hub_merged(self, tokenizer, repo_id, repo_id=repo_id, **kwargs)


def _mlx_push_to_hub_gguf(self, repo_id, tokenizer=None,
                            quantization_method="fast_quantized", **kwargs):
    from .mlx_utils import push_to_hub_gguf
    tokenizer = tokenizer or self._tokenizer
    push_to_hub_gguf(self, tokenizer, repo_id, repo_id=repo_id,
                     quantization_method=quantization_method, **kwargs)


def _mlx_save_lora_adapters(self, path, adapter_config=None):
    from .mlx_utils import save_lora_adapters
    save_lora_adapters(self, path, adapter_config=adapter_config)


def _patch_mlx_saving(model, tokenizer):
    """Attach save/push methods to the model, matching unsloth's CUDA pattern."""
    model._tokenizer = tokenizer
    model.save_pretrained_merged = types.MethodType(_mlx_save_pretrained_merged, model)
    model.save_pretrained_gguf   = types.MethodType(_mlx_save_pretrained_gguf, model)
    model.push_to_hub_merged     = types.MethodType(_mlx_push_to_hub_merged, model)
    model.push_to_hub_gguf       = types.MethodType(_mlx_push_to_hub_gguf, model)
    model.save_lora_adapters     = types.MethodType(_mlx_save_lora_adapters, model)


def _lora_walk_module(model, lora_config, target_modules, attr_names):
    """Walk a module tree and replace matching Linear/QuantizedLinear with LoRA.

    Used for vision encoders that don't have the flat `.layers` structure
    expected by mlx-lm's `linear_to_lora_layers`.
    """
    import mlx.nn as nn
    try:
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError:
        return

    for attr_name in attr_names:
        root = getattr(model, attr_name, None)
        if root is None:
            continue

        def _walk(module):
            for name, child in module.named_modules():
                leaf_name = name.split(".")[-1] if "." in name else name
                if leaf_name not in target_modules:
                    continue
                if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                    lora_layer = LoRALinear.from_base(
                        child,
                        r=lora_config["rank"],
                        dropout=lora_config.get("dropout", 0.0),
                        scale=lora_config["scale"],
                    )
                    # Navigate to parent and replace
                    parts = name.split(".")
                    parent = root
                    for p in parts[:-1]:
                        try:
                            parent = parent[int(p)]
                        except (ValueError, TypeError):
                            parent = getattr(parent, p)
                    setattr(parent, parts[-1], lora_layer)

        _walk(root)
        break  # These are alternative names for the same component — stop after first hit


class FastMLXModel:
    """MLX model loader for Apple Silicon.

    Mirrors the unsloth GPU API so notebooks work with minimal changes:
        model, tokenizer = FastLanguageModel.from_pretrained(...)
        model = FastLanguageModel.get_peft_model(model, r=16)

    Pass any HuggingFace model name directly — mlx-lm handles loading:
        "mlx-community/Llama-3.2-1B-Instruct-4bit"   (pre-quantized MLX)
        "mlx-community/Llama-3.2-1B-Instruct-8bit"   (8-bit MLX)
        "meta-llama/Llama-3.2-1B-Instruct"            (full precision)
        "Qwen/Qwen2.5-7B-Instruct"                    (any HF model)
    """

    @staticmethod
    def from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        token=None,
        trust_remote_code=False,
        text_only=None,
        **kwargs,  # Accept and ignore GPU-only kwargs
    ):
        """Load a model via mlx-lm (text) or mlx-vlm (vision) on Apple Silicon.

        Args:
            model_name: Any HuggingFace repo name.
            max_seq_length: Maximum sequence length for training.
            load_in_4bit: Accepted for API compat with CUDA unsloth.
            token: HuggingFace token for gated models.
            text_only: Loading mode:
                None  — auto-detect from config (default)
                True  — force text-only via mlx-lm
                False — force VLM via mlx-vlm
        """
        try:
            from mlx_lm import load as mlx_load
            from mlx_lm.utils import _download
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        # Step 1: Download config to decide loading path
        try:
            local_path = str(_download(model_name))
            config_path = local_path + "/config.json"
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except Exception:
            config_data = {}
            local_path = None

        # Step 2: Check unsloth custom loader registry
        model_type = config_data.get("model_type", "")
        try:
            from unsloth.models.mlx import get_unsloth_loader
            custom_loader = get_unsloth_loader(model_type)
        except (ImportError, AttributeError):
            # AttributeError: torch installed without CUDA triggers
            # torch._C._cuda_getCurrentRawStream failures in unsloth.kernels
            custom_loader = None

        if custom_loader is not None:
            model, tokenizer_or_processor = custom_loader(
                model_name, config_data, max_seq_length=max_seq_length, token=token
            )
            model._config = config_data
            model._hf_repo = model_name
            model._src_path = local_path
            model.max_seq_length = max_seq_length
            _patch_mlx_saving(model, tokenizer_or_processor)
            return model, tokenizer_or_processor

        # Step 3: Route based on text_only
        is_vlm = False
        if text_only is True:
            is_vlm = False
        elif text_only is False:
            is_vlm = True
        else:
            is_vlm = _is_vlm(config_data)

        tokenizer_config = {}
        if token:
            tokenizer_config["token"] = token

        if is_vlm:
            # VLM path via mlx-vlm
            try:
                from mlx_vlm import load as vlm_load
            except ImportError:
                raise ImportError(
                    "Unsloth: mlx-vlm is required for Vision Language Models. "
                    "Install via: pip install mlx-vlm\n"
                    "Or pass text_only=True to load as text-only via mlx-lm."
                )

            if text_only is False and not _is_vlm(config_data):
                warnings.warn(
                    f"text_only=False but '{model_name}' does not appear to be a VLM. "
                    f"Attempting mlx_vlm.load() anyway — this may fail.",
                    stacklevel=2,
                )

            already_quantized = _vlm_config_is_already_quantized(config_data)
            q_bits = kwargs.pop("q_bits", 4)
            q_group_size = kwargs.pop("q_group_size", 64)
            want_runtime_quant = load_in_4bit and not already_quantized

            if load_in_4bit and already_quantized:
                warnings.warn(
                    f"Unsloth: '{model_name}' is already quantized — "
                    f"ignoring load_in_4bit.",
                    stacklevel=2,
                )

            if want_runtime_quant:
                _ensure_vlm_load_model_patched()
                print(f"Unsloth: Loading {model_name} via mlx-vlm (VLM, "
                      f"runtime {q_bits}-bit quantization)...")
                model, processor = vlm_load(
                    model_name, q_bits=q_bits, q_group_size=q_group_size,
                )
            else:
                print(f"Unsloth: Loading {model_name} via mlx-vlm (VLM)...")
                model, processor = vlm_load(model_name)

            model._is_vlm_model = True
            model._processor = processor
            _fix_gemma4_kv_sharing(model)

            model._config = getattr(model, "_config", config_data)
            model._hf_repo = model_name
            model._src_path = local_path
            model.max_seq_length = max_seq_length

            _patch_mlx_saving(model, processor)
            return model, processor
        else:
            # Text path via mlx-lm (original behavior)
            print(f"Unsloth: Loading {model_name} via mlx-lm...")
            model, tokenizer, config = mlx_load(
                model_name,
                tokenizer_config=tokenizer_config if tokenizer_config else None,
                return_config=True,
            )
            model._is_vlm_model = False

            model._config = config
            model._hf_repo = model_name
            model._src_path = local_path
            model.max_seq_length = max_seq_length

            _patch_mlx_saving(model, tokenizer)
            return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r=16,
        target_modules=None,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=2048,
        train_vision=False,
        train_projector=False,
        **kwargs,  # Accept and ignore GPU-only kwargs
    ):
        """Apply LoRA via mlx-lm on Apple Silicon.

        For VLMs, applies LoRA to the language model and optionally to the
        vision tower (train_vision=True) and projector (train_projector=True).
        """
        try:
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for LoRA on Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        lora_config = {
            "rank": r,
            "alpha": lora_alpha,
            "dropout": 0.0,
            "scale": lora_alpha / r,
        }

        is_vlm = getattr(model, "_is_vlm_model", False)

        if is_vlm:
            # VLM path: freeze everything, then apply LoRA selectively
            _fix_missing_no_grad(model)
            _fix_gemma4_kv_sharing(model)
            model.freeze()

            # Apply LoRA to the language model
            lm = model.language_model
            num_layers = 0
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                num_layers = len(lm.model.layers)
            linear_to_lora_layers(
                lm,
                num_layers=num_layers,
                config=lora_config,
                use_dora=False,
            )

            # Optionally apply LoRA to vision tower
            if train_vision:
                _lora_walk_module(model, lora_config, target_modules,
                                  attr_names=("vision_tower", "vision_model",
                                              "vision_encoder"))

            # Optionally unfreeze projector
            if train_projector:
                for attr in ("multi_modal_projector", "mm_projector",
                             "connector", "aligner"):
                    proj = getattr(model, attr, None)
                    if proj is not None:
                        proj.unfreeze()
                        break

            # Unfreeze all LoRA params across the entire tree
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)
        else:
            # Text-only path (original behavior)
            num_layers = 0
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                num_layers = len(model.model.layers)

            linear_to_lora_layers(
                model,
                num_layers=num_layers,
                config=lora_config,
                use_dora=False,
            )

            model.freeze()
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)

        import mlx.utils
        trainable = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
        total = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
        pct = 100.0 * trainable / total if total > 0 else 0
        print(
            f"Unsloth: LoRA applied — {trainable:,} trainable params "
            f"({pct:.2f}% of {total:,} total)"
        )
        return model


# Aliases for backward compat
FastLanguageModel = FastMLXModel
FastModel = FastMLXModel
