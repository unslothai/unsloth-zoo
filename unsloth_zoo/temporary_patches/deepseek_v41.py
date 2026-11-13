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

"""DeepSeek-V4.1 (community port, transformers.models.deepseek_v41) training fixes. No-ops without the port.

1. Gradient checkpointing. The decoder layers are GradientCheckpointingLayers, but the port leaves
   `supports_gradient_checkpointing` False, and the CSA2 group state is one per-forward `shared` dict that KV / index
   source layers write and later layers read. A checkpoint recompute of a reader runs after the later sources have
   overwritten it (layer 3 recomputes with layer 4's compressed KV / indexer keys), so the replayed forward differs
   from the original one. Each checkpointed layer call now gets a private copy of `shared` that is reset to the state
   it saw in the first forward before every run (forward and recompute); its writes go back to the real dict once.
   Compiled helpers run compiled in both passes. `prepare_compiled_checkpoint()` specializes floats so a compiled
   region keeps one graph across the forward and the recompute (see there).
2. Fake-quant gradients. `_fake_quant_fp8_block` (window KV, QAT) routes the backward through a float8 cast, so the
   gradient itself is rounded to e4m3 and underflows to exactly zero (`kv_proj` / `kv_norm` never learn);
   `_fake_quant_fp4_block` (compressed KV) passes a gradient only through the block amax. Training uses the
   straight-through estimator; the forward values are unchanged (same function, same inputs).
3. Grouped LoRA for `o_a_proj` (DeepseekV41GroupedLinear) when unsloth's own is not registered (plain PEFT, CPU).
4. `config.num_experts` alias for TRL's MoE load-balancing path.
5. LoRA on the pooled compressor's kv_proj / gate_proj, which the port reads as `.weight` (adapter never applied).
"""

import importlib.util
import logging
import os
from functools import partial

import torch

# Only TEMPORARY_PATCHES from .common: the CPU runtime loads this file on its own (cpu/runtime.py).
from .common import TEMPORARY_PATCHES

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
logger = logging.getLogger(__name__)

__all__ = [
    "patch_deepseek_v41_gradient_checkpointing",
    "patch_deepseek_v41_fake_quant_ste",
    "patch_deepseek_v41_grouped_lora",
    "patch_deepseek_v41_config_aliases",
    "patch_deepseek_v41_engram_table",
    "prepare_compiled_checkpoint",
]


def _modeling():
    try:
        if importlib.util.find_spec("transformers.models.deepseek_v41") is None:
            return None
    except Exception:
        return None
    try:
        from transformers.models.deepseek_v41 import modeling_deepseek_v41
    except Exception:
        return None
    return modeling_deepseek_v41


# Checkpointed layer runs (first forward + recompute), for tests and engagement checks.
REPLAY_STATS = {"runs": 0, "compressor_lora": 0}


def prepare_compiled_checkpoint():
    """Let the checkpointed decoder layers run torch.compile'd helpers. Returns True when it applied.

    The layers always checkpoint non-reentrant (see `patch_deepseek_v41_gradient_checkpointing`), which checks that
    the recompute saves the same tensors as the forward. With unspecialized floats (torch's default), a compiled region
    holding a LoRA forward lifts `scaling` to a 0-dim CPU tensor input, and the recompute saved it and `lora_B` in the
    opposite order from the forward ("Recomputed values ... have different metadata"). Specialized floats bake the
    per-adapter constants in, one graph per region in both passes. Set once at load, before anything compiles: Dynamo
    guards its config, so toggling it per call would recompile.
    """
    dynamo_config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_config is None or not hasattr(dynamo_config, "specialize_float"):
        return False
    dynamo_config.specialize_float = True
    return True


class _ReplayShared(dict):
    """The `shared` dict one checkpointed layer call sees; `_snapshot` is the state at its first forward."""

    __slots__ = ("_snapshot",)


def patch_deepseek_v41_gradient_checkpointing():
    modeling = _modeling()
    if modeling is None:
        return
    layer_cls = getattr(modeling, "DeepseekV41DecoderLayer", None)
    pretrained_cls = getattr(modeling, "DeepseekV41PreTrainedModel", None)
    if layer_cls is None or pretrained_cls is None or getattr(layer_cls.__call__, "_unsloth_patched", False):
        return
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except Exception:
        return
    if not issubclass(layer_cls, GradientCheckpointingLayer):
        return
    import inspect

    if "shared" not in inspect.signature(layer_cls.forward).parameters:
        return
    base_call = GradientCheckpointingLayer.__call__

    def __call__(self, *args, **kwargs):
        shared = kwargs.get("shared")
        if not (self.gradient_checkpointing and self.training) or not isinstance(shared, dict):
            with _compressor_lora(self):
                return base_call(self, *args, **kwargs)
        replay = _ReplayShared(shared)
        replay._snapshot = dict(shared)
        kwargs["shared"] = replay
        # Same cache handling as GradientCheckpointingLayer.__call__.
        if kwargs.get("use_cache"):
            kwargs["use_cache"] = False
        if not getattr(self, "_can_checkpoint_with_cache", False):
            for key in ("past_key_values", "layer_past"):
                if kwargs.get(key) is not None:
                    kwargs[key] = None

        def run(*inner_args, **inner_kwargs):
            REPLAY_STATS["runs"] += 1
            replay.clear()
            replay.update(replay._snapshot)
            with _compressor_lora(self):
                return torch.nn.Module.__call__(self, *inner_args, **inner_kwargs)

        # Non-reentrant, always: the source layer's shared KV tensors leave through `replay`, not through the
        # checkpoint's outputs. A reentrant checkpoint (Unsloth's smart checkpointer forces one) would make them
        # under no_grad, and the layers that read them would send their source projections no gradient.
        out = _non_reentrant_checkpoint()(partial(run, **kwargs), *args, use_reentrant = False)
        shared.clear()
        shared.update(replay)
        return out

    # The port's own compressor, before a compiled cache replaces the class in the modeling module.
    _pooled_reads_weight(getattr(modeling, "DeepseekV41Compressor", None))
    _ORIGINAL_COMPRESSOR.append(getattr(modeling, "DeepseekV41Compressor", None))
    __call__._unsloth_patched = True
    layer_cls.__call__ = __call__
    pretrained_cls.supports_gradient_checkpointing = True
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: DeepSeek-V4.1 gradient checkpointing replays each layer with its own CSA2 group state.")
pass

TEMPORARY_PATCHES.append(patch_deepseek_v41_gradient_checkpointing)


def _non_reentrant_checkpoint():
    """torch's own checkpoint, also when Unsloth's smart checkpointing replaced torch.utils.checkpoint.checkpoint."""
    import torch.utils.checkpoint as ckpt
    return (getattr(ckpt, "_unsloth_pristine_checkpoint", None) or getattr(ckpt, "_old_checkpoint", None)
            or ckpt.checkpoint)


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fn):
        return fn(x)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _straight_through(fn):
    def wrapper(x, *args, **kwargs):
        if not (torch.is_grad_enabled() and isinstance(x, torch.Tensor) and x.requires_grad):
            return fn(x, *args, **kwargs)
        return _StraightThrough.apply(x, lambda t: fn(t, *args, **kwargs))

    wrapper._unsloth_patched = True
    wrapper._unsloth_original = fn
    wrapper.__name__ = getattr(fn, "__name__", "fake_quant")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def patch_deepseek_v41_fake_quant_ste():
    modeling = _modeling()
    if modeling is None:
        return
    patched = []
    for name in ("_fake_quant_fp8_block", "_fake_quant_fp4_block"):
        fn = getattr(modeling, name, None)
        if fn is None or getattr(fn, "_unsloth_patched", False):
            continue
        setattr(modeling, name, _straight_through(fn))
        patched.append(name)
    if patched and UNSLOTH_ENABLE_LOGGING:
        logger.info(f"Unsloth: DeepSeek-V4.1 straight-through gradients for {patched}.")
pass

TEMPORARY_PATCHES.append(patch_deepseek_v41_fake_quant_ste)


# 3. LoRA on the grouped output projection (fallback when unsloth's own grouped LoRA is not registered: plain PEFT,
#    the CPU loader). `o_a_proj` is a DeepseekV41GroupedLinear: an nn.Linear subclass whose
#    weight [G * out_g, in_g] holds G independent blocks, applied to x [..., G, in_g] -> [..., G, out_g]. PEFT wraps it
#    as a plain lora.Linear, whose B(A(x)) is [..., G, G * out_g] and cannot be added to the base output. The grouped
#    LoRA keeps PEFT's A [r, in_g] / B [G * out_g, r] (so delta_weight = B @ A has the base weight's shape and
#    merge / unmerge / save are unchanged) and applies group g's rows of B to group g's input, which is exactly what
#    the merged weight computes.
_GROUPED_LORA_CLS = None


def _grouped_lora_linear_cls():
    global _GROUPED_LORA_CLS
    if _GROUPED_LORA_CLS is not None:
        return _GROUPED_LORA_CLS
    from peft.tuners.lora.layer import Linear as LoraLinear

    class DeepseekV41GroupedLoraLinear(LoraLinear):
        def forward(self, x, *args, **kwargs):
            if (
                self.disable_adapters
                or self.merged
                or kwargs.get("adapter_names") is not None
                or not any(a in self.lora_A for a in self.active_adapters)
            ):
                return super().forward(x, *args, **kwargs)
            base = self.get_base_layer()
            n_groups = base.n_groups
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A or self.use_dora.get(active_adapter, False):
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                h = lora_A(dropout(x.to(lora_A.weight.dtype)))  # [..., G, r]
                b = lora_B.weight.view(n_groups, -1, lora_B.weight.shape[-1])  # [G, out_g, r]
                delta = torch.einsum("...gr,gor->...go", h, b)
                if lora_B.bias is not None:
                    delta = delta + lora_B.bias.view(n_groups, -1)
                result = result + (delta * scaling).to(result.dtype)
            return result

    _GROUPED_LORA_CLS = DeepseekV41GroupedLoraLinear
    return _GROUPED_LORA_CLS


def patch_deepseek_v41_grouped_lora():
    modeling = _modeling()
    if modeling is None:
        return
    grouped_cls = getattr(modeling, "DeepseekV41GroupedLinear", None)
    if grouped_cls is None or not hasattr(grouped_cls, "__init__"):
        return
    try:
        from peft.tuners.lora.model import LoraModel
        from peft.tuners.tuners_utils import BaseTunerLayer
    except Exception:
        return
    create = LoraModel.__dict__.get("_create_new_module")
    if create is None or getattr(getattr(create, "__func__", create), "_unsloth_dsv41_grouped", False):
        return
    original = create.__func__ if isinstance(create, staticmethod) else create

    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        base = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        # unsloth's get_peft_model registers its own grouped LoRA (unsloth/models/grouped_linear_lora.py) as a PEFT
        # custom module; that one wins. This covers plain PEFT and the CPU loader.
        custom = getattr(lora_config, "_custom_modules", None) or {}
        if (
            type(base) is grouped_cls
            and not any(isinstance(base, key) for key in custom)
            and not getattr(lora_config, "use_dora", False)
        ):
            kwargs.pop("loaded_in_8bit", None)
            kwargs.pop("loaded_in_4bit", None)
            return _grouped_lora_linear_cls()(target, adapter_name, config=lora_config, **kwargs)
        return original(lora_config, adapter_name, target, **kwargs)

    _create_new_module._unsloth_dsv41_grouped = True
    LoraModel._create_new_module = staticmethod(_create_new_module)
    if UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: DeepSeek-V4.1 grouped LoRA for DeepseekV41GroupedLinear (o_a_proj).")
pass

TEMPORARY_PATCHES.append(patch_deepseek_v41_grouped_lora)


# 4. `num_experts`: the text config names its routed experts `n_routed_experts` (aliased only as `num_local_experts`),
#    while TRL's MoE load-balancing path (and other MoE helpers) read `config.num_experts`.
def patch_deepseek_v41_config_aliases():
    modeling = _modeling()
    if modeling is None:
        return
    try:
        from transformers.models.deepseek_v41.configuration_deepseek_v41 import DeepseekV41TextConfig
    except Exception:
        return
    attribute_map = dict(getattr(DeepseekV41TextConfig, "attribute_map", {}) or {})
    if "num_experts" in attribute_map or "num_experts" in getattr(DeepseekV41TextConfig, "__dataclass_fields__", {}):
        return
    attribute_map["num_experts"] = "n_routed_experts"
    DeepseekV41TextConfig.attribute_map = attribute_map
pass

TEMPORARY_PATCHES.append(patch_deepseek_v41_config_aliases)


# 5. LoRA on the pooled (ratio > 1) compressor. That branch computes in fp32 as
#    `F.linear(hidden_states.float(), self.kv_proj.weight.float())` (same for gate_proj): it reads the weight, never
#    calls the module, so an adapter on compressor.kv_proj / gate_proj is skipped and gets no gradient. During the
#    layer call the two modules are swapped for views whose `.weight` is the differentiable merged weight
#    W + scaling * B @ A (fp32), which is exactly the adapted projection (LoRA dropout is not applied there). The swap is
#    done from the decoder layer, because the compiled cache replaces the compressor and attention classes.
class _MergedLoraWeight:
    __slots__ = ("module",)

    def __init__(self, module):
        self.module = module

    def merged_weight(self):
        module = self.module
        weight = module.get_base_layer().weight.float()
        for name in _active_adapters(module):
            if name not in module.lora_A:
                continue
            lora_A, lora_B = module.lora_A[name].weight, module.lora_B[name].weight
            weight = weight + (lora_B.float() @ lora_A.float()) * module.scaling[name]
        REPLAY_STATS["compressor_lora"] += 1
        return weight

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.merged_weight().to(x.dtype))

    def __getattr__(self, name):
        # Not a property: an AttributeError inside a property would fall back to __getattr__ and hand out the
        # base weight silently.
        if name == "weight":
            return self.merged_weight()
        return getattr(self.module, name)


def _active_adapters(module):
    active = getattr(module, "active_adapters", None)
    if active is None:
        active = getattr(module, "active_adapter", ())
    return [active] if isinstance(active, str) else list(active or ())


def _lora_active(module):
    lora_A = getattr(module, "lora_A", None)
    return (
        lora_A is not None
        and len(lora_A) > 0
        and not getattr(module, "disable_adapters", False)
        and not getattr(module, "merged", False)
        and any(name in lora_A for name in _active_adapters(module))
    )


class _compressor_lora:
    """Swap a pooled compressor's adapted kv_proj / gate_proj for merged-weight views during one layer call (forward
    or checkpoint recompute). Installed from the decoder layer's __call__, the one class the compiled cache keeps."""

    __slots__ = ("modules", "swapped")

    def __init__(self, layer):
        compressor = getattr(getattr(layer, "self_attn", None), "compressor", None)
        self.modules = getattr(compressor, "_modules", None)
        self.swapped = {}
        if not self.modules or getattr(compressor, "gate_proj", None) is None:
            return
        if not any(_pooled_reads_weight(cls) for cls in (type(compressor), *_ORIGINAL_COMPRESSOR)):
            return
        self.swapped = {
            name: self.modules[name]
            for name in ("kv_proj", "gate_proj")
            if self.modules.get(name) is not None and _lora_active(self.modules[name])
        }

    def __enter__(self):
        for name, module in self.swapped.items():
            self.modules[name] = _MergedLoraWeight(module)
        return self

    def __exit__(self, *exc):
        if self.swapped:
            self.modules.update(self.swapped)
        return False


_POOLED_READS_WEIGHT = {}
_ORIGINAL_COMPRESSOR = []


def _pooled_reads_weight(cls):
    """Whether this compressor class's pooled branch reads `self.kv_proj.weight`. The compiled cache's copy forwards
    to a module-level function, so the port's own class (read before the cache swaps it in) is checked too."""
    if cls is None:
        return False
    if cls not in _POOLED_READS_WEIGHT:
        import inspect

        try:
            _POOLED_READS_WEIGHT[cls] = "self.kv_proj.weight" in inspect.getsource(cls.forward)
        except Exception:
            _POOLED_READS_WEIGHT[cls] = False
    return _POOLED_READS_WEIGHT[cls]


# ----------------------------------------------------------------------------------------------
# Engram tables on disk
# ----------------------------------------------------------------------------------------------
# Checkpoint keys of an engram table: DeepSeek-native (`layers.<i>.engram.embed.{weight,scale}`, the released
# files) or the port's own names (`engram_tables.<i>.{weight,weight_scale_inv}`, a transformers save).
_ENGRAM_UNEXPECTED = (
    r"(^|\.)engram_tables\.\d+\.(weight|weight_scale_inv|scale)$",
    r"(^|\.)layers\.\d+\.engram\.embed\.(weight|scale)$",
)


def _engram_table_wanted(nbytes):
    from unsloth_zoo.frozen_table_store import resolve_frozen_table_mode
    mode = os.environ.get("UNSLOTH_FROZEN_TABLE", "auto").strip().lower()
    if os.environ.get("UNSLOTH_ENABLE_FULL_FINETUNING", "0") == "1":
        if mode not in ("disk", "ram"):
            return False
        logger.warning(f"Unsloth: UNSLOTH_FROZEN_TABLE={mode} keeps the DeepSeek-V4.1 engram tables frozen during "
                       "full finetuning; set UNSLOTH_FROZEN_TABLE=gpu to train them.")
    return resolve_frozen_table_mode(nbytes) != "gpu"


def _fp8_weights_kept(model, quantizer=None):
    """True when the load kept FP8 weights (no dequantize): the port's engram lookup then returns float32 rows.
    `quantizer` is the load's (transformers sets model.hf_quantizer only after the weights are in)."""
    if quantizer is None:
        quantizer = getattr(model, "hf_quantizer", None)
    qc = getattr(quantizer, "quantization_config", None)
    if qc is None:
        return False
    method = getattr(qc, "quant_method", None)
    method = getattr(method, "value", method)
    return str(method) == "fp8" and not bool(getattr(qc, "dequantize", False))


def _frozen_engram(modeling, key, emb):
    """An UnslothFrozenTableEmbedding standing in for one DeepseekV41EngramEmbedding, bound to the checkpoint
    after the load: fp8 rows x per-32-channel e8m0 scales (the released files), or plain rows (a bf16 copy)."""
    from unsloth_zoo.frozen_table_store import UnslothFrozenTableEmbedding, make_fp8_block_decode
    block = int(getattr(emb, "block_size", 32))
    table = UnslothFrozenTableEmbedding(
        emb.num_embeddings, emb.embedding_dim, dtype=emb.weight.dtype,
        table_name=f"deepseek_v41 engram layer {key}", row_shape=(emb.embedding_dim,),
    )
    table.checkpoint_prefixes = (f"layers.{key}.engram.embed", f"model.engram_tables.{key}", f"engram_tables.{key}")
    table.checkpoint_planes = {"weight": (r"weight",), "scale": (r"scale", r"weight_scale_inv")}
    table.optional_planes = ("scale",)
    table.export_suffixes = {"weight": "weight", "scale": "weight_scale_inv"}
    table.block_size = block
    # The e8m0 scale goes with fp8 rows only: a bf16 copy (a transformers save) carries an unused weight_scale_inv.
    table.select_planes = lambda dtypes: ("weight", "scale") if "scale" in dtypes and dtypes["weight"].startswith("F8_") else ("weight",)
    table.make_decode = lambda planes: make_fp8_block_decode(block_size=block) if "scale" in planes else None

    def after_bind(model, planes, quantizer=None, table=table):
        # DeepseekV41EngramEmbedding.forward: float32 rows when the fp8 table is kept (dequantize=False); a
        # dequantized load folded the scales into rows of the model dtype, which the lookup returns as is.
        if "scale" not in planes:
            return
        if _fp8_weights_kept(model, quantizer):
            table.table_dtype = torch.float32
        else:
            # The model holds dequantized rows and drops its quantization config: a save must write those rows,
            # not fp8 rows + scales that a reload would read as plain rows.
            table.export_decoded = "weight"

    table.after_bind = after_bind
    return table


def patch_deepseek_v41_engram_table():
    """Build the two ~98 GB engram tables as parameter-free UnslothFrozenTableEmbedding modules on a checkpoint
    load, streamed from the checkpoint files (mmap + MADV_RANDOM) instead of loading into host RAM, the same store
    as Qwen3.8-Flash-Next's n-gram table. UNSLOTH_FROZEN_TABLE=gpu (or auto on a small table) keeps the port's
    DeepseekV41EngramEmbedding."""
    modeling = _modeling()
    if modeling is None:
        return False
    mode = os.environ.get("UNSLOTH_FROZEN_TABLE", "auto").strip().lower()
    if mode == "gpu":
        return False
    text_cls = getattr(modeling, "DeepseekV41TextModel", None)
    emb_cls = getattr(modeling, "DeepseekV41EngramEmbedding", None)
    if text_cls is None or emb_cls is None:
        return False
    try:
        from unsloth_zoo.frozen_table_store import install_frozen_table_load_hook
        if not install_frozen_table_load_hook():
            raise RuntimeError("could not install the frozen table load hook")
    except Exception as e:
        if mode in ("disk", "ram"):
            raise RuntimeError(f"Unsloth: UNSLOTH_FROZEN_TABLE={mode} but the DeepSeek-V4.1 engram patch failed: {e}")
        logger.warning(f"Unsloth: DeepSeek-V4.1 engram table patch failed ({e}); the tables load into RAM.")
        return False
    if not getattr(text_cls.__init__, "_unsloth_frozen_engram", False):
        original_init = text_cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            tables = getattr(self, "engram_tables", None)
            if tables is None:
                return
            for key in list(tables.keys()):
                emb = tables[key]
                if type(emb) is not emb_cls or not emb.weight.is_meta:
                    continue   # from_config keeps a real table
                # The released rows are fp8 (one byte each) plus one e8m0 byte per 32 channels.
                nbytes = emb.num_embeddings * (emb.embedding_dim + emb.embedding_dim // max(1, int(getattr(emb, "block_size", 32))))
                if _engram_table_wanted(nbytes):
                    tables[key] = _frozen_engram(modeling, key, emb)

        __init__._unsloth_frozen_engram = True
        __init__.__wrapped__ = original_init
        text_cls.__init__ = __init__

    # The table keys now match no parameter: transformers files them as unexpected and never reads them.
    from transformers.modeling_utils import PreTrainedModel
    for obj in vars(modeling).values():
        if isinstance(obj, type) and issubclass(obj, PreTrainedModel) and (
            "_keys_to_ignore_on_load_unexpected" in obj.__dict__ or obj.__name__ == "DeepseekV41PreTrainedModel"
        ):
            current = list(obj.__dict__.get("_keys_to_ignore_on_load_unexpected") or [])
            obj._keys_to_ignore_on_load_unexpected = current + [p for p in _ENGRAM_UNEXPECTED if p not in current]
    return True


TEMPORARY_PATCHES.append(patch_deepseek_v41_engram_table)
