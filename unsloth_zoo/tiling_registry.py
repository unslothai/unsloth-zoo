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

"""Sequence tiling registry: which modules are token-local, and how their tiled outputs are put back together.

The tiled forward lives in the instance `forward`; the forward it tiles lives in `module._unsloth_forward`, so other
patches (the fused LoRA MLP kernel) write their forward there and it runs inside every tile."""

import functools
import itertools
import math
import os
import re
import types
import weakref
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from types import MethodType

import torch

from unsloth_zoo import sequence_tiling as _sequence_tiling
from unsloth_zoo.memory_budget import _default_target_gb, _max_flat_qlen
from unsloth_zoo.sequence_tiling import _is_compiling

__all__ = [
    "TileSpec",
    "TilePlan",
    "TilingReport",
    "PARALLEL_RULE",
    "DEFAULT_MODE",
    "parse_mode",
    "install",
    "install_module",
    "uninstall",
    "revalidate",
    "is_tiled",
    "tiled_modules",
    "lookup",
    "needs_rng_state",
    "is_off",
]

PARALLEL_RULE = (
    "Sequence tiling is per rank: it splits only the rows this rank holds. Under Ulysses context parallel the MLP "
    "sees the local sequence shard, so tiles are planned on the shard length. DDP and FSDP2 need nothing extra: "
    "parameters are used normally inside each tile, so their gradients are summed by autograd before any hook fires, "
    "and the tile function is the inner forward, so module hooks (FSDP2 all-gathers) run once per call, not per "
    "tile. Tensor parallel is skipped by default: a row-parallel all-reduce would run once per tile, so modules whose "
    "parameters are TP DTensors stay untiled (coverage reason \"tensor parallel\") unless UNSLOTH_TILED_MLP_TP=1."
)

DEFAULT_MODE = "auto"
_MAX_TILES = 4096
_ENV_MODULES = "UNSLOTH_TILED_MLP_MODULES"
_ENV_TP = "UNSLOTH_TILED_MLP_TP"
_OLD_SUFFIXES = (".mlp", ".ffn", ".feed_forward", ".ff", ".densereludense", ".block_sparse_moe")
_CONTRACTS = ("rows", "rows_flat", "sum", "refuse")


@dataclass(frozen = True)
class TileSpec:
    """`selector`: class name, tuple of class names or `predicate(module) -> bool`. Only the first positional forward
    argument is tiled; a call with more arguments runs untiled.
    `output_contract`: per output, one of rows / rows_flat / sum / refuse; a bare tensor output uses entry 0."""
    selector: object
    output_contract: tuple = ("rows",)
    family: str = ""
    refuse: str = ""

    def __post_init__(self):
        bad = [c for c in self.output_contract if c not in _CONTRACTS]
        if bad:
            raise ValueError(f"Unsloth: unknown tiling output contract {bad}; use one of {_CONTRACTS}")
        if not self.refuse and "refuse" in self.output_contract:
            object.__setattr__(self, "refuse", "an output has no tiling contract")

    def matches(self, module):
        sel = self.selector
        if isinstance(sel, str):
            return type(module).__name__ == sel
        if isinstance(sel, (tuple, list, frozenset, set)):
            return type(module).__name__ in sel
        return bool(sel(module))


_SEQ_MIX_TOKENS = (
    "attention", "attn", "conv", "mamba", "recurren", "rnn", "lstm", "gru", "pool", "shuffle", "merger",
    "mixer", "resampler", "perceiver", "downsample", "upsample", "patchembed", "deltanet", "ssm", "scan", "rotary",
)
# Batch and instance norms mix rows through their statistics in training, so a row tile changes their output.
_SEQ_MIX_TYPES = (
    torch.nn.modules.conv._ConvNd, torch.nn.RNNBase, torch.nn.MultiheadAttention,
    torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.instancenorm._InstanceNorm,
)
_DENSE_NAME = re.compile(r"(mlp|feedforward|feed_forward|ffn|denseactdense|densereludense|densegatedactdense)$", re.I)


def _sequence_mixer(module):
    for m in module.modules():
        if isinstance(m, _SEQ_MIX_TYPES):
            return type(m).__name__
        name = type(m).__name__
        if name == "Conv1D" and type(m).__module__.startswith("transformers"):
            continue
        low = name.lower()
        if any(t in low for t in _SEQ_MIX_TOKENS):
            return name
    return None


def _is_dense_mlp(module):
    name = type(module).__name__
    if not _DENSE_NAME.search(name) or "sparse" in name.lower():
        return False
    if hasattr(module, "experts") or hasattr(module, "router"):
        return False
    return True


def _has_seq_aux_gate(module):
    # DeepSeek remote-code MoEGate: with seq_aux the aux loss is computed per sequence, which a row tile cannot see.
    gate = getattr(module, "gate", None)
    if gate is None or not isinstance(gate, torch.nn.Module):
        return False
    return bool(getattr(gate, "seq_aux", False)) and float(getattr(gate, "alpha", 0.0) or 0.0) > 0


_ROUTER = ("rows", "rows_flat")

_BUILTINS = [
    TileSpec(
        ("SwitchTransformersSparseMLP", "NllbMoeSparseMLP", "GPTSanJapaneseSparseMLP"),
        family = "capacity-moe",
        refuse = "capacity-limited MoE (which tokens are dropped depends on how many share a call)",
    ),
    TileSpec(_has_seq_aux_gate, family = "seq-aux-moe", refuse = "sequence-level aux-loss gate (seq_aux)"),
    # Channel-first input ([B, C, H, W], flattened and transposed inside the forward): the last dim is not features.
    TileSpec(("SegformerMLP",), family = "channel-first", refuse = "channel-first input (features are not the last dim)"),
    TileSpec(
        (
            "Gemma3MultiModalProjector", "Mistral3MultiModalProjector", "AyaVisionMultiModalProjector",
            "Idefics3Connector", "SmolVLMConnector", "Idefics2Connector", "Qwen2VLPatchMerger",
            "Qwen2_5_VLPatchMerger", "Llama4VisionPixelShuffleMLP", "Mistral3PatchMerger",
        ),
        family = "vlm-projector",
        refuse = "sequence-mixing projector (pooling, pixel shuffle or patch merge)",
    ),
    # Token-local top-k MoE blocks. 4.x returns (hidden, router_logits [rows, E]); 5.x returns hidden only.
    TileSpec(
        (
            "Qwen2MoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock", "Qwen3_5MoeSparseMoeBlock", "MixtralSparseMoeBlock",
            "OlmoeSparseMoeBlock", "Qwen3NextSparseMoeBlock", "Qwen3VLMoeTextSparseMoeBlock", "DeepseekV3MoE",
        ),
        output_contract = _ROUTER,
        family = "topk-moe",
    ),
    TileSpec("GptOssMLP", output_contract = _ROUTER, family = "gpt-oss"),
    TileSpec("Llama4TextMoe", output_contract = ("rows_flat", "rows_flat"), family = "llama4-moe"),
    TileSpec(("NemotronHMLP", "NemotronHMoE", "NemotronHMOE"), family = "nemotron-h"),
    TileSpec(
        (
            "LlavaMultiModalProjector", "LlavaNextMultiModalProjector", "LlavaOnevisionMultiModalProjector",
            "PaliGemmaMultiModalProjector", "Llama4MultiModalProjector",
        ),
        family = "vlm-projector",
    ),
]
_DENSE = TileSpec(_is_dense_mlp, family = "dense-mlp")


def lookup(module):
    for spec in _BUILTINS:
        try:
            if spec.matches(module):
                return spec
        except Exception:
            continue
    if _is_dense_mlp(module):
        mixer = _sequence_mixer(module)
        if mixer is not None:
            return TileSpec(_is_dense_mlp, family = "dense-mlp", refuse = f"sequence-mixing submodule {mixer}")
        return _DENSE
    return None


@dataclass(frozen = True)
class TilePlan:
    """mode: auto (budget from free memory; one tile when tiling would not pay), arctic (tile rows = hidden), rows
    (fixed), target (fixed GB budget; target_gb 0 means the free-memory budget, always tiled)."""
    mode: str = DEFAULT_MODE
    rows: int = 0
    target_gb: float = 0.0
    modules: tuple = ()
    padded_length: int = 128

    def spelling(self):
        if self.mode == "rows":
            s = f"rows:{self.rows}"
        elif self.mode == "target" and self.target_gb > 0:
            s = f"target:{self.target_gb:g}"
        else:
            s = self.mode
        if self.modules:
            s += (":" if ":" in s else "::") + ",".join(self.modules)
        return s


_OFF = ("", "0", "false", "off", "no", "none")


def is_off(value):
    """True for every spelling of "tiling off": None, False, 0, "", "0", "false", "off", "no", "none"."""
    if value is None or value is False:
        return True
    if value is True or isinstance(value, TilePlan):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return str(value).strip().lower() in _OFF


def parse_mode(value, padded_length = 128):
    """`True` / "1" / "arctic" / "auto" / "target" / "target:GB" / "rows:N", each optionally followed by
    `:pattern,pattern` (only modules matching a pattern are tiled). Off spellings (see `is_off`) -> None.

    "target" without a size tiles with half of free memory as the budget, always; "auto" uses the same budget but
    keeps one tile when tiling would not lower the step peak enough (see estimate_tiling_gain)."""
    if isinstance(value, TilePlan):
        return value
    if is_off(value):
        return None
    if value is True:
        value = DEFAULT_MODE
    text = str(value).strip()
    parts = text.split(":")
    head = parts[0].strip().lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    modules = tuple(p.strip() for p in (parts[2] if len(parts) > 2 else "").split(",") if p.strip())
    if len(parts) > 3:
        raise ValueError(f"Unsloth: cannot parse tiled MLP mode {value!r}")
    rows, target_gb = 0, 0.0
    if head in ("1", "true", "on", "yes", "auto"):
        mode = "auto"
    elif head == "arctic":
        mode = "arctic"
    elif head == "target":
        mode = "target"
        if arg:
            try:
                target_gb = float(arg)
            except ValueError:
                raise ValueError(f"Unsloth: tiled MLP target:GB needs a number, got {arg!r}")
            if not target_gb > 0:
                raise ValueError(f"Unsloth: tiled MLP target:GB must be > 0, got {arg!r}")
    elif head == "rows":
        try:
            rows = int(arg)
        except ValueError:
            raise ValueError(f"Unsloth: tiled MLP rows:N needs an integer, got {arg!r}")
        if rows < 1:
            raise ValueError(f"Unsloth: tiled MLP rows:N must be >= 1, got {arg!r}")
        mode = "rows"
    else:
        raise ValueError(
            f"Unsloth: unknown tiled MLP mode {value!r}; use auto, arctic, rows:N or target:GB (optionally :modules)"
        )
    return TilePlan(mode = mode, rows = rows, target_gb = target_gb, modules = modules, padded_length = padded_length)


def _tile_rows(plan, hidden, intermediate, nbytes, budget_gb):
    if plan.mode == "rows":
        return plan.rows
    if plan.mode == "arctic":
        return max(1, int(hidden))
    gb = plan.target_gb if (plan.mode == "target" and plan.target_gb > 0) else budget_gb
    return _max_flat_qlen(
        hd = int(hidden), mlp_size = int(intermediate), nbytes = int(nbytes), target_gb = float(gb),
        padded_length = plan.padded_length,
    )


def _bucket(rows, tile_rows):
    # Power-of-two tile count: a compiled caller guards on the bucket, not the length.
    n = 1
    while n < _MAX_TILES and rows > tile_rows * n and n * 2 <= rows:
        n *= 2
    return n


def _int_attr(module, names):
    for name in names:
        v = getattr(module, name, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], int):
            return v[0]
    return None


def _infer_hidden(module):
    for m in module.modules():
        v = _int_attr(m, ("hidden_size", "hidden_dim", "in_features", "d_model", "embed_dim"))
        if v is not None:
            return v
        cfg = getattr(m, "config", None)
        if cfg is not None:
            v = _int_attr(cfg, ("hidden_size", "d_model"))
            if v is not None:
                return v
    return None


def _infer_intermediate(module, hidden):
    v = _int_attr(module, ("intermediate_size", "moe_intermediate_size", "ffn_dim", "d_ff"))
    if v is None:
        cfg = getattr(module, "config", None)
        v = _int_attr(cfg, ("intermediate_size", "moe_intermediate_size")) if cfg is not None else None
    if v is None:
        outs = [m.out_features for m in module.modules() if isinstance(getattr(m, "out_features", None), int)]
        v = max(outs) if outs else None
    return v or 4 * (hidden or 1)


def _input_width(module):
    for m in module.modules():
        v = getattr(m, "in_features", None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def _float_dtype(module):
    counts = {}
    for t in itertools.chain(module.parameters(), module.buffers()):
        if t.is_floating_point():
            counts[t.dtype] = counts.get(t.dtype, 0) + t.numel()
    return max(counts, key = counts.get) if counts else None


def _device(module):
    for t in itertools.chain(module.parameters(), module.buffers()):
        return t.device
    return None


_DROPOUT_TYPES = (getattr(torch.nn.modules.dropout, "_DropoutNd", torch.nn.Dropout),)
# Bumped when a submodule is added under a tiled module (PEFT lora_dropout after install).
_MODULE_TREE_VERSION = [0]
_TRACKED_MODULES = weakref.WeakValueDictionary()


def _bump_module_tree_version(module, name, submodule):
    if _TRACKED_MODULES.get(id(module)) is module:
        _MODULE_TREE_VERSION[0] += 1
    return None


torch.nn.modules.module.register_module_module_registration_hook(_bump_module_tree_version)


def needs_rng_state(module):
    if _is_compiling():
        # Weakrefs are unsupported under fullgraph=True; scan directly.
        return any(m.training and m.p > 0 for m in module.modules() if isinstance(m, _DROPOUT_TYPES))
    cached = module.__dict__.get("_unsloth_tiled_dropouts")
    if cached is None or cached[0] != _MODULE_TREE_VERSION[0] or cached[1] != id(module):
        modules = tuple(module.modules())
        _TRACKED_MODULES.update((id(m), m) for m in modules)
        dropouts = tuple(m for m in modules if isinstance(m, _DROPOUT_TYPES))
        cached = (_MODULE_TREE_VERSION[0], id(module), dropouts)
        module.__dict__["_unsloth_tiled_dropouts"] = cached
    for m in cached[2]:
        if m.training and m.p > 0:
            return True
    return False


_DETERMINISTIC_FAMILIES = frozenset({"dense-mlp", "topk-moe", "gpt-oss", "llama4-moe", "nemotron-h", "vlm-projector"})
# torch.nn classes that never draw random numbers (dropout is checked live; RReLU samples its slope in training).
_DETERMINISTIC_TORCH_NN = frozenset({
    "Module", "Linear", "Identity", "Bilinear", "LazyLinear", "Embedding", "EmbeddingBag",
    "ModuleList", "ModuleDict", "Sequential", "ParameterList", "ParameterDict", "Flatten", "Unflatten",
    "LayerNorm", "RMSNorm", "GroupNorm", "LocalResponseNorm",
    "ReLU", "ReLU6", "GELU", "SiLU", "Mish", "Tanh", "Sigmoid", "Hardtanh", "Hardswish", "Hardsigmoid", "ELU",
    "SELU", "CELU", "GLU", "LeakyReLU", "PReLU", "Softplus", "Softsign", "Softshrink", "Hardshrink", "Tanhshrink",
    "Threshold", "LogSigmoid", "Softmax", "Softmin", "LogSoftmax",
})
# Library code whose classes may skip the RNG replay: transformers' own models (not remote code, which loads as
# transformers_modules), PEFT, bitsandbytes layers and Unsloth. Anything else (remote or user code) replays.
_TRUSTED_LIBRARIES = ("transformers.", "peft.", "bitsandbytes.", "unsloth.", "unsloth_zoo.")
# PEFT's lora_dropout source names dropout but only draws through the live-checked submodule.
_DROPOUT_ONLY_MODULES = ("peft.", "unsloth.kernels.fast_lora", "bitsandbytes.")
_RANDOM_SOURCE = re.compile(
    r"dropout|rrelu|\brand|randn|randint|randperm|multinomial|normal_|uniform_|bernoulli|exponential_|poisson|"
    r"cauchy_|geometric_|log_normal_|jitter|noise|gumbel|torch\.empty_like\(.*\)\.\w+_\("
)
_SOURCE_VERDICT = {}


def _code_may_draw_random(fn):
    """True unless fn's source is readable and names no random op. Helpers are not followed, hence the trusted-class gate."""
    while isinstance(fn, functools.partial):
        fn = fn.func
    fn = getattr(fn, "__func__", fn)
    module_name = getattr(fn, "__module__", "") or ""
    if module_name.startswith(_DROPOUT_ONLY_MODULES):
        return False
    key = getattr(fn, "__code__", None) or fn
    try:
        hit = _SOURCE_VERDICT.get(key)
    except TypeError:
        hit = None
    if hit is not None:
        return hit
    try:
        import inspect
        verdict = bool(_RANDOM_SOURCE.search(inspect.getsource(fn)))
    except Exception:
        verdict = True
    try:
        _SOURCE_VERDICT[key] = verdict
    except TypeError:
        pass
    return verdict


def _class_is_trusted(m):
    cls = type(m)
    module_name = cls.__module__ or ""
    if module_name == "torch.nn" or module_name.startswith("torch.nn."):
        return cls.__name__ in _DETERMINISTIC_TORCH_NN or isinstance(m, _DROPOUT_TYPES)
    return module_name.startswith(_TRUSTED_LIBRARIES)


def _subtree_may_draw_random(module):
    for m in module.modules():
        if not _class_is_trusted(m):
            return True
        if isinstance(m, _DROPOUT_TYPES):
            continue
        d = m.__dict__
        candidates = [d.get("forward"), d.get("_unsloth_forward")]
        if not type(m).__module__.startswith("torch.nn"):
            candidates.append(type(m).forward)
        for fn in candidates:
            if fn is None or _is_ours(fn):
                continue
            if _code_may_draw_random(fn):
                return True
    return False


def needs_rng_replay(module, spec):
    """Off only when provably unnecessary; forced, opted-in and unknown classes always replay, in training and eval."""
    if needs_rng_state(module):
        return True
    if spec is None or spec.family not in _DETERMINISTIC_FAMILIES:
        return True
    cached = module.__dict__.get("_unsloth_tiled_random")
    if _is_compiling():
        # Source inspection cannot run while tracing: use the eager verdict (computed at install), else replay.
        return True if cached is None else cached[2]
    return _random_verdict(module)[2]


def _random_verdict(module):
    inner = module.__dict__.get("_unsloth_forward")
    cached = module.__dict__.get("_unsloth_tiled_random")
    if cached is None or cached[0] != _MODULE_TREE_VERSION[0] or cached[1] != id(module) or \
            cached[3] is not inner:
        _TRACKED_MODULES.update((id(m), m) for m in module.modules())
        cached = (_MODULE_TREE_VERSION[0], id(module), _subtree_may_draw_random(module), inner)
        module.__dict__["_unsloth_tiled_random"] = cached
    return cached


class _TileState:
    def __init__(self, spec, plan, hidden, intermediate, nbytes, budget_gb):
        self.spec = spec
        self.plan = plan
        self.hidden = hidden
        self.intermediate = intermediate
        self.nbytes = nbytes
        self.budget_gb = budget_gb
        self.tile_rows = _tile_rows(plan, hidden, intermediate, nbytes, budget_gb) if hidden else None
        self.in_width = None
        self.model_stats = None


def _bind(module, fn):
    if fn is None:
        fn = type(module).forward
    if isinstance(fn, types.FunctionType):
        # functools.partial, not MethodType: Dynamo cannot build a bound method while tracing (fullgraph).
        return functools.partial(fn, module)
    return fn


def _adoptable(fn):
    """A plain function on the instance is already bound; wrap it so _bind does not pass the module again."""
    if isinstance(fn, types.FunctionType):
        return functools.partial(fn)
    return fn


def _inner_forward(module):
    return _bind(module, module.__dict__.get("_unsloth_forward"))


def _flat(o, contract):
    if o is None or contract != "rows":
        return o
    return o.reshape(-1, o.shape[-1])


def _unflat(o, contract, lead):
    if o is None or contract != "rows":
        return o
    return o.reshape(*lead, o.shape[-1])


def _tile_fn(module, feed_3d, contract, x_tile):
    rows = x_tile.shape[0]
    head = (1, rows) if feed_3d else (rows,)
    out = _inner_forward(module)(x_tile.reshape(*head, x_tile.shape[-1]))
    if torch.is_tensor(out):
        return _flat(out, contract[0])
    return tuple(_flat(o, contract[k] if k < len(contract) else "rows") for k, o in enumerate(out))


def _engine_contract(contract):
    return None if set(contract) <= {"rows", "rows_flat"} else contract


def _run_tiled(module, state, x, n_tiles, compile_tile = False):
    contract = state.spec.output_contract
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    fn = functools.partial(_tile_fn, module, x.dim() >= 3, tuple(contract))
    if compile_tile:
        fn = _sequence_tiling._compiled_tile_fn(fn)
    out = _sequence_tiling.tiled_call(
        fn, x2,
        n_tiles = n_tiles, needs_rng = needs_rng_replay(module, state.spec),
        output_contract = _engine_contract(contract),
    )
    if torch.is_tensor(out):
        return _unflat(out, contract[0], lead)
    return tuple(_unflat(o, contract[k] if k < len(contract) else "rows", lead) for k, o in enumerate(out))


def _tiled_forward(self, *args, **kwargs):
    inner = _inner_forward(self)
    state = self.__dict__.get("_unsloth_tiling")
    if state is None or kwargs or len(args) != 1 or not torch.is_tensor(args[0]) or args[0].dim() < 2:
        return inner(*args, **kwargs)
    x = args[0]
    lead = x.shape[:-1]
    if state.in_width is not None and x.shape[-1] != state.in_width:
        # Channel-first input: the rows of x are not tokens, so tiling them would change the output.
        return inner(*args)
    rows = math.prod(lead)
    tile_rows = state.tile_rows
    if tile_rows is None:
        tile_rows = _tile_rows(
            state.plan, x.shape[-1], state.intermediate or 4 * x.shape[-1], x.element_size(),
            state.budget_gb or 4.0,
        )
    n_tiles = _bucket(rows, tile_rows)
    if n_tiles <= 1:
        return inner(*args)
    if state.plan.mode == "auto" and not _auto_tiling_pays(state, rows, x.element_size(), n_tiles):
        return inner(*args)
    if _is_compiling():
        mode = _sequence_tiling._outside_graph_mode()
        if mode != "unroll":
            return _run_tiled_outside_graph(self, state, x, n_tiles, mode == "compile_tile")
    return _run_tiled(self, state, x, n_tiles)


_run_tiled_outside_graph = torch.compiler.disable(_run_tiled, recursive = True)


AUTO_MIN_SAVING_MIB = 512
AUTO_MIN_SAVING_FRACTION = 0.10
_AUTO_DECLINED = set()


def estimate_tiling_gain(rows, hidden, intermediate, nbytes, n_tiles, param_bytes = 0, n_layers = 1):
    """(saving, peak) in bytes for one step, per decoder layer. Calibrated on G4: Qwen3-0.6B 1 x 16k, Qwen3-4B 4 x 8k."""
    mlp = 4 * rows * intermediate * nbytes
    other = 4 * rows * hidden * nbytes
    saving = max(0, mlp - max(other, mlp // max(1, n_tiles)))
    peak = param_bytes + n_layers * rows * hidden * nbytes + max(mlp, other)
    return saving, peak


def _auto_tiling_pays(state, rows, nbytes, n_tiles):
    param_bytes, n_layers = state.model_stats or (0, 1)
    intermediate = state.intermediate or 4 * (state.hidden or 1)
    saving, peak = estimate_tiling_gain(rows, state.hidden or 1, intermediate, nbytes, n_tiles, param_bytes, n_layers)
    pays = saving >= AUTO_MIN_SAVING_MIB * 2**20 and saving >= AUTO_MIN_SAVING_FRACTION * peak
    if not pays and not _is_compiling():
        key = (id(state.model_stats), int(rows))
        if key not in _AUTO_DECLINED and _is_rank0():
            _AUTO_DECLINED.add(key)
            print(
                f"Unsloth: Tiled MLP (auto) keeps one tile at {int(rows)} rows: estimated saving "
                f"{saving / 2**20:.0f} MiB of a ~{peak / 2**20:.0f} MiB step peak is too small to pay for the recompute.",
                flush = True,
            )
    return pays


_TILED_FORWARDS = {}


def _tiled_forward_for(cls):
    """_tiled_forward with its own code object per class: one shared function hits Dynamo's recompile limit."""
    fn = _TILED_FORWARDS.get(cls)
    if fn is None:
        code = _tiled_forward.__code__.replace(co_name = f"_tiled_forward_{len(_TILED_FORWARDS)}")
        fn = types.FunctionType(code, _tiled_forward.__globals__, code.co_name)
        fn._unsloth_tiled_forward = True
        _TILED_FORWARDS[cls] = fn
    return fn


_tiled_forward._unsloth_tiled_forward = True


def _is_ours(fn):
    return getattr(getattr(fn, "__func__", None), "_unsloth_tiled_forward", False)


def is_tiled(module):
    return _is_ours(module.__dict__.get("forward")) and "_unsloth_tiling" in module.__dict__


def tiled_modules(model):
    return [(n, m) for n, m in model.named_modules() if is_tiled(m)]


# Attributes other patches use to call "the forward before mine"; they must never point back at the tiled forward
# once it has adopted their forward as its inner one, or the two call each other forever.
_CHAIN_ATTRS = ("_orig_moe_forward", "_old_forward")


def _unchain(module, target):
    d = module.__dict__
    for attr in _CHAIN_ATTRS:
        if _is_ours(d.get(attr)):
            d[attr] = target


def _budget(device, cache):
    key = str(device)
    if key not in cache:
        try:
            cache[key] = float(_default_target_gb(device))
        except Exception:
            cache[key] = 4.0
    return cache[key]


def _attach(module, spec, plan, budget_cache):
    d = module.__dict__
    current = d.get("forward")
    if not _is_ours(current):
        if current is not None:
            prior = d.get("_unsloth_forward")
            d["_unsloth_forward"] = _adoptable(current)
            _unchain(module, _bind(module, prior))
        elif "_unsloth_forward" not in d:
            d["_unsloth_forward"] = type(module).forward
        d["forward"] = MethodType(_tiled_forward_for(type(module)), module)
    hidden = _infer_hidden(module)
    dtype = _float_dtype(module)
    nbytes = torch.empty((), dtype = dtype).element_size() if dtype is not None else 2
    free_budget = plan.mode == "auto" or (plan.mode == "target" and not plan.target_gb > 0)
    budget = _budget(_device(module), budget_cache) if free_budget else 0.0
    state = d["_unsloth_tiling"] = _TileState(spec, plan, hidden, _infer_intermediate(module, hidden), nbytes, budget)
    state.in_width = _input_width(module)
    _random_verdict(module)


def _detach(module):
    d = module.__dict__
    inner = d.pop("_unsloth_forward", None)
    d.pop("_unsloth_tiling", None)
    d.pop("_unsloth_tiled_random", None)
    restored = None
    if _is_ours(d.get("forward")):
        del d["forward"]
        if inner is not None and inner is not getattr(type(module), "forward", None):
            restored = d["forward"] = _bind(module, inner)
    _unchain(module, restored or _bind(module, inner))


_SELF_CHECK = weakref.WeakKeyDictionary()
_MISSING = object()


def _compare(want, got, dtype):
    tol = 1e-4 if dtype in (torch.float32, torch.float64, None) else 3e-2
    if want is None or got is None:
        return None if want is got else "output structure changed"
    if torch.is_tensor(want) != torch.is_tensor(got):
        return "output structure changed"
    if not torch.is_tensor(want):
        if not isinstance(want, tuple) or len(want) != len(got):
            return "output structure changed"
        for a, b in zip(want, got):
            msg = _compare(a, b, dtype)
            if msg:
                return msg
        return None
    if want.shape != got.shape:
        return f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    if not torch.allclose(want.float(), got.float(), rtol = tol, atol = tol):
        return f"max diff {(want.float() - got.float()).abs().max().item():.3g}"
    return None


def _contract_for(out, lead):
    def one(o):
        if o is None:
            return "rows"
        if not torch.is_tensor(o) or o.dim() == 0:
            return "refuse"
        if tuple(o.shape[:len(lead)]) == tuple(lead):
            return "rows"
        if o.shape[0] == math.prod(lead):
            return "rows_flat"
        return "refuse"
    if torch.is_tensor(out):
        return (one(out),)
    if isinstance(out, tuple):
        return tuple(one(o) for o in out)
    return ("refuse",)


def _probed_forward(module):
    d = module.__dict__
    current = d.get("forward")
    if current is not None and not _is_ours(current):
        return current
    return d.get("_unsloth_forward")


def _compare_tiled(module, spec, inner, x, hidden, n_tiles, dtype):
    with torch.no_grad():
        want = inner(x)
        if spec is None:
            spec = TileSpec(type(module).__name__, output_contract = _contract_for(want, x.shape[:-1]), family = "opt-in")
        if spec.refuse:
            return spec, f"self-check: {spec.refuse}"
        state = _TileState(spec, TilePlan(mode = "rows", rows = 1), hidden, hidden, 2, 0.0)
        got = _run_tiled(module, state, x, n_tiles)
    msg = _compare(want, got, dtype)
    return spec, (None if msg is None else f"self-check failed: tiled != untiled ({msg})")


def _self_check(module, spec, n_tiles = 3):
    """Tiled == untiled on a small row block, in eval and training; returns (spec or None, reason)."""
    # Per instance and forward: instances can differ in config, and a later wrapper or kernel changes the forward.
    probed = _probed_forward(module)
    cached = _SELF_CHECK.get(module)
    if cached is not None and cached[0] is probed:
        return cached[1]
    result = (None, "self-check could not run")
    hidden = _infer_hidden(module)
    device = _device(module)
    dtype = _float_dtype(module) or torch.float32
    if hidden is None:
        result = (None, "self-check: cannot infer the input width")
    elif device is None or device.type == "meta":
        result = (None, "self-check: weights are not materialised")
    else:
        flags = [(m, m.training) for m in module.modules()]
        buffers = [(b, b.detach().clone()) for b in module.buffers()]
        d = module.__dict__
        current = d.get("forward")
        slot_was = d.get("_unsloth_forward", _MISSING)
        try:
            g = torch.Generator().manual_seed(1234)
            x = torch.randn(2, 7, hidden, generator = g).to(device = device, dtype = dtype)
            if current is not None and not _is_ours(current):
                d["_unsloth_forward"] = _adoptable(current)
            inner = _inner_forward(module)
            module.eval()
            spec, msg = _compare_tiled(module, spec, inner, x, hidden, n_tiles, dtype)
            if msg is None:
                module.train()
                for m in module.modules():
                    if isinstance(m, _DROPOUT_TYPES):
                        m.eval()
                spec, msg = _compare_tiled(module, spec, inner, x, hidden, n_tiles, dtype)
                if msg is not None and not msg.startswith("self-check: "):
                    msg += " in training mode"
            result = (spec, "") if msg is None else (None, msg)
        except Exception as e:
            result = (None, f"self-check raised {type(e).__name__}: {str(e)[:120]}")
        finally:
            for m, t in flags:
                m.training = t
            with torch.no_grad():
                for b, saved in buffers:
                    b.copy_(saved)
            if current is not None and not _is_ours(current):
                if slot_was is _MISSING:
                    d.pop("_unsloth_forward", None)
                else:
                    d["_unsloth_forward"] = slot_was
    _SELF_CHECK[module] = (probed, result)
    return result


def _matches(name, cls_name, patterns):
    low, cls_low = name.lower(), cls_name.lower()
    for p in patterns:
        p = p.lower().lstrip("!")
        if any(c in p for c in "*?["):
            if fnmatch(low, p) or fnmatch(cls_low, p):
                return True
        elif low.endswith(p) or cls_low == p:
            return True
    return False


def _env_patterns():
    return tuple(p.strip() for p in os.environ.get(_ENV_MODULES, "").split(",") if p.strip())


@dataclass
class TilingReport:
    plan: object = None
    tiled: list = field(default_factory = list)
    skipped: list = field(default_factory = list)

    @staticmethod
    def _counts(rows, key):
        counts = {}
        for row in rows:
            k = key(row)
            counts[k] = counts.get(k, 0) + 1
        return counts

    def line(self):
        mode = self.plan.spelling() if self.plan is not None else "off"
        skipped = self._counts(self.skipped, lambda r: (r[1], r[2]))
        skip_txt = "; ".join(f"{c} x{n}: {reason}" for (c, reason), n in skipped.items())
        if not self.tiled:
            text = f"Unsloth: Tiled MLP ({mode}) found no eligible modules in this model"
            return text + (f"; skipped {len(self.skipped)} [{skip_txt}]" if skipped else "")
        tiled = self._counts(self.tiled, lambda r: r[1])
        text = f"Unsloth: Tiled MLP ({mode}) on {len(self.tiled)} modules [" + \
            ", ".join(f"{c} x{n}" for c, n in tiled.items()) + "]"
        return text + (f"; skipped {len(self.skipped)} [{skip_txt}]" if skipped else "")

    def __str__(self):
        return self.line()


def _is_rank0():
    for var in ("RANK", "LOCAL_RANK"):
        if os.environ.get(var, "0") not in ("0", ""):
            return False
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


try:
    from torch.distributed.tensor import DTensor as _DTensor
except Exception:
    try:
        from torch.distributed._tensor import DTensor as _DTensor
    except Exception:
        _DTensor = None

_TP_MESH_NAMES = ("tp", "tensor", "model")


def _tensor_parallel(module):
    """True for a tensor-parallel DTensor parameter; FSDP2 meshes (dp / dp_shard / dp_replicate, Shard(0)) are not TP."""
    if _DTensor is None:
        return False
    for p in module.parameters():
        if not isinstance(p, _DTensor):
            continue
        names = getattr(p.device_mesh, "mesh_dim_names", None)
        if names:
            if any(any(t in str(n).lower() for t in _TP_MESH_NAMES) for n in names):
                return True
            continue
        placements = tuple(p.placements)
        if not all(getattr(pl, "is_shard", lambda *a: False)(0) for pl in placements):
            return True
    return False


def _select(model, plan, extra, force):
    out = []
    claimed = []
    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in claimed):
            continue
        cls_name = type(module).__name__
        if plan.modules and not _matches(name, cls_name, plan.modules):
            continue
        forced = bool(force) and _matches(name, cls_name, force)
        spec = lookup(module)
        opted = bool(plan.modules) or (bool(extra) and _matches(name, cls_name, extra))
        if spec is None and not (opted or forced):
            if name.lower().endswith(_OLD_SUFFIXES) and _sequence_mixer(module) is None:
                out.append((name, module, None, f"unknown class, opt in with {_ENV_MODULES}"))
                claimed.append(name + "." if name else "")
            continue
        claimed.append(name + "." if name else "")
        if forced:
            out.append((name, module, spec if spec is not None and not spec.refuse else
                        TileSpec(cls_name, family = "forced"), ""))
            continue
        if spec is not None and spec.refuse:
            out.append((name, module, None, spec.refuse))
            continue
        if os.environ.get(_ENV_TP, "0") != "1" and _tensor_parallel(module):
            out.append((name, module, None, f"tensor parallel (set {_ENV_TP}=1 to tile anyway)"))
            continue
        if spec is None:
            mixer = _sequence_mixer(module)
            if mixer is not None:
                out.append((name, module, None, f"sequence-mixing submodule {mixer}"))
                continue
            spec, reason = _self_check(module, None)
            out.append((name, module, spec, reason))
            continue
        out.append((name, module, spec, ""))
    return out


def install(model, mode = DEFAULT_MODE, modules = None, *, force = None, verbose = True):
    """Tile every eligible module of `model`; idempotent. `force` patterns skip the refusal and the self-check."""
    plan = parse_mode(mode)
    if plan is None:
        uninstall(model)
        return TilingReport(plan = None)
    if modules:
        modules = (modules,) if isinstance(modules, str) else tuple(modules)
        plan = replace(plan, modules = tuple(modules))
    force = tuple(p.lstrip("!") for p in (force or ()))
    force += tuple(p[1:] for p in plan.modules + _env_patterns() if p.startswith("!"))
    extra = _env_patterns()
    report = TilingReport(plan = plan)
    budget_cache = {}
    keep = set()
    for name, module, spec, reason in _select(model, plan, extra, force):
        if spec is None:
            report.skipped.append((name, type(module).__name__, reason))
            continue
        _attach(module, spec, plan, budget_cache)
        keep.add(id(module))
        report.tiled.append((name, type(module).__name__, spec.family))
    if report.tiled:
        stats = (sum(p.numel() * p.element_size() for p in model.parameters()), len(report.tiled))
        for _, module in model.named_modules():
            st = module.__dict__.get("_unsloth_tiling") if id(module) in keep else None
            if st is not None:
                st.model_stats = stats
    for _, module in model.named_modules():
        # "_unsloth_tiling" too: a wrapper installed after tiling (accelerate) hides the tiled forward from is_tiled.
        if id(module) not in keep and (is_tiled(module) or "_unsloth_tiling" in module.__dict__):
            _detach(module)
    previous = model.__dict__.get("_unsloth_tiling_report")
    model.__dict__["_unsloth_tiling_plan"] = (plan, force)
    model.__dict__["_unsloth_tiling_report"] = report.line()
    if verbose and _is_rank0() and previous != report.line():
        print(report.line(), flush = True)
    return report


def install_module(module, mode = DEFAULT_MODE, spec = None):
    """Tile one module regardless of the registry's verdict (the caller vouches for it)."""
    plan = parse_mode(mode)
    if plan is None:
        _detach(module)
        return module
    if spec is None:
        spec = lookup(module)
        if spec is None or spec.refuse:
            spec = TileSpec(type(module).__name__, family = "forced")
    _attach(module, spec, plan, {})
    return module


def uninstall(model):
    """Restore every tiled module's forward; returns how many were restored."""
    n = 0
    for _, module in list(model.named_modules()):
        if is_tiled(module) or "_unsloth_tiling" in module.__dict__:
            _detach(module)
            n += 1
    for _, module in model.named_modules():
        module.__dict__.pop("_unsloth_tiling_plan", None)
        module.__dict__.pop("_unsloth_tiling_report", None)
    return n


def revalidate(model, verbose = True):
    """Re-run `install` with the plan the model was tiled with (modules added since, e.g. by PEFT, are covered).
    Returns the report, or None when the model was never tiled."""
    if model is None or not hasattr(model, "named_modules"):
        return None
    for _, module in model.named_modules():
        stored = module.__dict__.get("_unsloth_tiling_plan")
        if stored is not None:
            plan, force = stored
            return install(module, plan, force = force, verbose = verbose)
    return None
