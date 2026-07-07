"""Dynamic regression-prevention test for the PR #624 failure mode.

PR #624 fixed a Gemma-4 MoE LoRA crash: the patch set
`Gemma4TextExperts._unsloth_already_patched=True` but never registered
`_unsloth_lora_extractor_fn`. Without the extractor the default canonical
permutation yields tensors whose contraction dims mismatch for
`torch._grouped_mm` on PEFT 0.19+ swapped 3D LoRA layouts, crashing on the
first training step with `RuntimeError: contraction dimension of mat_a and
mat_b must match`.

This test applies every `TEMPORARY_PATCHES` entry, walks loaded
`transformers.models.*`, and asserts every patched class with 3D
`gate_up_proj`/`down_proj` params also has `_unsloth_lora_extractor_fn`.
GPU-free, version-agnostic (walks the live tree), single test. When a
class can be instantiated it also drives per-expert delta parity on PEFT
0.18 raw and 0.19 swapped factors, surfaced as a separate assertion.
Importing transformers submodules may fail (optional deps); such failures
are skipped, not test failures.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

# Importing the package side-effect-populates TEMPORARY_PATCHES.
import unsloth_zoo.temporary_patches  # noqa: F401  side effect: register patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES


# Match 3D `self.gate_up_proj`/`self.down_proj = nn.Parameter(...)` class
# declarations. Mxfp4 / GptOss style is intentionally not matched (different
# LoRA path).
_3D_PARAM_PATTERNS = [
    re.compile(r"self\.gate_up_proj\s*=\s*nn\.Parameter\("),
    re.compile(r"self\.down_proj\s*=\s*nn\.Parameter\("),
]


def _apply_all_temporary_patches() -> None:
    for fn in TEMPORARY_PATCHES:
        try:
            fn()
        except Exception:  # noqa: BLE001 - missing modules are fine here
            pass


def _iter_modeling_modules():
    """Yield every cleanly-importable transformers.models.*.modeling_* module."""
    import transformers.models as tm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sub in pkgutil.iter_modules(tm.__path__):
            if sub.ispkg is False:
                continue
            try:
                pkg = importlib.import_module(f"transformers.models.{sub.name}")
            except Exception:
                continue
            for child in pkgutil.iter_modules(pkg.__path__):
                if not child.name.startswith("modeling_"):
                    continue
                try:
                    yield importlib.import_module(
                        f"transformers.models.{sub.name}.{child.name}",
                    )
                except Exception:
                    continue


def _looks_like_grouped_moe_experts(cls: type) -> bool:
    """True if `cls.__init__` declares both `gate_up_proj` and `down_proj`
    as `nn.Parameter`s (the grouped-MoE LoRA extractor's surface)."""
    try:
        src = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        return False
    return all(p.search(src) for p in _3D_PARAM_PATTERNS)


def _has_unsloth_patched_forward(cls: type) -> bool:
    """True when `patch_function` has replaced `cls.forward`. It stores the
    original under `_original_<modeling_tail>_<ClassName>_forward`
    (`utils.py:_get_unique_storage_name`); that attribute is a uniform marker
    independent of family-specific flags like `_unsloth_already_patched`.
    """
    cls_name = cls.__name__
    for attr in dir(cls):
        if attr.startswith("_original_") and attr.endswith(f"_{cls_name}_forward"):
            return True
    return False


def _discover_patched_moe_classes() -> list[type]:
    """Find every class whose `forward` unsloth-zoo replaced AND whose source
    matches the grouped-MoE 3D shape (`gate_up_proj` + `down_proj`
    nn.Parameters). That combination requires a registered extractor; PR #624
    was a class satisfying both where registration was forgotten.
    """
    seen: set[type] = set()
    out: list[type] = []
    for modeling in _iter_modeling_modules():
        for name, obj in inspect.getmembers(modeling, inspect.isclass):
            if not isinstance(obj, type):
                continue
            if obj in seen:
                continue
            seen.add(obj)
            if getattr(obj, "__module__", None) != modeling.__name__:
                continue
            if not _has_unsloth_patched_forward(obj):
                continue
            if not _looks_like_grouped_moe_experts(obj):
                continue
            out.append(obj)
    return out


# --------------------------------------------------------------------------
# Opportunistic parity helpers
# --------------------------------------------------------------------------


class _StubWrapper:
    def __init__(self, parameter_name: str, base: Any, peft_swapped: bool):
        self.parameter_name = parameter_name
        self._base = base
        self._did_swap_in_out_features = peft_swapped

    def get_base_layer(self):
        return self._base


def _try_instantiate_experts(cls: type):
    """Best-effort instantiate an MoE experts class with tiny synthetic dims;
    returns the instance or None.

    Dims H=16, I=10, 2*I=20 are all distinct (no ambiguous PEFT dispatch) and
    keep `H > I`, `2*I > H` to match production MoE configs (Qwen3-MoE,
    Gemma-4 MoE, Glm4-MoE-Lite, DeepSeek-V3). Other regimes surface fragile
    orientation behavior unrelated to the PR #624 contract."""
    cfg_cls = _find_sibling_config(cls)
    if cfg_cls is None:
        return None
    overrides = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 12,
        "moe_intermediate_size": 10,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "n_routed_experts": 4,
        "num_local_experts": 4,
        "top_k_experts": 2,
        "first_k_dense_replace": 1,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "rms_norm_eps": 1e-6,
        "hidden_activation": "gelu_pytorch_tanh",
    }
    try:
        sig = inspect.signature(cfg_cls.__init__)
        kwargs = {k: v for k, v in overrides.items() if k in sig.parameters}
        cfg = cfg_cls(**kwargs)
    except Exception:
        return None
    # Some experts modules want a nested text_config; try it if direct fails.
    for cfg_arg in (cfg, getattr(cfg, "text_config", None)):
        if cfg_arg is None:
            continue
        try:
            return cls(cfg_arg)
        except Exception:
            continue
        try:
            return cls(cfg_arg, layer_idx=0)
        except Exception:
            continue
    return None


def _find_sibling_config(cls: type):
    """Find the likely Config class in `cls`'s modeling module, trying
    Foo*Config / Foo*TextConfig after stripping common experts suffixes."""
    mod = importlib.import_module(cls.__module__)
    base = cls.__name__
    for suffix in ("Experts", "NaiveMoe", "MoEBlock", "MoeBlock", "MoE", "Moe"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    candidates = [
        base + "TextConfig",
        base + "Config",
        cls.__name__.split("Experts")[0] + "TextConfig",
        cls.__name__.split("Experts")[0] + "Config",
    ]
    for cand in candidates:
        cfg = getattr(mod, cand, None)
        if isinstance(cfg, type):
            return cfg
    return None


def _read_dims(experts) -> tuple[int, int] | None:
    H = getattr(experts, "hidden_dim", None)
    I = getattr(experts, "intermediate_dim", None)
    if H is None or I is None:
        return None
    return int(H), int(I)


def _parity_one(extractor, experts, name: str, in_dim: int, out_dim: int,
                peft_swap: bool, E: int = 4, R: int = 3) -> tuple[bool, str]:
    torch.manual_seed(0)
    if peft_swap:
        wA = torch.randn(E * R, out_dim)
        wB = torch.randn(in_dim, E * R)
    else:
        wA = torch.randn(E * R, in_dim)
        wB = torch.randn(out_dim, E * R)
    try:
        res = extractor(_StubWrapper(name, experts, peft_swap), wA, wB, 1.0, E)
    except Exception as ex:  # noqa: BLE001
        return False, f"extractor raised: {ex!r}"
    if res is None:
        return False, "extractor returned None"
    first, second = res[0], res[1]
    if tuple(first.shape) != (E, in_dim, R):
        return False, f"first.shape {tuple(first.shape)} != ({E},{in_dim},{R})"
    if tuple(second.shape) != (E, R, out_dim):
        return False, f"second.shape {tuple(second.shape)} != ({E},{R},{out_dim})"
    x = torch.randn(5, in_dim)
    for e in range(E):
        Ae = wA[e * R : (e + 1) * R]
        Be = wB[:, e * R : (e + 1) * R]
        naive = (x @ Be @ Ae) if peft_swap else (x @ Ae.T @ Be.T)
        via = (x @ first[e]) @ second[e]
        if not torch.allclose(via, naive, atol=1e-4, rtol=1e-4):
            return False, f"per-expert delta mismatch on expert {e}"
    return True, "ok"


# --------------------------------------------------------------------------
# The single test
# --------------------------------------------------------------------------


def test_every_patched_moe_experts_class_has_lora_extractor():
    """Regression test for PR #624: every class whose `forward` unsloth-zoo
    patched to the grouped-MoE backend with `(E, 2*I, H)`/`(E, H, I)` 3D
    params MUST register `_unsloth_lora_extractor_fn`, else LoRA crashes on
    step 1 with a `torch._grouped_mm` contraction-dim mismatch on PEFT
    0.19+."""
    _apply_all_temporary_patches()

    patched = _discover_patched_moe_classes()
    if not patched:
        # Zero discovery has three causes: (a) transformers predates every
        # targeted MoE class and no patch fn set a marker -- legit skip;
        # (b) the runtime broke ALL targeting patches before their
        # `_unsloth_already_patched = True` line -- infra failure, not marker
        # drift; (c) the `_original_..._forward` convention
        # `_has_unsloth_patched_forward` reads drifted from `patch_function`
        # -- a real test-helper regression.
        # Disambiguate via `_unsloth_already_patched=True`: if any class
        # carries it, a patch fn ran fully so missing discovery is (c), a real
        # regression. If none carries it, we're in (a)/(b) and skip.
        already = []
        for modeling in _iter_modeling_modules():
            for _name, cls in inspect.getmembers(modeling, inspect.isclass):
                if not isinstance(cls, type):
                    continue
                if getattr(cls, "__module__", None) != modeling.__name__:
                    continue
                if getattr(cls, "_unsloth_already_patched", False) is True:
                    already.append(f"{cls.__module__}.{cls.__name__}")
        if already:
            raise AssertionError(
                "Discovery produced zero patched MoE classes, but at "
                f"least one class carries `_unsloth_already_patched=True` "
                f"({already[:5]}). That means a patch fn ran to the end "
                "and announced itself, but `_has_unsloth_patched_forward` "
                "(which reads `_original_<modeling_tail>_<ClassName>_"
                "forward`) does not see it. Either `patch_function` "
                "stopped storing the original under that attribute name "
                "or the test helper drifted -- realign one or the other "
                "before re-converting this branch into a skip."
            )
        pytest.skip(
            "No patched MoE classes discovered AND no class in "
            "transformers.models.* carries `_unsloth_already_patched=True`. "
            "Either transformers is older than every grouped-MoE family "
            "unsloth_zoo targets, or every targeting patch fn raised "
            "before its `_unsloth_already_patched = True` line on this "
            "runtime (e.g. CPU-only spoof environments hit by some "
            "non-spoofed CUDA call inside a patch). Cells running a "
            "fully supported runtime still exercise the real assertion."
        )

    # Hard contract: extractor must be registered on every patched class.
    missing = [
        f"{c.__module__}.{c.__name__}"
        for c in patched
        if getattr(c, "_unsloth_lora_extractor_fn", None) is None
    ]
    assert not missing, (
        "The following transformers MoE classes have `_unsloth_already_patched"
        "=True` but no `_unsloth_lora_extractor_fn` registered. This is the "
        "exact regression PR #624 fixed for Gemma-4: LoRA training will crash "
        "on PEFT 0.19+ with `RuntimeError: contraction dimension of mat_a and "
        "mat_b must match` on the first training step. Register the extractor "
        f"in the patch function. Offenders: {missing}"
    )

    # Soft contract: opportunistic per-expert parity catches the
    # extractor-orientation bug class (cheap to assert alongside).
    parity_failures = []
    for cls in patched:
        experts = _try_instantiate_experts(cls)
        if experts is None:
            continue
        dims = _read_dims(experts)
        if dims is None:
            continue
        H, I = dims
        ext = cls._unsloth_lora_extractor_fn
        for proj_name, in_dim, out_dim in [
            ("gate_up_proj", H, 2 * I),
            ("down_proj", I, H),
        ]:
            for peft_swap in (False, True):
                ok, msg = _parity_one(ext, experts, proj_name, in_dim, out_dim, peft_swap)
                if not ok:
                    parity_failures.append(
                        f"{cls.__name__}.{proj_name} swap={peft_swap}: {msg}"
                    )

    if parity_failures:
        # pytest.fail (not assert) keeps orientation separate from the
        # primary registration-coverage assertion above.
        pytest.fail(
            "Per-expert delta parity failed for some patched MoE families. "
            "This is independent of registration coverage but indicates an "
            "extractor orientation bug. Failures:\n  - "
            + "\n  - ".join(parity_failures)
        )
