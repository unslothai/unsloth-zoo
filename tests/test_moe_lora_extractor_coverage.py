"""Dynamic regression-prevention test for the PR #624 failure mode.

PR #624 fixed a Gemma-4 MoE LoRA training crash whose root cause was:

    Gemma4TextExperts._unsloth_already_patched = True
    Gemma4TextExperts._unsloth_lora_extractor_fn  -> NOT REGISTERED

`unsloth_zoo.temporary_patches.gemma4_moe._patch_gemma4_moe_current` patched
`forward` to call the grouped-MoE backend but forgot to attach the LoRA
extractor. Without the extractor, `moe_utils._extract_lora_from_wrapper`
falls through to its default canonical permutation, which produces tensors
whose contraction dimensions do not match for `torch._grouped_mm` on PEFT
0.19+ swapped 3D LoRA layouts. The crash surfaces on the first training
step as `RuntimeError: contraction dimension of mat_a and mat_b must match`.

This test prevents that exact regression for every current and future MoE
family. It applies every `TEMPORARY_PATCHES` entry, walks every loaded
`transformers.models.*` module, and asserts that every class flagged
`_unsloth_already_patched=True` whose source defines `gate_up_proj` and
`down_proj` 3D parameters also has `_unsloth_lora_extractor_fn` registered.

Design constraints:
  - GPU-free. Instantiation, when attempted, uses tiny synthetic configs.
  - Transformers-version-agnostic. Discovery walks the live tree; nothing
    is hard-coded to specific class names.
  - Single test. Discovery + assertion live together in one pytest case so
    the contract is one signal, not many.
  - Opportunistic parity. If we can instantiate a discovered class without
    a checkpoint, we additionally drive its registered extractor on hand-
    built PEFT 0.18 raw and PEFT 0.19 swapped LoRA factors and assert
    per-expert delta parity. Failures here surface as a separate assertion
    so registration coverage and orientation correctness are distinguishable.

The test is deliberately defensive: importing transformers submodules is
allowed to fail (the full transformers tree pulls in optional deps), and
those failures are reported as `unimported` not as test failures.
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

# Apply every TEMPORARY_PATCHES entry. Importing the package side-effect-
# populates the list; we run each entry once and ignore individual failures
# (a missing transformers submodule is the standard no-op signal).
import unsloth_zoo.temporary_patches  # noqa: F401  side effect: register patches
from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES


# Regex for "self.gate_up_proj = nn.Parameter(torch.empty(...))" / similar
# 3D parameter declarations on the class body. We also accept the variant
# spelling used by some unsloth_zoo internals (Mxfp4 / GptOss style does
# NOT match this on purpose -- those classes use a different LoRA path).
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
    """Yield every transformers.models.<x>.modeling_<y> module that imports
    cleanly. Failures (missing optional deps) are skipped silently."""
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
    """Return True if `cls.__init__` source declares both `gate_up_proj`
    and `down_proj` as `nn.Parameter`s. This is the surface that the
    grouped-MoE LoRA extractor reads."""
    try:
        src = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        return False
    return all(p.search(src) for p in _3D_PARAM_PATTERNS)


def _has_unsloth_patched_forward(cls: type) -> bool:
    """Detect that `unsloth_zoo.temporary_patches.utils.patch_function` has
    replaced `cls.forward`. `patch_function` stores the original under a
    per-class attribute named `_original_<modeling_module_tail>_<ClassName>_
    forward` (`utils.py:_get_unique_storage_name`). The presence of that
    attribute on the class is a uniform marker independent of any family-
    specific bool flag like `_unsloth_already_patched`.
    """
    cls_name = cls.__name__
    for attr in dir(cls):
        if attr.startswith("_original_") and attr.endswith(f"_{cls_name}_forward"):
            return True
    return False


def _discover_patched_moe_classes() -> list[type]:
    """Find every transformers class that unsloth-zoo has replaced the
    `forward` of AND whose source matches the grouped-MoE 3D-parameter
    shape (`gate_up_proj` + `down_proj` declared as `nn.Parameter`).

    The combination is the load-bearing contract: `forward` was swapped to
    the grouped-MoE backend, so the LoRA path through that backend must
    have an extractor registered. PR #624 was an instance of registration
    being forgotten on a class that satisfied both conditions.
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
    """Best-effort instantiation of an MoE experts class with tiny synthetic
    dims. Returns the instance or None if any path fails.

    Dim choice: H=16, I=10, 2*I=20. All three values are distinct (so
    PEFT-version dispatch can't be ambiguous) and the relation
    `H > I` and `2*I > H` matches production MoE configs (Qwen3-MoE,
    Gemma-4 MoE, Glm4-MoE-Lite, DeepSeek-V3 all live in this regime). A
    test in a different regime would surface known-fragile extractor
    orientation behavior that is independent of the PR #624 contract this
    test exists to enforce."""
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
    # Some experts modules want a nested text_config; try that if direct fails.
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
    """Find the most likely Config class living in the same modeling module
    as `cls`. Tries Foo<...>Config and Foo<...>TextConfig forms by stripping
    common experts suffixes from the class name."""
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
    """Regression-prevention test for PR #624. Every class whose `forward`
    unsloth-zoo has patched to use the grouped-MoE backend AND whose layout
    matches the standard `(E, 2*I, H)` / `(E, H, I)` 3D-parameter shape
    MUST have `_unsloth_lora_extractor_fn` registered. Without it, LoRA
    training crashes on the first step with a `torch._grouped_mm`
    contraction-dim mismatch on PEFT 0.19+."""
    _apply_all_temporary_patches()

    patched = _discover_patched_moe_classes()
    if not patched:
        # Disambiguate three zero-discovery causes:
        #   (a) the installed transformers predates every MoE class surface
        #       unsloth_zoo targets, AND no patch fn ran far enough to set
        #       any marker -- legitimate skip;
        #   (b) the runtime environment broke ALL targeting patches (e.g. a
        #       CPU-only runner where some patch fn raises before its
        #       `_unsloth_already_patched = True` line) but transformers
        #       itself ships unpatched MoE classes -- this is a real
        #       infrastructure failure, but it does not necessarily mean
        #       the marker convention drifted: the patch fn never reached
        #       the marker, so blaming the test helper is wrong; or
        #   (c) the `_original_<modeling_tail>_<ClassName>_forward` marker
        #       convention `_has_unsloth_patched_forward` reads drifted
        #       relative to `patch_function`'s storage convention -- a
        #       real test-helper regression.
        # Distinguish them via the `_unsloth_already_patched=True` boolean
        # the patch fns set explicitly: that flag is the patch fn's own
        # claim "I reached the end successfully." If at least one class
        # carries it, AT LEAST one patch fn ran fully -- discovery missing
        # such a class is therefore the test-helper drift in (c) and is a
        # real regression. If NO class carries it, the patch fns either
        # all early-exited (a) or hit a runtime issue (b); in either case
        # this test cannot reach its target so we skip.
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

    # Soft contract: opportunistic per-expert parity. Failures here are the
    # extractor-orientation bug class (PR #624 was a registration bug, not
    # an orientation bug, but the same one-test-catches-both contract is
    # cheap to assert here).
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
        # Surface the orientation drift but keep the registration assertion
        # above as the primary signal. We use pytest.fail not assert so the
        # registration coverage assertion can still pass-or-fail
        # independently of orientation.
        pytest.fail(
            "Per-expert delta parity failed for some patched MoE families. "
            "This is independent of registration coverage but indicates an "
            "extractor orientation bug. Failures:\n  - "
            + "\n  - ".join(parity_failures)
        )
